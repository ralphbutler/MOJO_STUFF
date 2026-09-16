#!/usr/bin/env python3
"""air2spir.py — retarget a Mojo Metal (AIR) GPU-kernel .ll to SPIR for llvm-spirv.

G2 route: Mojo's Metal backend (`mojo build --emit asm --target-accelerator apple-m4`)
writes each GPU kernel as a small, optimized LLVM IR sidecar (`*.ll`, triple air64).
Metal's AIR descends from SPIR and uses the same address-space numbering
(1 = global/device buffer, 2 = constant, 3 = threadgroup/local), so the kernel
body can be reused as-is. This script only rewrites the Metal-specific shell:

  * target triple/datalayout      -> spir64-unknown-unknown
  * kernel `define`               -> `define spir_kernel void @<name>(...)`
  * Metal thread-position params  -> OpenCL builtins called at entry
        threadgroup_position_in_grid   -> get_group_id(d)
        thread_position_in_threadgroup -> get_local_id(d)
        threads_per_threadgroup        -> get_local_size(d)
        threads_per_grid               -> get_global_size(d)
        thread_index_in_simdgroup      -> get_sub_group_local_id()
  * `call @air.wg.barrier(...)`    -> OpenCL barrier(CLK_LOCAL_MEM_FENCE)  (G3)
  * `call @air.{max,min}.{s,u}.iN(a, b)` -> inline `icmp` + `select`  (G4)
      (04b's matmul/colsum kernels guard the K loop with max(K, 0). Inlining keeps
       the module free of external symbols; no OpenCL import for IGC to resolve.)
  * `internal addrspace(3) global` (Mojo AddressSpace.SHARED stack_allocation)
                                  -> extra `ptr addrspace(3)` kernel args, i.e. OpenCL
                                     __local args; Level Zero allocates them when the
                                     host sets the arg with a size and a NULL value (G3)
  * Metal attributes / !air metadata / `declare @air.*` -> removed
  * optionally `ptr addrspace(2)` (constant buffers) -> `ptr addrspace(1)`
  * no-op `%r = bitcast ptr addrspace(K) %x to ptr addrspace(K)` -> `getelementptr i8, ..., i64 0`
      (Mojo's Metal IR has hundreds; llvm-spirv's SPIRVLowerBitCastToNonStandardType pass
       aborts on them next to wide vectors, e.g. 03c's <64 x float> register tiles. A GEP
       keeps value numbering intact.)
  * vectors with non-standard SPIR-V widths (not 2/3/4/8/16), e.g. 03c's <64 x float>
      register tiles, are scalarized: phi -> one scalar phi per element; insertelement /
      extractelement chains are resolved away. (llvm-spirv aborts on extractelement from
      such vectors unless SPV_EXT_long_vector is enabled, which IGC may not support.)
  * writes an args manifest (--args-out) the host reads to bind arguments:
        arg <i> buffer|const          (pointer to a TileTensor/scalar holder)
        arg <i> local <bytes>         (shared local memory)

It refuses (exit 2) on anything it doesn't understand yet — calls into @air.*
intrinsics, metadata references inside function bodies, etc. — rather than
guessing. Grow it rung by rung (G3 matmul will need threadgroup memory).

Usage:
  python3 air2spir.py IN.ll OUT.ll --name vecadd [--const-as-global] [--args-out OUT.args]
"""

import argparse
import re
import sys

SPIR_TRIPLE = "spir64-unknown-unknown"
SPIR_DATALAYOUT = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-n8:16:32:64"

# Metal vec3 thread params -> OpenCL builtin (Itanium-mangled, takes i32 dim, returns i64)
VEC3_BUILTINS = {
    "threadgroup_position_in_grid": "_Z12get_group_idj",
    "thread_position_in_threadgroup": "_Z12get_local_idj",
    "threads_per_threadgroup": "_Z14get_local_sizej",
    "threads_per_grid": "_Z15get_global_sizej",
}
NOOP_BITCAST = re.compile(r"^(\s*%[A-Za-z0-9_.]+) = bitcast (ptr(?: addrspace\(\d+\))?) (%[A-Za-z0-9_.]+|@[A-Za-z0-9_.$]+) to \2$")
BARRIER_FN = "_Z7barrierj"          # OpenCL C: void barrier(cl_mem_fence_flags)
# air.max.s.i64(a, b) and friends: integer min/max, inlined as icmp+select (G4)
AIR_MINMAX = re.compile(
    r"^(\s*)(%[-\w.$]+) = call (i\d+) @air\.(max|min)\.([su])\.i\d+\("
    r"i\d+ ([^,]+), i\d+ ([^)]+)\)(?:\s*#\d+)?$")
MINMAX_PRED = {("max", "s"): "sgt", ("max", "u"): "ugt",
               ("min", "s"): "slt", ("min", "u"): "ult"}
CLK_LOCAL_MEM_FENCE = 1
ELEM_BYTES = {"float": 4, "double": 8, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "half": 2}

# Metal scalar thread param -> OpenCL builtin returning i32
SCALAR_BUILTINS = {
    "thread_index_in_simdgroup": "_Z22get_sub_group_local_idv",
}


STD_VEC_WIDTHS = {2, 3, 4, 8, 16}
VEC_RE = re.compile(r"<(\d+) x ([a-z0-9]+)>")
TOK_RE = re.compile(r"%[A-Za-z0-9_.]+")


def _zero(ty):
    return "0.000000e+00" if ty in ("float", "double", "half") else "0"


def scalarize_long_vectors(text):
    """Scalarize phi/insertelement/extractelement on non-standard-width vectors.

    Only these three instructions (with constant indices) are supported; anything
    else touching such a vector is an error. Unnamed values are renamed %N -> %uN
    first, so dropping instructions cannot break LLVM's sequential numbering.
    """
    if not any(int(n) not in STD_VEC_WIDTHS for n, _ in VEC_RE.findall(text)):
        return text
    sys.setrecursionlimit(100000)
    lines = text.split("\n")
    d0 = next(i for i, l in enumerate(lines) if l.startswith("define "))
    d1 = next(i for i in range(d0, len(lines)) if lines[i] == "}")

    # --- rename unnamed values and labels inside the function ---
    n_unnamed_params = len(re.findall(r"%\d+(?![A-Za-z0-9_.])", lines[d0].split("{")[0]))
    def ren(l):
        l = re.sub(r"%(\d+)(?![A-Za-z0-9_.])", r"%u\1", l)
        return re.sub(r"^(\d+):", r"u\1:", l)
    body = [ren(l) for l in lines[d0:d1 + 1]]
    body.insert(1, f"u{n_unnamed_params}:")      # explicit label for the entry block

    def is_long(ty_n):
        return int(ty_n) not in STD_VEC_WIDTHS

    phis, inserts, extracts = {}, {}, {}
    for l in body:
        m = re.match(r"^\s*(%[\w.]+) = phi <(\d+) x (\w+)> (.*)$", l)
        if m and is_long(m.group(2)):
            phis[m.group(1)] = (int(m.group(2)), m.group(3), re.findall(r"\[ ([^,]+), (%[\w.]+) \]", m.group(4)))
            continue
        m = re.match(r"^\s*(%[\w.]+) = insertelement <(\d+) x (\w+)> ([^,]+), \w+ ([^,]+), i(?:32|64) (\d+)$", l)
        if m and is_long(m.group(2)):
            inserts[m.group(1)] = (m.group(4).strip(), m.group(5).strip(), int(m.group(6)), m.group(3))
            continue
        m = re.match(r"^\s*(%[\w.]+) = extractelement <(\d+) x (\w+)> ([^,]+), i(?:32|64) (\d+)$", l)
        if m and is_long(m.group(2)):
            extracts[m.group(1)] = (m.group(4).strip(), int(m.group(5)))
            continue
        for n, _ in VEC_RE.findall(l):
            if is_long(n):
                die(f"scalarize: unsupported use of a long vector: {l.strip()[:160]}")

    memo_v, memo_s = {}, {}
    def scalar(opnd):
        if opnd in extracts:
            if opnd not in memo_s:
                v, i = extracts[opnd]
                memo_s[opnd] = scalar(elem(v, i))
            return memo_s[opnd]
        return opnd
    def elem(v, j, ty="float"):
        if v in ("zeroinitializer",):
            return _zero(ty)
        if v in ("poison", "undef"):
            return "poison"
        key = (v, j)
        if key in memo_v:
            return memo_v[key]
        if v in phis:
            r = f"{v}.e{j}"
        elif v in inserts:
            w, sval, idx, ety = inserts[v]
            r = scalar(sval) if j == idx else elem(w, j, ety)
        else:
            die(f"scalarize: vector {v} is not a phi/insertelement (argument or load?)")
        memo_v[key] = r
        return r

    out = []
    for l in body:
        m = re.match(r"^(\s*)(%[\w.]+) = phi <", l)
        if m and m.group(2) in phis:
            n, ety, incoming = phis[m.group(2)]
            for j in range(n):
                inc = ", ".join(f"[ {scalar(elem(val.strip(), j, ety))}, {bb} ]" for val, bb in incoming)
                out.append(f"{m.group(1)}{m.group(2)}.e{j} = phi {ety} {inc}")
            continue
        m = re.match(r"^\s*(%[\w.]+) = (?:insertelement|extractelement) ", l)
        if m and (m.group(1) in inserts or m.group(1) in extracts):
            continue
        def sub(mt):
            t = mt.group(0)
            if t in phis or t in inserts:
                die(f"scalarize: long vector {t} used outside phi/insert/extract: {l.strip()[:160]}")
            return scalar(t)
        out.append(TOK_RE.sub(sub, l) if not l.startswith("define ") else l)

    return "\n".join(lines[:d0] + out + lines[d1 + 1:])


def die(msg):
    print(f"air2spir: ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def split_params(s):
    """Split a parameter list on top-level commas (ignoring (), <>, {}, [] and quotes)."""
    out, depth, cur, in_q = [], 0, [], False
    for ch in s:
        if ch == '"':
            in_q = not in_q
        elif not in_q:
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>":
                depth -= 1
            elif ch == "," and depth == 0:
                out.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def param_type_and_name(p):
    """'ptr addrspace(1) noundef "x" %0' -> ('ptr addrspace(1)', '%0')."""
    m = re.match(r"^(ptr(?: addrspace\(\d+\))?|<\d+ x [a-z0-9]+>|[a-z][a-z0-9]*|\{.*\})\s", p)
    name = p.split()[-1]
    if not m or not name.startswith("%"):
        die(f"cannot parse parameter: {p!r}")
    return m.group(1), name


def find_matching_paren(s, open_idx):
    depth, in_q = 0, False
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == '"':
            in_q = not in_q
        elif not in_q:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
    die("unbalanced parentheses in define line")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--name", required=True, help="SPIR-V kernel entry-point name")
    ap.add_argument("--const-as-global", action="store_true",
                    help="rewrite ptr addrspace(2) (Metal constant buffers) to addrspace(1)")
    ap.add_argument("--args-out", help="write the kernel argument manifest here")
    ap.add_argument("--no-scalarize", action="store_true",
                    help="leave non-standard-width vectors alone")
    ap.add_argument("--keep-bitcasts", action="store_true",
                    help="do not rewrite no-op pointer bitcasts (G2/G3.1 behaviour)")
    ap.add_argument("--barrier", choices=["opencl", "drop"], default="opencl",
                    help="lower Metal barriers to OpenCL barrier() (default) or drop them")
    args = ap.parse_args()

    lines = open(args.inp).read().splitlines()

    # --- which attribute groups mark Metal kernels ---
    kernel_attr_groups = set()
    for ln in lines:
        m = re.match(r"^attributes (#\d+) = \{(.*)\}$", ln)
        if m and '"metal.kernel"="true"' in m.group(2):
            kernel_attr_groups.add(m.group(1))

    # --- threadgroup (shared local) memory globals -> kernel args ---
    slm = []  # (global name, bytes)
    for ln in lines:
        m = re.match(r"^(@[^\s=]+) = internal addrspace\(3\) global \[(\d+) x ([a-z0-9]+)\] (?:undef|poison|zeroinitializer)", ln)
        if m:
            if m.group(3) not in ELEM_BYTES:
                die(f"unsupported shared-memory element type: {m.group(3)}")
            slm.append((m.group(1), int(m.group(2)) * ELEM_BYTES[m.group(3)]))
        elif re.match(r"^@\S+ = ", ln):
            die(f"unsupported module-level global: {ln[:120]}")
    slm_pats = [(re.compile(re.escape(g) + r"(?![A-Za-z0-9_.$])"), f"%slm{i}") for i, (g, _) in enumerate(slm)]
    manifest = []

    def referenced(name):
        """True if %name is used anywhere except its own definition in a define line."""
        pat = re.compile(re.escape(name) + r"(?![A-Za-z0-9_.])")
        return any(pat.search(l) for l in lines if not l.lstrip().startswith("define "))

    out, used_builtins, n_kernels, in_body = [], set(), 0, False
    n_minmax = 0
    for ln in lines:
        s = ln.strip()

        # module-level lines to drop or replace
        if s.startswith("target datalayout"):
            out.append(f'target datalayout = "{SPIR_DATALAYOUT}"')
            continue
        if s.startswith("target triple"):
            out.append(f'target triple = "{SPIR_TRIPLE}"')
            continue
        if s.startswith("source_filename"):
            continue
        if re.match(r"^@\S+ = internal addrspace\(3\) global", s):
            continue
        if re.match(r"^declare .*@air\.", s):
            continue
        if s.startswith("attributes #"):
            continue
        if s.startswith("!"):
            continue  # all named + numbered metadata (air.*, llvm.ident, module flags)

        if s.startswith("define "):
            open_idx = ln.index("(", ln.index("@"))
            close_idx = find_matching_paren(ln, open_idx)
            head, params_s, tail = ln[:open_idx], ln[open_idx + 1:close_idx], ln[close_idx + 1:]
            groups = set(re.findall(r"#\d+", tail))
            is_kernel = bool(groups & kernel_attr_groups)

            new_params, entry = [], []
            for p in split_params(params_s):
                ty, name = param_type_and_name(p)
                bare = name[1:]
                if (bare in VEC3_BUILTINS or bare in SCALAR_BUILTINS) and not referenced(name):
                    continue  # unused Metal thread param: drop, no builtin call
                if bare in VEC3_BUILTINS:
                    fn = VEC3_BUILTINS[bare]
                    used_builtins.add(("i64", fn, "i32"))
                    prev = "poison"
                    for d in range(3):
                        entry.append(f"  %{bare}.d{d}.64 = call spir_func i64 @{fn}(i32 {d})")
                        entry.append(f"  %{bare}.d{d} = trunc i64 %{bare}.d{d}.64 to i32")
                        tgt = name if d == 2 else f"%{bare}.v{d}"
                        entry.append(f"  {tgt} = insertelement <3 x i32> {prev}, i32 %{bare}.d{d}, i64 {d}")
                        prev = tgt
                elif bare in SCALAR_BUILTINS:
                    fn = SCALAR_BUILTINS[bare]
                    used_builtins.add(("i32", fn, ""))
                    entry.append(f"  {name} = call spir_func i32 @{fn}()")
                else:
                    new_params.append(f"{ty} {name}")
                    if is_kernel:
                        kind = "const" if "addrspace(2)" in ty else "buffer"
                        manifest.append(f"arg {len(new_params) - 1} {kind}")

            if is_kernel:
                for i, (_, nbytes) in enumerate(slm):
                    new_params.append(f"ptr addrspace(3) %slm{i}")
                    manifest.append(f"arg {len(new_params) - 1} local {nbytes}")
                n_kernels += 1
                new_head = f"define spir_kernel void @{args.name}"
                if not re.match(r"^define (?:[a-z_]+ )*void @", head):
                    die("kernel does not return void")
            else:
                new_head = re.sub(r"^define ", "define spir_func ", head)
            tail = re.sub(r"\s*#\d+", "", tail)          # drop attribute-group refs
            out.append(f"{new_head}({', '.join(new_params)}){tail}")
            out.extend(entry)
            in_body = True
            continue

        if in_body:
            for pat, repl in slm_pats:
                ln = pat.sub(repl, ln)
            if not args.keep_bitcasts:
                ln = NOOP_BITCAST.sub(r"\1 = getelementptr i8, \2 \3, i64 0", ln)
            s = ln.strip()
            mb = re.match(r"^(\s*)call void @air\.wg\.barrier\([^)]*\)(?:\s*#\d+)?$", ln)
            if mb:
                if args.barrier == "opencl":
                    used_builtins.add(("void", BARRIER_FN, "i32"))
                    out.append(f"{mb.group(1)}call spir_func void @{BARRIER_FN}(i32 {CLK_LOCAL_MEM_FENCE})")
                continue
            mm = AIR_MINMAX.match(ln)
            if mm:
                ind, dst, ty, op, sign, a, b = mm.groups()
                tmp = f"%.airminmax{n_minmax}"
                n_minmax += 1
                out.append(f"{ind}{tmp} = icmp {MINMAX_PRED[(op, sign)]} {ty} {a}, {b}")
                out.append(f"{ind}{dst} = select i1 {tmp}, {ty} {a}, {ty} {b}")
                continue
            if s == "}":
                in_body = False
            elif "@air." in s:
                die(f"unsupported Metal intrinsic call in body: {s}")
            elif re.search(r"![A-Za-z0-9_.]", s):
                die(f"metadata reference in body not supported yet: {s}")

        out.append(ln)

    if n_kernels != 1:
        die(f"expected exactly 1 Metal kernel, found {n_kernels}")

    text = "\n".join(out)
    if args.const_as_global:
        text = text.replace("ptr addrspace(2)", "ptr addrspace(1)")
    if not args.no_scalarize:
        text = scalarize_long_vectors(text)

    decls = [f"declare spir_func {ret} @{fn}({arg})" for ret, fn, arg in sorted(used_builtins)]
    # put declarations right after the triple line
    text = text.replace(f'target triple = "{SPIR_TRIPLE}"',
                        f'target triple = "{SPIR_TRIPLE}"\n\n' + "\n".join(decls), 1)
    if args.args_out:
        open(args.args_out, "w").write(f"kernel {args.name}\n" + "\n".join(manifest) + "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    open(args.out, "w").write(text.rstrip() + "\n")
    print(f"air2spir: wrote {args.out} (kernel @{args.name}, builtins: "
          f"{', '.join(fn for _, fn, _ in sorted(used_builtins)) or 'none'}; "
          f"shared-local args: {len(slm)})")
    for m in manifest:
        print(f"  {m}")


if __name__ == "__main__":
    main()

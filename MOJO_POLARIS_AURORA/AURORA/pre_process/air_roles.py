#!/usr/bin/env python3
"""air_roles.py — identify Mojo Metal (AIR) kernel sidecars and map them to stable role names.

`mojo build --emit asm --target-accelerator <apple>` writes one .ll per GPU kernel INSTANTIATION,
named with a mangled hash:

    host__04b_train_mlp_gpu_matmul_kernel6A6A_1c8acdaa86cf136f.ll

The hash is not a role. A generic kernel instantiated per TensorLayout produces several sidecars
with the same base name, and the host has to know which is which — in 04b, `matmul_kernel` appears
as both K_fwd1 (X @ W1) and K_fwd2 (a1 @ W2), which differ only in the layout strides compiled into
their bodies. Getting that mapping wrong silently computes the wrong thing.

This script derives a SIGNATURE for each sidecar that does not depend on the hash:

    <base name> | buffers=<n> consts=<n> | strides=<sorted integer multipliers in the body>

Row-major strides are exactly what distinguishes the instantiations, so the signature is stable
across rebuilds while the hash is not guaranteed to be.

Two modes:
  survey  — print the signature of every sidecar (use this to author a .map file)
  names   — print `<base name> <path>` per sidecar; for sources whose kernel base names are
            already unique, so no map is needed
  resolve — read a .map file (`<role> <signature>` per line) and print `<role> <path>` pairs,
            failing loudly if any sidecar is unmatched, any role is unused, or a signature is
            ambiguous. A source change that alters the kernels is then a hard error, not a
            silently wrong mapping.

Usage:
  python3 air_roles.py survey  <dir-of-ll-files>
  python3 air_roles.py names   <dir-of-ll-files>
  python3 air_roles.py resolve <dir-of-ll-files> <roles.map>
"""

import os
import re
import sys
from collections import defaultdict

DEFINE = re.compile(r"^define\s+(?:[a-z_]+\s+)*void\s+@(\S+?)\(")
MUL = re.compile(r"=\s*mul(?:\s+nsw|\s+nuw)*\s+i\d+\s+%[\w.]+,\s*(\d+)\b")
SHL = re.compile(r"=\s*shl(?:\s+nsw|\s+nuw)*\s+i\d+\s+%[\w.]+,\s*(\d+)\b")


def base_name(mangled):
    """`_04b_train_mlp_gpu_matmul_kernel6A6A_1c8acd…` -> `matmul_kernel`."""
    name = mangled.lstrip("_")
    name = re.sub(r"6A6A.*$", "", name)           # Mojo parameter-mangling marker
    name = re.sub(r"_[0-9a-f]{16}$", "", name)
    # strip the leading module name (the source stem, which precedes the function name)
    parts = name.split("_")
    for i in range(len(parts)):
        tail = "_".join(parts[i:])
        if tail.endswith(("kernel", "ker")) or "kernel" in tail:
            return tail
    return name


def signature(path):
    text = open(path).read()
    mangled, nbuf, nconst = None, 0, 0
    for line in text.splitlines():
        m = DEFINE.match(line)
        if m:
            mangled = m.group(1)
            head = line[: line.index("{")] if "{" in line else line
            nbuf = len(re.findall(r"ptr addrspace\(1\) noundef", head))
            nconst = len(re.findall(r"ptr addrspace\(2\) noundef", head))
            break
    if mangled is None:
        return None, None
    strides = sorted(int(x) for x in MUL.findall(text) + [str(2 ** int(s)) for s in SHL.findall(text)])
    sig = f"{base_name(mangled)}|buffers={nbuf} consts={nconst}|strides={','.join(map(str, strides))}"
    return sig, mangled


def collect(directory):
    out = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".ll"):
            continue
        path = os.path.join(directory, fn)
        sig, mangled = signature(path)
        if sig is None:
            continue
        out.setdefault(sig, []).append(path)
    return out


def read_map(path):
    roles = []
    for raw in open(path):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        role, _, sig = line.partition(" ")
        roles.append((role.strip(), sig.strip()))
    return roles


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode, directory = sys.argv[1], sys.argv[2]
    found = collect(directory)

    if mode == "survey":
        print(f"# {sum(len(v) for v in found.values())} sidecars, {len(found)} distinct signatures")
        print("# columns: <role>  <signature>   (fill in <role>, then use as a .map file)")
        for sig, paths in sorted(found.items()):
            note = "" if len(paths) == 1 else f"   # {len(paths)} identical instantiations share this"
            print(f"ROLE_ME {sig}{note}")
        return 0

    if mode == "names":
        seen = {}
        for sig, paths in found.items():
            base = sig.split("|")[0]
            seen.setdefault(base, []).extend(paths)
        dupes = {b: v for b, v in seen.items() if len(v) > 1}
        if dupes:
            sys.stderr.write("air_roles: base names are not unique; a role map is required for:\n")
            for b, v in dupes.items():
                sys.stderr.write(f"  {b}  ({len(v)} instantiations)\n")
            return 2
        for base, paths in sorted(seen.items()):
            print(f"{base} {paths[0]}")
        return 0

    if mode != "resolve" or len(sys.argv) < 4:
        sys.exit(__doc__)

    roles = read_map(sys.argv[3])
    by_sig = defaultdict(list)
    for role, sig in roles:
        by_sig[sig].append(role)

    errors = []
    for sig, assigned in by_sig.items():
        if len(assigned) > 1:
            errors.append(f"signature claimed by {len(assigned)} roles ({', '.join(assigned)}): {sig}")
        if sig not in found:
            errors.append(f"role '{assigned[0]}' matches no sidecar; signature: {sig}")
    for sig, paths in found.items():
        if sig not in by_sig:
            errors.append(f"sidecar has no role in the map: {sig}\n    file: {os.path.basename(paths[0])}")

    if errors:
        sys.stderr.write(
            "air_roles: the kernels do not match the role map.\n"
            "The source's kernel set changed, so the mapping must be re-derived\n"
            "(run: python3 air_roles.py survey <dir>) before the build can be trusted.\n\n"
        )
        for e in errors:
            sys.stderr.write("  ERROR: " + e + "\n")
        return 2

    for role, sig in roles:
        print(f"{role} {found[sig][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

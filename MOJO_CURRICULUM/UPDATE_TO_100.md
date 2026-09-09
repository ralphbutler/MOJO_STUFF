# 🔧 Migrating this curriculum from Mojo 1.0.0b2 to Mojo 1.0.0

> **What this file is:** the brief that was written to drive the migration, kept
> verbatim, with the **✅ Outcome** section at the bottom recording what actually
> happened. It is here as a worked example — what a real Mojo version bump costs,
> what the release notes don't tell you, and how to find the rest by asking the
> compiler. The migration is **done**; nothing here is an outstanding task.

## The original brief

Written 2026-09-09 for a fresh session to execute. Read this whole file first.

## The situation

This project is pinned to **Mojo 1.0.0b2**. Mojo 1.0.0 has shipped and changed a lot; **b2 code
does not compile on 1.0.0**. The sibling project `../MOJO_TERNARY/` is already on 1.0.0 and its
`README.md` / `STATUS_CURR.md` record the differences that were found the hard way — all of them
verified by actually compiling, not inferred.

There is also a latent hazard to fix regardless of anything else:

```toml
"max[serve]>=26.4.0"     # floating
"mojo>=1.0.0b2"          # floating — and 1.0.0 > 1.0.0b2
prerelease = "allow"
```

Only `uv.lock` currently holds this at b2. A single `uv lock --upgrade`, or a fresh checkout whose
lock gets regenerated, silently jumps to 1.0.0 and every GPU file stops compiling — looking like
the curriculum is broken rather than the toolchain having moved.

## Ground rules (these matter more than speed)

1. **Never claim a file works until you have compiled AND run it.** A previous session on the
   sibling project repeatedly asserted things about Mojo APIs that turned out to be false. Every
   claim in this file was compile-verified; keep that standard.
2. **`RESULTS01.md` and `CURRICULUM.md` record expected outputs.** They are the regression gate.
   Diff each migrated file's output against them and report differences rather than explaining
   them away.
3. **Ask the compiler when a signature is in doubt.** A deliberate arity error makes it print
   every overload, e.g. `alloc[Scalar[DType.int8]](64, 1, 2, 3)`. The Mojo stdlib ships without
   source, so this is the only reliable way to check.
4. **The bundled `mojo-gpu-fundamentals` skill is wrong in places** — it documents `max.gpu` for
   device-side symbols (should be `std.gpu`) and its worked example passes `size: Int`, which does
   not compile. Invoke `mojo-syntax` as usual, but trust the table below over the skills.
5. Ralph runs commands himself where practical; give exact commands with absolute paths.
6. Do not write a `CLAUDE.md`, and do not create memory files. Put findings in this file or the
   project's own docs.

## Step 0 — pin the toolchain deliberately

In `pyproject.toml`:

```toml
dependencies = [
    "max[serve]==26.5.0",   # Mojo 1.0.0 ships as part of Modular 26.5
    "mojo==1.0.0",          # PINNED, never >= — a floating pin is what created this mess
]
```

Then `uv sync` and confirm:

```bash
cd MOJO_CURRICULUM
uv sync
.venv/bin/mojo --version        # expect: Mojo 1.0.0 (ed45d567)
```

Note `max` is now required for more than serving — `parallelize` lives in it (see below).

## The known differences, b2 → 1.0.0

All verified by compiling on 2026-09-08/09.

| 1.0.0b2 (what this project uses) | 1.0.0 |
|---|---|
| `from std.algorithm import parallelize` | `from max.algorithm import parallelize` — left the stdlib; `vectorize` and `map` stayed |
| `from std.gpu.host import DeviceContext` | `from max.gpu.host import DeviceContext` — host side moved to `max` |
| `from std.gpu.sync import barrier` | `from max.gpu import barrier` |
| `from std.gpu.memory import AddressSpace` | `AddressSpace` is in the **prelude** — drop the import |
| `from std.gpu import global_idx, thread_idx, block_idx` | **unchanged** — device-side symbols stayed in `std.gpu` |
| `from std.memory import alloc` | still there, but `alloc[T](n)` warns; see below |
| `UnsafePointer[T, origin]` | `Pointer[T, origin]` |
| `ptr[i]` | `ptr[unsafe_offset=i]` |
| `ptr.load[width=W](i)` / `ptr.store(i, v)` | `ptr.unsafe_load[width=W](i)` / `ptr.unsafe_store(i, v)` |
| `ptr + i` | `ptr.unsafe_offset(i)` |
| `def __del__(deinit self)` | `def __deinit__(deinit self)` |
| `ptr.free()` | `dealloc(allocation^)` — hold an `Allocation`, not a leaked pointer |
| `alloc[T](count)` | `alloc(Layout[T](count=n))` → `Allocation[T]`; `Layout`/`Allocation`/`dealloc` come from `std.memory` |
| `read` (argument convention) | `imm` |
| `InlineArray[T, N]` | `Array[T, N]` |
| `alias X = ...` | `comptime X = ...` (may already be done here) |

Two 1.0.0 traps that cost real time on the sibling project:

- **Scalar GPU kernel arguments must be fixed-width.** `size: Int` fails with *"Int and UInt do
  not conform to DevicePassable; use a fixed-width type such as Int32 or Int64"*. Use `Int32` and
  wrap with `Int(...)` inside the kernel for comparisons. **`04b`, `02`, and the `03*` files all
  pass scalar sizes to kernels — expect to hit this.**
- **A `Pointer[T, MutUntrackedOrigin]` struct field is a silent use-after-free.** An untracked
  origin does not extend the owner's lifetime, so ASAP destruction can free it while the pointer
  is still live; reads then return zeros with no warning. Hold an `Allocation` instead. (Reported
  upstream as `modular/skills#11`.)

Also: bracket literals infer `Array[T, N]`, not `List[T]`. `var ids: List[Int] = [...]` needs the
annotation if a `List` is required.

## Files, and what each is expected to need

Sixteen `.mojo` files. Twelve contain at least one affected construct.

| file | expected work |
|---|---|
| `hello.mojo` | probably none — verify first, it is the smoke test |
| `00_simd_type.mojo` | likely none; SIMD API was stable |
| `01_vecadd_safe.mojo` | pointer API (`load`/`store`/indexing) |
| `01_vecadd_unsafe.mojo` | pointer API, `alloc`/`free` → `Allocation`/`dealloc` |
| `02_vecadd_gpu.mojo` | `std.gpu.host` → `max.gpu.host`; scalar kernel arg → `Int32` |
| `03a_matmul_naive.mojo` | as `02` |
| `03b_matmul_tiled.mojo` | as `02`, plus `barrier` → `max.gpu`, `AddressSpace` import dropped |
| `03c_matmul_coarse.mojo` | as `03b` |
| `03d_matmul_check.mojo` | as `03b` |
| `03e_matmul_cpu.mojo` | `parallelize` → `max.algorithm`; pointer API |
| `04_train_mlp.mojo` | `parallelize`, `vectorize`, pointer API |
| `04b_train_mlp_gpu.mojo` | the big one — GPU host+device imports, shared memory, scalar args |
| `custom_op_kernels/relu.mojo` | GPU imports; also check `@compiler.register` still applies |
| `custom_op_kernels/__init__.mojo` | probably none |
| `scalar_sum.mojo` | `parallelize`, pointer API |
| `spinlock_atomic.mojo` | atomics + pointer API; check `Atomic` API did not move |

## Suggested order

Bottom-up, so a failure is always in the file you just touched:

1. `hello.mojo` — proves the toolchain
2. `00`, `01_safe`, `01_unsafe` — CPU, pointer API only
3. `02_vecadd_gpu.mojo` — **the first GPU file; get the import set right here and the rest follow**
4. `03a` → `03b` → `03c` → `03d` — matmul ladder, increasing GPU feature use
5. `03e_matmul_cpu.mojo`, `scalar_sum.mojo`, `spinlock_atomic.mojo` — CPU parallelism
6. `04_train_mlp.mojo`, then `04b_train_mlp_gpu.mojo`
7. `custom_op_kernels/relu.mojo` + `05_custom_max_op.py` — the MAX capstone

Run each as you go:

```bash
cd MOJO_CURRICULUM
.venv/bin/mojo run 02_vecadd_gpu.mojo
```

(`uv run mojo ...` also works; `.venv/bin/mojo` is explicit about which toolchain.)

## Expect Nabla to be a casualty

`train_nabla_mlp.py` + `setup_nabla_venv.sh` use **Nabla**, which is alpha and pins Modular
**nightly** in its own `.venv-nabla`. It may simply not work against 1.0.0, and that is not
something to fix here. Acceptable outcomes: leave `.venv-nabla` on whatever nightly it wants and
note the version skew in the README, or mark the Nabla path as unsupported with a dated note.
Do not spend long on it, and do not let it block the rest.

`train_torch_mlp.py` and `bench_torch.py` are PyTorch and unaffected.

## What to update when the code is done

- **`README.md`** — the `📦 Setup` section, and the toolchain version wherever it appears.
- **`CURRICULUM.md`** — any inline import snippets in the per-file walkthrough.
- **`RESULTS01.md`** — append the 1.0.0 numbers next to the b2 ones rather than overwriting; if
  performance changed, that is itself a result worth recording.
- **`MAX_VS_MOJO_GETTING_STARTED.md`** — check for version-specific instructions.
- **`../README.md`** (the working-tree parent) — its version table lists this
  project as b2; move it to the 1.0.0 row.
- Then Ralph will want help updating `~/GitHub/MOJO_STUFF/` — its top-level `README.md` carries
  the same version table and the same b2/1.0.0 warning section, both of which shrink once this
  project migrates.

## Report honestly at the end

State plainly: which files compile and run, which produce output matching `RESULTS01.md`, which
differ and by how much, and anything left broken. If Nabla or the MAX capstone does not survive,
say so rather than glossing it. A partially-migrated project that is honestly labelled is more
useful than one claimed complete.

Git has the b2 version committed in `~/GitHub/MOJO_STUFF/`, so rollback is free.

---

# ✅ Outcome — executed 2026-09-09

**Status: complete.** All 16 `.mojo` files compile, run, and produce output matching
`RESULTS01.md` / `CURRICULUM.md`. Zero compiler warnings across the whole set. The MAX
tier, the PyTorch tier, and Nabla all still work. Nothing was left broken.

## Toolchain

`pyproject.toml` now pins `max[serve]==26.5.0` and `mojo==1.0.0` (exact, with a comment
saying why). `uv sync` → `Mojo 1.0.0 (ed45d567)`.

Note the venv also had a stale shebang from the `~/Desktop/DEMO/` → `~/Desktop/LLMs/`
move; `uv sync` repaired it.

## What each file actually needed

| file | changed | what |
|---|---|---|
| `hello.mojo` | no | ran unmodified |
| `00_simd_type.mojo` | no | ran unmodified |
| `01_vecadd_safe.mojo` | yes | `load`/`store` → `unsafe_load`/`unsafe_store` |
| `01_vecadd_unsafe.mojo` | yes | `alloc`/`free` → `Allocation`/`dealloc`; `unsafe_offset=`; `unsafe_load`/`unsafe_store` |
| `02_vecadd_gpu.mojo` | yes | `max.gpu.host`; `size: Int` → `Int32` |
| `03a_matmul_naive.mojo` | yes | `max.gpu.host` only (no scalar kernel args — it uses comptime `N`) |
| `03b_matmul_tiled.mojo` | yes | `max.gpu.host`, `max.gpu` for `barrier`, dropped the `AddressSpace` import |
| `03c_matmul_coarse.mojo` | yes | as `03b`, `Array`, **and the register-spill fix below** |
| `03d_matmul_check.mojo` | yes | as `03c` |
| `03e_matmul_cpu.mojo` | yes | `max.algorithm`; `Allocation`/`dealloc`; `unsafe_origin_cast` |
| `04_train_mlp.mojo` | no | ran unmodified — it uses no pointers and no `parallelize` |
| `04b_train_mlp_gpu.mojo` | yes | `max.gpu.host`; all 9 kernels' scalar args → `Int32` + `Int(...)` at the compare sites |
| `custom_op_kernels/relu.mojo` | yes | `@compiler.register` → `@register` from `extensibility` |
| `custom_op_kernels/__init__.mojo` | no | — |
| `scalar_sum.mojo` | yes | `max.algorithm`; pointer API; `read` → `imm`; `kahan_add[1](...)` |
| `spinlock_atomic.mojo` | yes | `Pointer(to=…)`; `unsafe_offset=`. `Atomic` did **not** move. |

## Corrections and additions to the table above

The table in this file was right about everything it listed. Four things it did not cover,
each found by compiling:

1. **`import compiler` is gone.** The custom-op decorator moved:
   `from extensibility import register`, then `@register("relu")`. There is no top-level
   `compiler` module and no `max.compiler`. This was the only thing that broke the capstone.
2. **`Allocation` is an owner, not a pointer.** It has no `__getitem__`, no `unsafe_load`.
   The shape is `alloc(Layout[T](count=n))` → `Allocation[T]`, then `.unsafe_ptr()` →
   `Pointer[T, origin_of(...)]`, and you index/load/store through the pointer.
3. **That pointer's origin is *tracked*, and will not convert to `MutAnyOrigin`.** Passing
   it to a helper typed `Pointer[T, MutAnyOrigin]` is a compile error. Two ways out: make
   helpers generic over `[o: MutOrigin]` (the trait is `MutOrigin`, **not**
   `MutableOrigin` — that name does not exist), or erase it once with
   `p.unsafe_origin_cast[MutAnyOrigin]()`. `03e` uses the second, and keeps the five
   `Allocation`s alive in `main()` until the explicit `dealloc`.
4. **Backward SIMD-width inference through a `mut` argument is gone.** `scalar_sum.mojo`'s
   `kahan_add[w: Int](mut s: SIMD[dtype, w], …)` used to infer `w = 1` when called with a
   `Float32`. On 1.0.0 that fails with *"l-value of type 'Float32' cannot be converted to
   reference of type 'SIMD[DType.float32, w]', it depends on an unresolved parameter 'w'"*.
   The scalar call sites now write `kahan_add[1](s, c, x)`; the vector ones still infer.

Also: `InlineArray` still exists and still compiles without a deprecation warning, so the
`Array` rename is optional for *compiling* — but see the next section for why it is not
optional for *performance*. `stack_allocation` for shared memory comes from `layout` in
these files (taking a layout), not from `std.memory`.

## ⚠️ The one real regression, and the fix

`03c_matmul_coarse.mojo` first ran at **6.9 ms / 2,490 GFLOP/s** against b2's
**3.968 ms / 4,330 GFLOP/s** — a 1.7× loss that wiped out the register-blocking win the
curriculum is built around. PyTorch on the same machine was unchanged (MPS 1.322 ms vs
b2's 1.284 ms), so it was the toolchain, not the machine.

Cause: the per-thread accumulators, held in `InlineArray[Scalar[dtype], TM*TN]`. Under
1.0.0 the Metal backend **spills that array to memory**. Renaming to `Array` changed
nothing — verified by timing both spellings, 6.9 ms either way.

Fix: hold them in `SIMD[dtype, TM*TN]` — one register-resident value of the same length,
indexed identically by the `comptime` loops. That gives **3.99 ms / 4,303 GFLOP/s**,
matching b2. Applied to `03c` and `03d`; `03d` still reports 0/65536 mismatches, max
error 0.0.

This is now written up in `RESULTS01.md` as a teaching point, since "in registers" turning
out to be a property of the generated code rather than of your intent is exactly the kind
of thing the curriculum exists to show.

## Regression check against RESULTS01.md

N = 2048, 50 iters, same machine. Every row within run-to-run noise; leaderboard order
unchanged.

| | b2 | 1.0.0 |
|---|---|---|
| PyTorch MPS | 1.284 ms / 13,381 | 1.322 ms / 12,995 |
| Mojo coarse | 3.968 ms / 4,330 | 3.99 ms / 4,303 |
| PyTorch CPU | 5.412 ms / 3,174 | 5.115 ms / 3,359 |
| Mojo naive | 6.379 ms / 2,693 | 6.43 ms / 2,670 |
| Mojo tiled | 8.010 ms / 2,145 | 7.85 ms / 2,188 |

`04`/`04b` print the identical loss curve (`2.1783555 → 0.00056147523`, bit-for-bit).
Timings: the small-size GPU penalty *improved*, ~30× slower on b2 → ~10× on 1.0.0
(0.27 vs 0.027 ms/epoch); the large-size win is unchanged at ~15× (7.5 vs 117 ms/epoch at
`N=8192, H=1024`). `CURRICULUM.md` updated accordingly.

`03e` (CPU ladder, N=1024): naive 2.5 → SIMD 28.3 → parallel 199 GFLOP/s, PASS.
`scalar_sum`, `spinlock_atomic`, `03d`, `01*`, `02` all PASS.

## Nabla was not a casualty

Contrary to the expectation above, `train_nabla_mlp.py` still runs and converges
(`6.0340 → 0.0003` over 400 epochs). It lives in `.venv-nabla`, which pins
`modular==26.2.0.dev2026021705` — entirely independent of `./.venv`, so upgrading the main
venv could not touch it. No action needed; the version skew was already deliberate and
already documented in `setup_nabla_venv.sh`.

`train_torch_mlp.py` (100% train accuracy) and `bench_torch.py` unaffected, as expected.

## MAX tier re-verified

- `./max_generate.sh` — Qwen2.5-0.5B on `gpu[0]` (Metal), 22.9 tok/s.
- `./max_serve_litellm.sh` — server up in 42 s, litellm round-trip returned a completion.
- `05_custom_max_op.py` — `device: GPU (Metal)`, `RESULT: PASS`.

## Docs updated

`README.md` (new `📦 Setup` section with the pin and the reason), `CURRICULUM.md` (the
`alloc`/`Allocation` explanation in `01`, the `Int32` kernel-argument rule in `02`, the
register-spill note in `03c`, `parallelize`'s move in `03e`, `@register` in `05`, and the
re-measured `04`/`04b` crossover), `RESULTS01.md` (1.0.0 numbers appended beside the b2
ones, plus the regression write-up), `MAX_VS_MOJO_GETTING_STARTED.md` (26.4.0 → 26.5.0
re-verification), and `../README.md` (version table moved to the 1.0.0 row, four new
gotchas added).

Still to do: `~/GitHub/MOJO_STUFF/README.md` carries the same version table and b2/1.0.0
warning section — both shrink now that only the b2 projects remain on b2.

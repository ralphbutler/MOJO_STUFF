# 🔬 Mojo on Aurora's Intel GPUs — the tuning follow-up, and an assessment

**Follow-up to the note of 16 September**, which promised the numbers, the method, what it cost, and
how useful this is likely to be.

R. Butler and Opus 5 · 2026-09-18

---

## 📌 The answer to the question you are holding

That note said our matrix multiply reached about 7% of what Intel's own library gets on the same GPU,
that the example kernel was tuned for Apple and NVIDIA, and that tuning for this hardware was work we
had not done.

**We have now done it. 7% → 36.8%: a 5.3× speedup, in one day.** It took changing five constants in
the kernel and asking Intel's compiler for the GPU's larger register file. Nothing about the compiler
backend, the runtime or the hardware changed.

For the situations we actually care about — code you write yourself, because no library does it:

| | |
|---|---|
| The same kernel on a Polaris A100 | 8,966 GFLOP/s · **we now get 7,782 on one Aurora tile — 87%** |
| Our hand-written, Intel-specific C++ for the training loop | 0.366–0.370 ms/epoch · **Mojo does 0.344 — faster** |
| Aurora's own CPUs, 102 cores, heavily parallelised | 419 GFLOP/s · **the *unoptimised* GPU kernel does 2,449 — 5.8×** |
| A stencil we wrote ourselves — no vendor library exists for it | **631 GB/s on Aurora, ~666 on a MacBook** |

Results are exact, not approximate, in every case — all 4,194,304 cells of the matrix multiply, all
8,388,608 of the stencil. Aurora figures are means of three runs (spread under 0.4%; we quote ±2.5%
for variation between compute nodes). Laptop figures are medians of nine and are much noisier — a
MacBook is not a dedicated compute node.

**What produced them.** Mojo's compiler was open-sourced on 13 September. It shipped GPU support for
NVIDIA, AMD and Apple and none for Intel — which is why, in July, our GPU examples ran on Polaris and
only on Aurora's CPUs. We added an Intel GPU target to the compiler, plus a runtime beneath Mojo's own
device API. Three unmodified programs from our curriculum — vector add, matrix multiply, and a small
neural-network training loop — now run on an Aurora GPU tile, host code and all.

**The question this really answers** is not "is Mojo fast on Aurora." It is: **can a small team make
Mojo follow them onto hardware the vendor does not support, and get usable performance?**

---

## 💰 What it cost

| | |
|---|---|
| New compiler backend (C++) | **593 lines**, 6 files |
| New GPU runtime (C++, Level Zero under Mojo's device API) | **931 lines**, 1 file |
| Edits to existing compiler and standard-library files | **185 lines added**, 27 removed, across 13 files |
| **Total** | **~1,700 lines, 23 files** |
| Elapsed | 13 Sept (open-sourced) → 16 Sept (running) → 17 Sept (tuned) |
| People | Two — one domain scientist, one AI assistant |

**Our own estimate beforehand was 4–6 months, or 9–15 if Mojo's GPU lowering turned out to be
closed-source.** It took days. Why we were wrong is the most useful thing in this report: the GPU
lowering is in the open tree and target-parameterised, so it needed a new target rather than a
reimplementation; **LLVM's SPIR-V backend did the code generation**, so we never wrote one; and Mojo's
existing Apple output showed us exactly what correct output looks like, so we reproduced a known-good
mapping instead of inventing one.

**The transferable lesson: the cost of adding a target is dominated by whether your hardware already
has an LLVM backend.** If it does, this is days-to-weeks by people who are not compiler specialists.
If it does not, the estimate goes back to months. That is the question to ask about any new chip you
care about.

---

## ⚡ Does it run well?

Running is not enough — a port 10× off the hardware is close to not having a port.

Both compilation paths are proven: `mojo run` (just-in-time) and `mojo build` (standalone
executables), with the same results and the same speed to within measurement noise.

**Against hardware you trust:** 7,782 GFLOP/s on one Aurora tile, against 8,966 for the *identical*
kernel on an A100 — **87%**, with a backend days old.

**Against hand-written, machine-specific code:** the training loop runs at 0.344 ms/epoch through
Mojo, *faster* than the Intel-specific C++ we wrote for it earlier in the summer. Writing it in Mojo
cost nothing against writing it by hand for this machine.

**Against Intel's own library:** oneMKL's SGEMM reaches 21,148 GFLOP/s, so our kernel sits at 36.8%.
If you write numerical libraries for a living, that is disappointing, and we will not dress it up.

**But that is not the work this supports.** We are not competing with oneMKL or cuBLAS — when your
code needs a dense matrix multiply, call the vendor's library, from Mojo exactly as from C++. What
this supports is **writing good user code, which sometimes requires a kernel no library provides**: a
stencil, a particle push, a domain-specific update rule, the one loop in your application that is
yours alone. The question is not "can it beat oneMKL" but "when I must write a kernel myself, do I get
good use of the hardware?" For the record, the remaining gap to oneMKL is named, not mysterious: it
uses Intel's systolic matrix instructions, which our backend does not emit yet.

**Tuning was ordinary work, and that matters more than the number.** The first working version ran at
1,480 GFLOP/s. Intel's compiler was reporting that our kernel spilled registers; two independent fixes
— five geometry constants, and asking for the larger register file — took it to 7,782. No vendor
assistance, no exotic tools, one day.

⚠️ **One caveat for kernel authors.** The larger-register-file setting is **not** free and should never
be on by default. It won ×2.98 on the matrix multiply, but cost a memory-bound kernel **a third of its
bandwidth** (our streaming copy fell 708 → 459 GB/s) and the training loop 3.2%: it halves the threads
resident per execution unit, and memory-bound code needs those in flight. **Turn it on when Intel's
compiler reports a register spill, otherwise leave it alone.**

---

## 🖥️ One source, two machines

All of this was developed on a MacBook and ported to Aurora. The same unmodified files run on both,
same Mojo version — stock on the Mac, our fork on Aurora:

| program | Mac (M4 Max) | Aurora (one PVC tile) |
|---|---|---|
| Vector add, 1M elements | exact · 0.102 ms/pass | exact · **0.039 ms/pass** |
| Matrix multiply, N=2048, *as tuned for the Mac* | exact · **4,356 GFLOP/s** | exact · 1,480 GFLOP/s |
| Matrix multiply, *retuned for Aurora* | — | **7,782 GFLOP/s** |
| Training loop, 300 epochs | 7 s.f. agreement · **0.273 ms/epoch** | · 0.344 ms/epoch |
| 5-point Jacobi stencil, 2048² | exact · **~666 GB/s** | exact · **631 GB/s** |

**Read the matrix-multiply rows together — they are the whole tuning story in miniature.** Those
constants were chosen on the Mac: 4,356 GFLOP/s there, 1,480 on Aurora. Retuned for Aurora, the same
algorithm reaches 7,782 — **1.8× the Mac**. Neither machine is "the fast one." The constants belong to
the hardware, and moving them is ordinary work, in both directions.

The training loop reproduces a NumPy float32 reference curve, 2.1783555 → 0.0005614754, at every
printed epoch. Across different machines that curve agrees to about seven significant figures
rather than exactly (Aurora 0.0005614754, Mac 0.00056147523). Float32 addition is not associative, so
a different machine accumulates in a different order. That is expected, and true of any pair of GPUs.

(We reported on Polaris separately, earlier in the summer. Its Mojo installation predates a
module-path change in the language and would need updating before these exact files run there.)

---

## 🔍 What we learned about Mojo itself

We set out to assess the ecosystem, not advocate for it. Two findings cut the other way.

**Mojo has no way to say "these GPU operations are independent."** Its device API presents an in-order
stream: each launch completes before the next begins. We measured the cost — for a kernel large enough
to matter, about **38 microseconds per launch** — and confirmed it is inherent to the ordering, not to
our implementation: an alternative mechanism gave the same number, and removing ordering entirely
collapsed it to 2.8 µs. CUDA streams and Level Zero queues both let a program express independence and
overlap the work. **Mojo's API currently does not, so a program with several large independent GPU
operations cannot recover that time on any backend — NVIDIA included.** We found it by building
against the API; it does not affect programs whose kernels are small, which is most of our own code.

**The ordering guarantee is load-bearing, and its absence is not a soft failure.** With ordering
removed, the training loop did not merely produce wrong numbers — it crashed the GPU with a page
fault. Worth knowing for anyone tempted to relax synchronisation for speed.

---

## 🚧 What this does not cover

- **This is our fork, not official Mojo.** Modular does not support Intel GPUs and we expect that to
  remain true. An official release still reports no accelerator on Aurora.
- **One GPU tile.** Aurora nodes have 12. Multi-tile and multi-GPU are untouched.
- **32-bit floats only.** No fp64, which some of you need.
- **A slice of the runtime** — 18 functions, covering what these programs use. No streams, events or
  graphs.
- **Compiler flags are still needed** to point at MAX's packages, because MAX is not yet built with
  our fork by default. Cosmetic, but it means the command line is not yet as clean as the source.
- **The newer system image only** — login nodes `uan-0007`/`0008` and the `next-eval` queue, until the
  October update.
- **MAX itself — Modular's inference layer — is out of reach.** It is closed, so it will not run on
  these GPUs whatever we do to the compiler.
- **We have no trustworthy efficiency denominator.** We can say what our kernels achieve in absolute
  terms; we cannot yet say what fraction of the hardware's real capability that is. Our attempt to
  measure a bandwidth ceiling with a simple copy kernel failed its own sanity check — on the MacBook
  the "ceiling" came out below the stencil, which is impossible. A vectorised streaming benchmark is
  the missing piece.
- **Narrow evidence base.** Four programs. Nothing here proves anything about *your* code: sparse
  structures, irregular access and deep multi-kernel pipelines are all unmeasured.

---

## 🔮 What would make it production-worthy

**Multi-tile and multi-GPU** — 12 tiles per node is the reason to use Aurora at all. **fp64**, for the
codes that need it. **Systolic instructions**, which is the remaining gap to vendor-library matmul
speed. **More of the runtime** — streams and events, which is also where any fix to the independence
gap above would live. **Upstreaming is not possible yet:** Modular does not accept compiler
contributions until roughly the end of 2026, so this is a fork we carry, not merge.

---

## ⚠️ Risks and unknowns

**Fork maintenance.** Mojo moves fast; every release we follow is a rebase, and nobody is obliged to
keep our target working. Our footprint is small (~1,700 lines), which helps, but this is real ongoing
cost. **Modular's direction.** If Intel GPU support never becomes official, this stays a community
path — we think that is likely and you should plan on it. **Node-to-node variation.** Within a job our
numbers repeat to better than 0.4%; across three compute nodes the same measurement spread by about
2.5%, and we quote the wider figure. **We are not a support team**, and this is not a product. It is
an existence proof with numbers attached.

Code and reproduction instructions will be published alongside our Mojo curriculum.

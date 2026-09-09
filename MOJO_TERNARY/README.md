# 🌱 MOJO_TERNARY — a ternary inference engine in Mojo

An inference engine for **ternary (1.58-bit) LLMs**, written from scratch in **Mojo**, CPU-first
on an Apple M4 Max. Two bets in one project: weights that are only −1/0/+1, and a language built
for accelerators. Runs PrismML's **Ternary-Bonsai** models (1.7B and 8B) straight from GGUF.

## 📊 Status — measured, not projected

**Works:** loads real Bonsai 1.7B and 8B from GGUF and generates correct text. Output matches
HuggingFace transformers exactly.

**Speed:** 34.7 tok/s (1.7B), 12.9 tok/s (8B).

**Same machine, llama.cpp:** 58.9 / 25.7 on CPU, 327 / 129 on GPU.

**So:** a correct engine, roughly 2× slower than llama.cpp's CPU path and 10× slower than its GPU
path. **The performance goal is not met.** The best kernel here is CPU packed weights at 1.06×
over the original; three GPU kernel shapes were tried and none beat the CPU. What remains is a
kernel-optimisation problem, described in `STATUS_CURR.md` under *Kernel Work*.

## 📦 What this repo omits, and how to rebuild it

Four things are deliberately not committed. All are reproducible; none takes long.

**1. The Mojo toolchain (`.venv/`, ~930 MB)**
```bash
uv sync                      # versions are pinned in pyproject.toml / uv.lock
```
Mojo is **not** on `PATH` — it installs as a Python dependency. Always invoke `.venv/bin/mojo`.
`max` is required because `parallelize` left the stdlib in Mojo 1.0.

**2. The models** — they live in the HuggingFace cache, not here.
```bash
huggingface-cli download prism-ml/Ternary-Bonsai-1.7B-gguf Ternary-Bonsai-1.7B-PQ2_0.gguf
huggingface-cli download prism-ml/Ternary-Bonsai-1.7B-gguf Ternary-Bonsai-1.7B-F16.gguf
huggingface-cli download prism-ml/Ternary-Bonsai-8B-gguf   Ternary-Bonsai-8B-PQ2_0.gguf
```
Use the **`PQ2_0`** files. The `*-Q2_0.gguf` files declare ggml type 42 while storing the
group-128 layout, which current llama.cpp reads as group-64 and refuses — see
[PrismML-Eng/llama.cpp#167](https://github.com/PrismML-Eng/llama.cpp/issues/167). This engine's
reader accepts both, because it derives the block layout from tensor offsets rather than trusting
the type id.

**3. Reference activations (`oracle/ref/`, ~3 MB)** — the correctness oracle.
```bash
uv run --with torch --with transformers --with safetensors oracle/dump_reference.py
```
Deliberately external: it runs the model through HuggingFace transformers, which is authored by
someone else and is what the model was released against, so it cannot share a misunderstanding
with our Mojo.

**4. The llama.cpp reference build (`vendor/`, ~1.0 GB)** — the speed bar.
```bash
mkdir -p vendor && cd vendor
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 16 --target llama-bench llama-cli
# CPU baseline — the fair comparison, since this engine is CPU-only:
./build/bin/llama-bench -m <path>/Ternary-Bonsai-8B-PQ2_0.gguf -p 0 -n 128 -ngl 0 -t 16 -r 3
```

## 🏃 Running it

Paths default to the 1.7B in the HuggingFace cache; every program takes overrides.

```bash
# Read the GGUF, verify the block layout, verify dequantization (ends "M0 DONE.")
.venv/bin/mojo run gguf_meta.mojo

# The kernel: scalar vs portable SIMD vs NEON sdot vs sdot+threads
.venv/bin/mojo run q2_gemv.mojo
.venv/bin/mojo run q2_gemv.mojo token_embd.weight        # 310 MB, DRAM-bound

# Full forward pass + generation, diffed against the transformers oracle
.venv/bin/mojo run m2.mojo                               # ~22 s
.venv/bin/mojo run m2.mojo <path-to-8B-PQ2_0.gguf>       # the 8B
.venv/bin/mojo run m2.mojo <gguf> fp32                   # slow fp32 reference path

# Packed weights: CPU int8 vs CPU packed vs GPU packed, all checked against each other
.venv/bin/mojo run packed_gemv.mojo token_embd.weight

# The GPU kernel on its own (it loses to the CPU — that is the finding)
.venv/bin/mojo run gpu_gemv.mojo token_embd.weight

# Synthetic bandwidth probe — needs no model file
.venv/bin/mojo run q2_dot.mojo
```

`gguf_meta.mojo`, `m2.mojo` and `packed_gemv.mojo` **raise on failure** rather than warn — they
are regression gates, so a silent pass is a real pass.

## 🗂️ Files

| file | |
|---|---|
| `gguf.mojo` | Shared GGUF reader: header parsing, metadata, tensor lookup, Q2_0 dequantization |
| `gguf_meta.mojo` | Verifies the container and the dequantizer against the F16 build |
| `model.mojo` | Model loader (architecture read from metadata, not hardcoded) + CPU kernels |
| `m2.mojo` | Full 28/36-block forward pass, generation, oracle diff, built-in profile |
| `tokenizer.mojo` | GPT-2 byte-level **decoder**. Encoding is not implemented — see below |
| `packed_gemv.mojo` | Packed weights, CPU and GPU. The best kernel, and the open problem |
| `gpu_gemv.mojo` | The int8 GEMV as an Apple GPU kernel |
| `q2_gemv.mojo` | Kernel ladder on real weights |
| `q2_dot.mojo` | Synthetic 16384² benchmark with a read-only bandwidth roofline |
| `oracle/dump_reference.py` | The external correctness oracle |
| `STATUS_CURR.md` | The detailed record: what was built, measured, and got wrong |
| `IDEA_ORIG.md` | Superseded first draft, kept deliberately. Four of its technical claims are wrong |

## ⚠️ Known gaps

- **No prompt encoder.** Decoding is pure Mojo; encoding needs qwen2's pre-tokenizer regex and
  Mojo's stdlib has none. Prompt token ids currently come from the oracle. So this **cannot chat
  or serve** — it generates. Three pieces of plumbing away, none of it research.
- **Performance goal unmet**, as above.
- **27B not attempted.** It is a different architecture (48 of 64 layers are Mamba-style
  state-space, plus a different tokenizer and a vision tower) — a second engine, not a bigger one.

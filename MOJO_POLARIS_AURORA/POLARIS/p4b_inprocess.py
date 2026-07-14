# p4b_inprocess.py — load a model ONCE and use it in-process, no server.
# This is MAX's "option 2": the LLM lives inside your program, you call .generate()
# directly. It's the representative HPC pattern (embed inference in a job/pipeline on a
# compute node) — contrast with `max serve`, which wraps this in an HTTP endpoint.
#
# IMPORTANT: MAX's LLM launches a background telemetry worker via multiprocessing (spawn),
# which re-imports this file. All work MUST live under `if __name__ == "__main__":` or the
# child re-runs it and Python raises "start a new process before bootstrapping finished".
#
# Requires the same GPU env as everything else on Polaris:
#   MODULAR_NVPTX_COMPILER_PATH=/soft/compilers/cudatoolkit/cuda-12.8.1/bin/ptxas
#   HF_HUB_OFFLINE=1   (weights pre-cached; compute node is offline)
# Run with:  .venv-max/bin/python p4b_inprocess.py

import time

from max.entrypoints.llm import LLM
try:
    from max.pipelines import PipelineConfig
except ImportError:  # fall back if the public path differs in this MAX build
    from max.entrypoints.pipelines import PipelineConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Who was the first US president?",
    "In one sentence, what is a GPU?",
    "Name three primary colors.",
]


def main():
    # load once (the expensive step: AOT model-graph compile, ~1-2 min the first time)
    t0 = time.time()
    llm = LLM(PipelineConfig(model_path=MODEL))
    print(f"[load] model resident after {time.time() - t0:.1f}s")

    # reuse the resident model for many prompts (the whole point of in-process)
    t1 = time.time()
    outputs = llm.generate(PROMPTS, max_new_tokens=100, use_tqdm=False)
    dt = time.time() - t1

    for prompt, out in zip(PROMPTS, outputs):
        print("\n=== PROMPT:", prompt)
        print(out)

    print(f"\n[generate] {len(PROMPTS)} prompts in {dt:.2f}s "
          f"({dt / len(PROMPTS):.2f}s/prompt, load amortized across all of them)")


if __name__ == "__main__":
    main()

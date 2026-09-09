"""Dump reference activations for Ternary-Bonsai-1.7B using HuggingFace transformers.

This is M2's oracle. It must be INDEPENDENTLY AUTHORED from our Mojo forward pass --
transformers' Qwen3 implementation is the reference the model was released against, so
a shared misunderstanding cannot hide a bug the way a self-written numpy copy would.

Outputs raw float32 binaries + a JSON manifest into oracle/ref/, for the Mojo side to read.

Run:  uv run --with torch --with transformers --with safetensors oracle/dump_reference.py
"""
import json, os, pathlib, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "prism-ml/Ternary-Bonsai-1.7B-unpacked"
OUT = pathlib.Path(__file__).parent / "ref"
OUT.mkdir(exist_ok=True)
PROMPT = "The capital of France is"

# float32 throughout: bf16 rounding would blur the comparison we are trying to make.
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
model.eval()

ids = tok(PROMPT, return_tensors="pt").input_ids
print("prompt :", repr(PROMPT))
print("tokens :", ids[0].tolist())
print("pieces :", [tok.decode([t]) for t in ids[0].tolist()])

with torch.no_grad():
    out = model(ids, output_hidden_states=True)

hs = out.hidden_states          # tuple: embeddings, then one per layer
logits = out.logits[0]          # [T, vocab]

def save(name, arr):
    a = np.ascontiguousarray(arr.detach().numpy().astype(np.float32))
    a.tofile(OUT / f"{name}.bin")
    print(f"  {name:<22} shape={list(a.shape)}")
    return list(a.shape)

print("saving:")
shapes = {
    "embeddings":   save("embeddings", hs[0][0]),      # [T, 2048] after embed, before layer 0
    "after_layer0": save("after_layer0", hs[1][0]),    # [T, 2048]
    "after_layer1": save("after_layer1", hs[2][0]),
    "final_hidden": save("final_hidden", hs[-1][0]),   # [T, 2048] after the last block + final norm
    "logits":       save("logits", logits),            # [T, 151669]
}

# The RoPE inverse frequencies actually used, so the Mojo side can match YaRN exactly
# rather than guessing at the scaling.
rot = model.model.rotary_emb
inv_freq = rot.inv_freq.detach().numpy().astype(np.float32)
inv_freq.tofile(OUT / "inv_freq.bin")
att_scaling = float(getattr(rot, "attention_scaling", 1.0))
print(f"  inv_freq               shape={list(inv_freq.shape)}  attention_scaling={att_scaling}")

top = torch.topk(logits[-1], 10)
manifest = {
    "prompt": PROMPT,
    "tokens": ids[0].tolist(),
    "n_tokens": int(ids.shape[1]),
    "shapes": shapes,
    "inv_freq_len": int(inv_freq.shape[0]),
    "attention_scaling": att_scaling,
    "rope_type": str(getattr(rot, "rope_type", "?")),
    "top10_last_pos": [
        {"id": int(i), "logit": float(v), "piece": tok.decode([int(i)])}
        for v, i in zip(top.values.tolist(), top.indices.tolist())
    ],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("\ntop-10 next-token predictions:")
for e in manifest["top10_last_pos"]:
    print(f"   {e['logit']:8.3f}  {e['id']:>7}  {e['piece']!r}")
print("\nwrote", OUT)

"""
Sample from a nanoGPT checkpoint trained on a SentencePiece-tokenized dataset.

nanoGPT's own sample.py only knows GPT-2 BPE and character-level vocabularies.
This one decodes with the SentencePiece model recorded in the dataset's meta.pkl
and stops each sample at the end-of-story token.

Usage (from any directory):
    python sample_uk.py --out_dir nanoGPT/out-uk-stories
    python sample_uk.py --out_dir nanoGPT/out-uk-stories --start "Жило-було кошеня" --num_samples 3
"""
import argparse
import os
import pickle
import sys
from contextlib import nullcontext

import sentencepiece as spm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NANOGPT = os.path.join(HERE, "nanoGPT")
sys.path.insert(0, NANOGPT)
from model import GPT, GPTConfig  # noqa: E402


def load_model(out_dir, device):
    checkpoint = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, checkpoint


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k, stop_id):
    """Like GPT.generate, but stops at stop_id (end of story)."""
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        if next_id.item() == stop_id:
            break
        idx = torch.cat((idx, next_id), dim=1)
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(NANOGPT, "out-uk-stories"))
    p.add_argument("--start", default="", help="beginning of the story; empty = model starts from scratch")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.out_dir, args.device)
    meta_path = os.path.join(NANOGPT, "data", checkpoint["config"]["dataset"], "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    sp = spm.SentencePieceProcessor(model_file=meta["model_file"])
    eos = meta["eos_id"]

    # training data is "...story</s>story</s>...", so </s> is the natural "start of a new story"
    start_ids = [eos] + sp.encode(args.start)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    ctx = nullcontext()
    if "cuda" in args.device:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)

    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"iter {checkpoint['iter_num']}, best val loss {checkpoint['best_val_loss']:.3f}\n")
    with ctx:
        for _ in range(args.num_samples):
            y = generate(model, x, args.max_new_tokens, args.temperature, args.top_k, eos)
            print(sp.decode(y[0, 1:].tolist()))
            print("---------------")


if __name__ == "__main__":
    main()

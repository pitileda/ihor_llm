"""
Build the bilingual Ukrainian/English nanoGPT dataset described in mix.json.

Stages (each one skips work whose output already exists, unless --force):
  extract    sample, clean and filter documents of every source up to its token budget
             -> <work_dir>/<mix>/docs/<source>.jsonl (+ .stats.json written when complete)
  tokenizer  train a SentencePiece BPE on a sample weighted like the mix
             -> tokenizer/<tokenizer name>.model
  encode     tokenize every source (mark_line_starts first), eos after each document,
             split train/val by document
             -> <work_dir>/<mix>/tokens/<source>.{train,val}.bin
  compose    take each source's share of total_tokens into one train.bin, plus val.bin
             and per-source val/<source>.bin, and meta.pkl -> this folder

Usage:
    python prepare.py --raw_dir /mnt/secure_ssd/llm/ua_data_set
    python prepare.py --raw_dir ... --stages extract --only wikisource_uk opus100_en_uk
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "datagen", "real"))
sys.path.insert(0, os.path.join(ROOT, "tokenizer"))
from corpus import Filters, doc_hash, estimate_size, iter_documents  # noqa: E402
from download import load_sources  # noqa: E402
from train_tokenizer import mark_line_starts  # noqa: E402

STAGES = ["extract", "tokenizer", "encode", "compose"]
EXTRACT_MARGIN = 1.15  # extract a bit more than the budget; compose trims to the exact share
ENCODE_BATCH = 2048


def load_mix(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        mix = json.load(f)
    total = sum(s["weight"] for s in mix["sources"].values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{path}: source weights sum to {total}, expected 1.0")
    return mix


def target_tokens(mix: dict, name: str) -> int:
    """Train tokens this source contributes to the mix."""
    return int(mix["total_tokens"] * mix["sources"][name]["weight"])


def is_val(text: str, fraction: float) -> bool:
    return doc_hash(text) % 100_000 < fraction * 100_000


def read_docs(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ---------------------------------------------------------------------------

def extract(mix, sources, names, raw_dir, work, seed, force):
    filters_base = mix["filters"]
    for name in names:
        out = os.path.join(work, "docs", f"{name}.jsonl")
        stats_path = out.replace(".jsonl", ".stats.json")
        if os.path.exists(stats_path) and not force:
            print(f"extract {name}: done before, skipping")
            continue
        cfg, src = mix["sources"][name], sources[name]
        filters = Filters(**filters_base, **({"min_edu_score": cfg["min_edu_score"]}
                                             if "min_edu_score" in cfg else {}))
        budget_chars = (target_tokens(mix, name) * (1 + mix["val_fraction"])
                        * mix["chars_per_token"][src["lang"]] * EXTRACT_MARGIN)
        rows, est_chars = estimate_size(raw_dir, name, src, seed, cfg.get("max_files", 0),
                                        filters.min_edu_score)
        keep = min(1.0, 1.3 * budget_chars / max(est_chars, 1))  # 1.3: room for filtered rows
        print(f"extract {name}: ~{est_chars / 1e9:.2f}B raw chars in {rows:,} rows, "
              f"budget {budget_chars / 1e9:.2f}B chars -> keep {keep:.2%} of rows")

        t0, chars, stats = time.time(), 0, {}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            for doc in iter_documents(raw_dir, name, src, filters, keep, seed,
                                      cfg.get("max_files", 0), stats):
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                chars += len(doc["text"])
                if chars >= budget_chars:
                    break
        os.replace(out + ".tmp", out)
        stats.update(chars=chars, budget_chars=int(budget_chars), keep=keep,
                     seconds=round(time.time() - t0))
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=1)
        short = " (source exhausted before the budget)" if chars < budget_chars else ""
        print(f"  {stats['kept']:,} docs, {chars / 1e9:.2f}B chars, filtered "
              f"{stats['filtered']:,} of {stats['rows']:,} rows read, {stats['seconds']} s{short}")


def train_tokenizer(mix, names, work, force):
    from train_tokenizer import train_bpe
    cfg = mix["tokenizer"]
    prefix = os.path.join(ROOT, "tokenizer", cfg["name"])
    if os.path.exists(prefix + ".model") and not force:
        print(f"tokenizer: {prefix}.model exists, skipping")
        return
    sample = os.path.join(work, "tokenizer_sample.txt")
    weights = {n: mix["sources"][n]["weight"] for n in names}
    with open(sample, "w", encoding="utf-8") as f:
        for name, w in weights.items():
            budget, chars = cfg["sample_chars"] * w / sum(weights.values()), 0
            for doc in read_docs(os.path.join(work, "docs", f"{name}.jsonl")):
                for line in doc["text"].split("\n"):  # SentencePiece trains on lines
                    if line.strip():
                        f.write(line + "\n")
                        chars += len(line)
                if chars >= budget:
                    break
            print(f"tokenizer sample: {name} {chars / 1e6:.0f}M chars")
    train_bpe(sample, prefix, cfg["vocab_size"])


def encode(mix, names, work, force):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(ROOT, "tokenizer", mix["tokenizer"]["name"] + ".model"))
    assert sp.get_piece_size() < 2 ** 16, "uint16 storage needs vocab < 65536"
    eos = sp.eos_id()
    for name in names:
        base = os.path.join(work, "tokens", name)
        if os.path.exists(base + ".json") and not force:
            print(f"encode {name}: done before, skipping")
            continue
        os.makedirs(os.path.dirname(base), exist_ok=True)
        t0, counts = time.time(), {"train": 0, "val": 0, "docs": 0, "chars": 0}
        with open(base + ".train.bin", "wb") as ftrain, open(base + ".val.bin", "wb") as fval:
            batch = []

            def flush():
                ids_batch = sp.encode([mark_line_starts(t) for t in batch], num_threads=os.cpu_count())
                for text, ids in zip(batch, ids_batch):
                    arr = np.array(ids + [eos], dtype=np.uint16)
                    split = "val" if is_val(text, mix["val_fraction"]) else "train"
                    (fval if split == "val" else ftrain).write(arr.tobytes())
                    counts[split] += len(arr)
                    counts["chars"] += len(text)
                counts["docs"] += len(batch)
                batch.clear()

            for doc in read_docs(os.path.join(work, "docs", f"{name}.jsonl")):
                batch.append(doc["text"])
                if len(batch) == ENCODE_BATCH:
                    flush()
            flush()
        counts["chars_per_token"] = round(counts["chars"] / max(counts["train"] + counts["val"], 1), 3)
        with open(base + ".json", "w") as f:
            json.dump(counts, f, indent=1)
        print(f"encode {name}: {counts['train'] / 1e6:.1f}M train + {counts['val'] / 1e6:.2f}M val "
              f"tokens, {counts['chars_per_token']} chars/token, {time.time() - t0:.0f} s")


def copy_tokens(src_path: str, dst, n: int) -> int:
    """Append the first n tokens of a .bin file to an open file; returns tokens copied."""
    arr = np.memmap(src_path, dtype=np.uint16, mode="r")
    n = min(n, len(arr))
    for i in range(0, n, 1 << 26):
        dst.write(arr[i:min(n, i + (1 << 26))].tobytes())
    return n


def compose(mix, sources, names, work):
    import sentencepiece as spm
    model_file = os.path.join(ROOT, "tokenizer", mix["tokenizer"]["name"] + ".model")
    sp = spm.SentencePieceProcessor(model_file=model_file)
    os.makedirs(os.path.join(HERE, "val"), exist_ok=True)
    report = {}
    with open(os.path.join(HERE, "train.bin"), "wb") as ftrain, open(os.path.join(HERE, "val.bin"), "wb") as fval:
        for name in names:
            base = os.path.join(work, "tokens", name)
            want = target_tokens(mix, name)
            got = copy_tokens(base + ".train.bin", ftrain, want)
            # val keeps the mix proportions too
            want_val = int(want * mix["val_fraction"]) or 1
            with open(os.path.join(HERE, "val", f"{name}.bin"), "wb") as fsrc:
                got_val = copy_tokens(base + ".val.bin", fsrc, want_val)
            copy_tokens(base + ".val.bin", fval, want_val)
            report[name] = {"train": got, "val": got_val, "target": want}
            flag = "" if got >= want else f"  SHORT by {(want - got) / 1e6:.1f}M"
            print(f"compose {name:18} {got / 1e6:8.1f}M train {got_val / 1e6:6.2f}M val{flag}")
    total = sum(r["train"] for r in report.values())
    print(f"compose total: {total / 1e6:.1f}M train tokens")
    for lang in ("uk", "en", "en-uk"):
        share = sum(r["train"] for n, r in report.items() if sources[n]["lang"] == lang) / max(total, 1)
        print(f"  {lang}: {share:.1%}")
    with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": sp.get_piece_size(), "tokenizer": "sentencepiece",
                     "model_file": model_file, "eos_id": sp.eos_id(), "mix": mix["name"],
                     "line_start_space": True,  # decode with unmark_line_starts
                     "sources": report}, f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", required=True, help="download.py's --out_dir")
    p.add_argument("--work_dir", help="intermediate files (default: <raw_dir>/work)")
    p.add_argument("--mix", default=os.path.join(HERE, "mix.json"))
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--only", nargs="+", help="limit extract/encode to these sources")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--force", action="store_true", help="redo stages whose output exists")
    args = p.parse_args()

    try:
        mix = load_mix(args.mix)
        sources = load_sources(os.path.join(ROOT, "datagen", "real", "sources.json"))
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"bad config: {e}")
    unknown = (set(mix["sources"]) | set(args.only or [])) - sources.keys()
    if unknown:
        sys.exit(f"unknown sources: {sorted(unknown)}")
    names = [n for n in mix["sources"] if not args.only or n in args.only]
    work = os.path.join(args.work_dir or os.path.join(args.raw_dir, "work"), mix["name"])
    os.makedirs(work, exist_ok=True)

    if "extract" in args.stages:
        extract(mix, sources, names, args.raw_dir, work, args.seed, args.force)
    if "tokenizer" in args.stages:
        train_tokenizer(mix, list(mix["sources"]), work, args.force)
    if "encode" in args.stages:
        encode(mix, names, work, args.force)
    if "compose" in args.stages:
        compose(mix, sources, list(mix["sources"]), work)


if __name__ == "__main__":
    main()

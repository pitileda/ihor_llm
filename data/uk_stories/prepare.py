"""
Prepare synthetic Ukrainian stories for nanoGPT.

  1. read stories from generate_stories.py JSONL files
  2. clean (same normalization as the tokenizer), drop exact duplicates
  3. train a SentencePiece BPE tokenizer on the stories (unless --tokenizer is given)
  4. split train/val by story, encode, put </s> (eos) after every story
  5. write train.bin / val.bin (uint16) and meta.pkl into this folder

Usage:
    python prepare.py --inputs ../../datagen/stories/luna_*.jsonl --vocab_size 8192
    python prepare.py --inputs ... --tokenizer ../../tokenizer/uk_bpe16384.model   # reuse a tokenizer

nanoGPT reads data/<dataset>/ relative to its own folder, so this folder is symlinked
as nanoGPT/data/uk_stories; train with dataset = 'uk_stories'.
"""
import argparse
import glob
import json
import os
import pickle
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tokenizer"))
from train_tokenizer import clean, show_stats, train_bpe  # noqa: E402


# generator artifacts that stand for a line break: HTML <br> and a literal backslash-n
LINE_BREAK_ARTIFACTS = re.compile(r"<br\s*/?>|\\n", re.I)
# after those are fixed, any of these means a broken story: Latin letters (English
# phrases, Latin look-alikes inside Cyrillic words like "совa") or markup leftovers
FOREIGN = re.compile(r"[A-Za-z\\<>]")


def normalize(text):
    text = clean(LINE_BREAK_ARTIFACTS.sub("\n", text)).strip()
    # one "\n" per paragraph break: blank lines carry no extra meaning
    return re.sub(r"[ \t]*\n\s*", "\n", text)


def load_stories(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        sys.exit(f"no files match {patterns}")
    seen, stories = set(), []
    for path in files:
        n_kept = n_dup = n_foreign = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = normalize(json.loads(line)["text"])
                if FOREIGN.search(text):
                    n_foreign += 1
                    continue
                if text in seen:
                    n_dup += 1
                    continue
                seen.add(text)
                stories.append(text)
                n_kept += 1
        print(f"{path}: kept {n_kept} stories, dropped {n_dup} duplicates "
              f"and {n_foreign} with Latin letters or markup")
    return stories


def encode_to_bin(sp, stories, path):
    eos = sp.eos_id()
    ids = sp.encode(stories, num_threads=os.cpu_count())
    total = sum(len(x) + 1 for x in ids)
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
    pos = 0
    for x in ids:
        arr[pos:pos + len(x)] = x
        arr[pos + len(x)] = eos
        pos += len(x) + 1
    arr.flush()
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="story JSONL files or globs")
    p.add_argument("--tokenizer", help="existing SentencePiece .model; default: train a new one")
    p.add_argument("--vocab_size", type=int, default=8192, help="when training a new tokenizer")
    p.add_argument("--val_fraction", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    stories = load_stories(args.inputs)
    random.Random(args.seed).shuffle(stories)
    n_val = max(1, int(len(stories) * args.val_fraction))
    val, train = stories[:n_val], stories[n_val:]
    print(f"{len(stories)} unique stories -> train {len(train)}, val {n_val}")

    # tokenizer is trained on train stories only, one paragraph per line
    corpus = os.path.join(HERE, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for s in train:
            f.write(s + "\n")

    if args.tokenizer:
        import sentencepiece as spm
        model_file = os.path.abspath(args.tokenizer)
        sp = spm.SentencePieceProcessor(model_file=model_file)
    else:
        prefix = os.path.join(ROOT, "tokenizer", f"uk_stories_bpe{args.vocab_size}")
        sp = train_bpe(corpus, prefix, args.vocab_size)
        model_file = prefix + ".model"
    show_stats(sp, corpus)

    vocab_size = sp.get_piece_size()
    assert vocab_size < 2 ** 16, "uint16 storage needs vocab < 65536"

    n_train = encode_to_bin(sp, train, os.path.join(HERE, "train.bin"))
    n_val_tok = encode_to_bin(sp, val, os.path.join(HERE, "val.bin"))
    with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "tokenizer": "sentencepiece",
                     "model_file": model_file, "eos_id": sp.eos_id()}, f)

    # sanity check: decode the start of val.bin back to text
    arr = np.memmap(os.path.join(HERE, "val.bin"), dtype=np.uint16, mode="r")
    first = arr[: int(np.argmax(arr == sp.eos_id()))].tolist()
    print(f"\nround trip ok: {sp.decode(first) == val[0]}")
    print(f"train: {n_train:,} tokens   val: {n_val_tok:,} tokens   vocab: {vocab_size}")
    print(f"avg tokens per story: {n_train / len(train):.0f}")


if __name__ == "__main__":
    main()

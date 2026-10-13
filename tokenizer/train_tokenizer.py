"""
Train a Ukrainian BPE tokenizer with SentencePiece on Ukrainian Wikipedia.

Usage:
    pip install sentencepiece datasets
    python train_tokenizer.py --vocab_size 8192 --sample_mb 300
    python train_tokenizer.py --vocab_size 8192 --input corpus.txt --name uk_stories_bpe  # own text

Outputs (in this folder):
    uk_wiki_sample.txt   - text sample used for training
    uk_bpe<vocab>.model  - tokenizer (used for encoding/decoding)
    uk_bpe<vocab>.vocab  - human-readable list of learned tokens
"""
import argparse
import itertools
import os
import re

import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))

# Ukrainian text uses several apostrophe variants: ' ’ ʼ `  -> unify them,
# otherwise the tokenizer wastes vocabulary on duplicates like "м'ясо" / "м’ясо".
APOSTROPHES = re.compile(r"[’ʼ`ʹ‘]")


def clean(text: str) -> str:
    text = APOSTROPHES.sub("'", text)
    text = text.replace(" ", " ")  # non-breaking space
    return text


def dump_sample(path: str, sample_mb: int) -> None:
    """Stream Ukrainian Wikipedia and write ~sample_mb MB of paragraphs, one per line."""
    if os.path.exists(path):
        print(f"{path} already exists, reusing it")
        return
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.uk", split="train", streaming=True)
    limit = sample_mb * 1024 * 1024
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for article in ds:
            for para in clean(article["text"]).split("\n"):
                para = para.strip()
                if len(para) < 20:  # skip headings, empty lines, list fragments
                    continue
                line = para + "\n"
                f.write(line)
                written += len(line.encode("utf-8"))
            if written >= limit:
                break
    print(f"wrote {written / 1e6:.0f} MB to {path}")


def train_bpe(input_path: str, prefix: str, vocab_size: int) -> spm.SentencePieceProcessor:
    """Train BPE on a text file (one paragraph per line); writes prefix.model / prefix.vocab."""
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=0.9995,    # keep almost all characters; ultra-rare ones go to byte fallback
        byte_fallback=True,           # unknown chars (emoji, Chinese...) -> UTF-8 bytes, never <unk>
        split_digits=True,            # numbers split into single digits (helps arithmetic, saves vocab)
        input_sentence_size=3_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=16384,    # bytes; wiki paragraphs can be long
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,  # eos is used as the document separator
        # keep text exactly as clean() leaves it: the default NFKC rule turns "\n" into a space,
        # losing paragraphs and dialogue lines; newline gets its own token instead
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        user_defined_symbols=["\n"],
        num_threads=os.cpu_count(),
    )
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


def show_stats(sp: spm.SentencePieceProcessor, sample_path: str) -> None:
    demo = "Українська мова має багату морфологію: книга, книги, книзі, книгою, книгами."
    pieces = sp.encode(demo, out_type=str)
    print("\nexample:", pieces)
    print(f"{len(demo)} chars -> {len(pieces)} tokens")

    # Compression on held-out-ish text: average characters per token (higher = better)
    with open(sample_path, encoding="utf-8") as f:
        lines = list(itertools.islice(f, 20000))
    chars = sum(len(l) for l in lines)
    tokens = sum(len(x) for x in sp.encode(lines))
    print(f"avg chars per token: {chars / tokens:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--sample_mb", type=int, default=300)
    p.add_argument("--input", help="train on this text file instead of a Wikipedia sample")
    p.add_argument("--name", default="uk_bpe", help="output name prefix (vocab size is appended)")
    args = p.parse_args()

    sample_path = args.input
    if not sample_path:
        sample_path = os.path.join(HERE, "uk_wiki_sample.txt")
        dump_sample(sample_path, args.sample_mb)

    sp = train_bpe(sample_path, os.path.join(HERE, f"{args.name}{args.vocab_size}"), args.vocab_size)
    show_stats(sp, sample_path)


if __name__ == "__main__":
    main()

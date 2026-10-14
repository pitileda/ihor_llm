"""
Read, clean and filter documents from the raw corpora downloaded by download.py.

Every source in sources.json names a "reader" that knows its parquet layout:
  wiki   title + text (Wikipedia, Wikisource)
  text   a plain "text" column (Kobza, FineWeb-Edu)
  pdf    FinePDFs: text plus per-page quality scores; skips the test split
  pairs  OPUS translation pairs, grouped into documents of EN:/UK: lines

iter_documents() yields cleaned documents in a random but reproducible order,
keeping each row with probability `keep`, so a fixed budget is spread evenly over
the whole source instead of coming from its first file.
"""
import functools
import glob
import hashlib
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass

import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tokenizer"))
from train_tokenizer import clean as normalize_chars  # noqa: E402  apostrophes, nbsp

CYRILLIC = re.compile(r"[а-яА-ЯіїєґІЇЄҐёЁъЪыЫэЭ]")
LATIN = re.compile(r"[a-zA-Z]")
RUSSIAN_ONLY = re.compile(r"[ёЁъЪыЫэЭ]")
UKRAINIAN_ONLY = re.compile(r"[іїєґІЇЄҐ']")
LETTER = re.compile(r"[^\W\d_]")
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")         # "сло-\nво" from PDF line wrapping
PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.M)
TRAILING_SPACE = re.compile(r"[ \t]+\n")
MANY_NEWLINES = re.compile(r"\n{3,}")
MANY_SPACES = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class Filters:
    min_chars: int = 200
    max_chars: int = 100_000         # longer documents are cut at a paragraph boundary
    min_letter_ratio: float = 0.6    # letters / non-space characters: drops tables, code, junk
    min_script_ratio: float = 0.8    # share of letters in the document's own script
    max_russian_ratio: float = 0.002  # ы э ъ ё per Cyrillic letter, for Ukrainian text
    min_edu_score: float = 0.0       # pdf reader: mean FineWeb-Edu score of the pages


def clean_text(text: str, pdf: bool = False) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = normalize_chars(text)
    if pdf:
        text = HYPHEN_BREAK.sub(r"\1\2", text)
        text = PAGE_NUMBER_LINE.sub("", text)
    text = TRAILING_SPACE.sub("\n", text)
    text = MANY_SPACES.sub(" ", text)
    return MANY_NEWLINES.sub("\n\n", text).strip()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[:cut if cut > max_chars // 2 else max_chars].rstrip()


def passes(text: str, lang: str, f: Filters) -> bool:
    if len(text) < f.min_chars:
        return False
    letters = len(LETTER.findall(text))
    non_space = len(text) - text.count(" ") - text.count("\n")
    if not letters or letters / max(non_space, 1) < f.min_letter_ratio:
        return False
    if lang == "uk":
        cyr = len(CYRILLIC.findall(text))
        if cyr / letters < f.min_script_ratio:
            return False
        if len(RUSSIAN_ONLY.findall(text)) / max(cyr, 1) > f.max_russian_ratio:
            return False
    elif lang == "en":
        if len(LATIN.findall(text)) / letters < f.min_script_ratio:
            return False
    return True


def doc_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


# ---------------------------------------------------------------------------
# Readers: (parquet file, row group) -> raw documents as (text, meta) pairs

def _wiki(table):
    for title, text in zip(table.column("title").to_pylist(), table.column("text").to_pylist()):
        yield (f"{title}\n\n{text}" if title and not text.startswith(title) else text), {}


def _text(table):
    sources = table.column("source").to_pylist() if "source" in table.column_names else None
    for i, text in enumerate(table.column("text").to_pylist()):
        yield text, ({"source": sources[i]} if sources else {})


def _pdf(table):
    for text, scores in zip(table.column("text").to_pylist(), table.column("fw_edu_scores").to_pylist()):
        yield text, {"edu": sum(scores) / len(scores) if scores else 0.0}


@functools.lru_cache(maxsize=1)
def slavic_detector():
    """Language detector for the Cyrillic languages OPUS "uk" sides get mixed up with."""
    from lingua import Language, LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_languages(
        Language.UKRAINIAN, Language.RUSSIAN, Language.BELARUSIAN, Language.BULGARIAN).build()


def ukrainian_mask(sentences: list) -> list:
    """True for sentences that are Ukrainian; OPUS "uk" sides are often Russian or garbage."""
    from lingua import Language
    langs = slavic_detector().detect_languages_in_parallel_of(sentences)
    return [bool(CYRILLIC.search(s)) and not RUSSIAN_ONLY.search(s) and lang != Language.RUSSIAN
            and (lang == Language.UKRAINIAN or bool(UKRAINIAN_ONLY.search(s)))
            for s, lang in zip(sentences, langs)]


def _pairs(table, rng, group=16):
    """Translation pairs -> documents of `group` pairs, random direction per pair."""
    pairs = [((p.get("en") or "").strip(), normalize_chars((p.get("uk") or "").strip()))
             for p in table.column("translation").to_pylist()]
    pairs = [(en, uk) for en, uk in pairs if en and uk and not CYRILLIC.search(en)]
    lines = []
    for (en, uk), ok in zip(pairs, ukrainian_mask([uk for _, uk in pairs])):
        if not ok:
            continue
        lines.append(f"EN: {en}\nUK: {uk}" if rng.random() < 0.5 else f"UK: {uk}\nEN: {en}")
        if len(lines) == group:
            yield "\n\n".join(lines), {}
            lines = []
    if lines:
        yield "\n\n".join(lines), {}


READERS = {
    "wiki": (["title", "text"], _wiki),
    "text": (["text", "source"], _text),
    "pdf": (["text", "fw_edu_scores"], _pdf),
    "pairs": (["translation"], _pairs),
}


def source_files(raw_dir: str, name: str, reader: str) -> list:
    files = sorted(glob.glob(os.path.join(raw_dir, name, "**", "*.parquet"), recursive=True))
    files = [f for f in files if "/.cache/" not in f]
    if reader == "pdf":
        files = [f for f in files if "/test/" not in f]
    elif reader == "pairs":
        files = [f for f in files if os.path.basename(f).startswith("train")]
    return files


def source_units(files: list) -> list:
    """(file, row_group, n_rows) for every row group of every file."""
    units = []
    for f in files:
        meta = pq.ParquetFile(f).metadata
        units += [(f, i, meta.row_group(i).num_rows) for i in range(meta.num_row_groups)]
    return units


def estimate_size(raw_dir: str, name: str, src: dict, seed: int, max_files: int = 0,
                  probe_rows: int = 2000) -> tuple:
    """(rows, approximate raw chars) of a source, from parquet metadata and a sample."""
    reader = src["reader"]
    files = source_files(raw_dir, name, reader)
    if not files:
        raise FileNotFoundError(f"no parquet files for '{name}' in {raw_dir}")
    random.Random(f"{seed}:{name}").shuffle(files)  # same file choice as iter_documents
    if max_files:
        files = files[:max_files]
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    column = "translation" if reader == "pairs" else "text"
    batch = next(pq.ParquetFile(files[0]).iter_batches(batch_size=probe_rows, columns=[column]))
    values = batch.column(0).to_pylist()
    if reader == "pairs":
        lengths = [len(v.get("en") or "") + len(v.get("uk") or "") + 9 for v in values]
    else:
        lengths = [len(v or "") for v in values]
    return rows, rows * sum(lengths) / max(len(lengths), 1)


def iter_documents(raw_dir: str, name: str, src: dict, filters: Filters, keep: float,
                   seed: int, max_files: int = 0, stats: dict = None):
    """Yield cleaned documents of one source as dicts {"text", "lang", **meta}."""
    reader = src["reader"]
    columns, read = READERS[reader]
    rng = random.Random(f"{seed}:{name}")
    files = source_files(raw_dir, name, reader)
    if not files:
        raise FileNotFoundError(f"no parquet files for '{name}' in {raw_dir}")
    rng.shuffle(files)
    if max_files:
        files = files[:max_files]
    units = source_units(files)
    rng.shuffle(units)
    stats = stats if stats is not None else {}
    for key in ("rows", "kept", "filtered"):
        stats.setdefault(key, 0)
    seen = set()
    for path, rg, _ in units:
        pf = pq.ParquetFile(path)
        table = pf.read_row_group(rg, columns=[c for c in columns if c in pf.schema_arrow.names])
        rows = read(table, rng) if reader == "pairs" else read(table)
        for text, meta in rows:
            stats["rows"] += 1
            if keep < 1.0 and rng.random() >= keep:
                continue
            if reader == "pdf" and meta["edu"] < filters.min_edu_score:
                stats["filtered"] += 1
                continue
            text = truncate(clean_text(text or "", pdf=reader == "pdf"), filters.max_chars)
            if reader != "pairs" and not passes(text, src["lang"], filters):
                stats["filtered"] += 1
                continue
            h = doc_hash(text)
            if h in seen:
                stats["filtered"] += 1
                continue
            seen.add(h)
            stats["kept"] += 1
            yield {"text": text, "lang": src["lang"], **meta}

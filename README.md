# ihor_llm

A small GPT trained from scratch to write Ukrainian text, with as few weights as possible.
It follows the TinyStories recipe: an LLM API generates simple Ukrainian children's stories,
we train a SentencePiece BPE tokenizer on them, and then train a ~15M-parameter GPT-2-style model
with [nanoGPT](https://github.com/karpathy/nanoGPT).

## Setup

nanoGPT is a **git submodule**. Clone with:

```bash
git clone --recurse-submodules <repo-url> ihor_llm
# or, in an existing clone:
git submodule update --init
```

nanoGPT reads datasets from `nanoGPT/data/<name>/`, so link our data folder into it
(the link stays local; it is not committed to the submodule):

```bash
ln -s ../../data/uk_stories nanoGPT/data/uk_stories
echo -e "data/uk_stories\nout-*" >> .git/modules/nanoGPT/info/exclude   # keep the submodule clean
```

Python environment: 3.10+ is required (current torch and the `anthropic` SDK don't support 3.8). Use either one:

```bash
# a) venv (needs a system Python 3.10+, e.g. `sudo apt install python3.12 python3.12-venv`)
python3.12 -m venv .venv
source .venv/bin/activate

# b) conda
conda create -n ihor python=3.12 && conda activate ihor
```

Then install the dependencies:

```bash
pip install -U pip
pip install -r requirements.txt   # pick the torch build matching your CUDA, see pytorch.org
```

API keys go in `.env` (git-ignored):

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

## Layout

| Path | What |
|---|---|
| `nanoGPT/` | submodule: `model.py`, `train.py` |
| `tokenizer/train_tokenizer.py` | SentencePiece BPE trainer (`clean()`, `train_bpe()`); standalone run trains on a Ukrainian Wikipedia sample |
| `datagen/synthetic/generate_stories.py` | synthetic story generator (Claude / OpenAI, direct or batch API) → `datagen/synthetic/stories/*.jsonl` |
| `datagen/synthetic/story_prompts.json` | prompt data: system prompt, genres, audiences, features, seed word lists (edit to change topics) |
| `datagen/real/download.py` | downloads the real corpora listed in `datagen/real/sources.json` (Kobza, Wikipedia, Wikisource, FinePDFs, FineWeb-Edu, OPUS) from the Hugging Face Hub, pinned and resumable |
| `data/uk_stories/prepare.py` | stories → tokenizer → `train.bin`, `val.bin`, `meta.pkl` |
| `config/train_uk_stories.py` | nanoGPT training config |
| `sample_uk.py` | generate text from a checkpoint with the matching tokenizer |

## Real data: bilingual uk/en base model

```bash
# 1. download raw corpora (~262 GB, resumable, writes manifest.json)
cd datagen/real
python download.py --out_dir /path/to/raw --list   # sources and sizes
python download.py --out_dir /path/to/raw

# 2. extract + clean -> tokenizer -> encode -> compose (mix in data/uk_en/mix.json)
#    ~1.5 h on 24 cores; intermediate files go to <raw_dir>/work (~10 GB)
cd ../../data/uk_en
python prepare.py --raw_dir /path/to/raw

# 3. train model S (~42M params, 1B tokens; ~2 s/iter, ~4.5 h on an RTX A2000 6 GB)
ln -sfn ../../data/uk_en ../../nanoGPT/data/uk_en
cd ../../nanoGPT
python train.py ../config/train_uk_en_s.py

# 4. sample (prompt continuation; the model is a base model, not a chatbot)
cd ..
python sample_uk.py --out_dir nanoGPT/out-uk-en-s --start "Тарас Шевченко народився"
```

Resulting mix (999.9M train tokens): Ukrainian 59.4%, English 39.0%, en-uk translation
pairs 1.6%. Measured tokenizer efficiency: ~3.5 chars/token for Ukrainian, 3.5–4.0 for English.

| Path | What |
|---|---|
| `datagen/real/sources.json` | where each corpus comes from and which reader parses it |
| `datagen/real/corpus.py` | readers, cleaning, quality and language filters |
| `data/uk_en/mix.json` | the recipe: total tokens, share per source, filters, tokenizer |
| `data/uk_en/prepare.py` | builds `train.bin`, `val.bin`, `val/<source>.bin`, `meta.pkl` |

## Synthetic stories pipeline

```bash
set -a; source .env; set +a

# 1. generate stories (OpenAI Batch API, 50% cheaper; resumable via a state file)
cd datagen/synthetic
python generate_stories.py --backend openai-batch --model gpt-6-luna --json \
    --n_requests 100000 --out stories/luna_v1.jsonl

# 2. tokenizer + binary dataset
cd ../../data/uk_stories
python prepare.py --inputs ../../datagen/synthetic/stories/luna_*.jsonl --vocab_size 8192

# 3. train (set max_iters ≈ epochs * train_tokens / 65536 in the config)
cd ../../nanoGPT
python train.py ../config/train_uk_stories.py

# 4. sample
cd ..
python sample_uk.py --start "Жило-було кошеня"
```

A checkpoint only works with the tokenizer it was trained with: `meta.pkl` records
the tokenizer path, and `sample_uk.py` follows it. Don't overwrite a `.model` file
that an existing checkpoint depends on.

Generated data, tokenizers, `.bin` files and checkpoints are not committed; rerun the
pipeline to reproduce them.

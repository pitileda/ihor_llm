"""
Generate a synthetic Ukrainian "TinyStories"-style dataset with an LLM.

Variety comes from randomizing every request (TinyStories recipe): each story gets
3 random words (noun, verb, adjective) it must use, plus a random genre, audience
and 1-2 narrative features. Several stories are generated per request to save cost.
The system prompt and all these lists are in story_prompts.json (--prompt_data).

Backends:
  anthropic-batch  Claude Message Batches API (50% cheaper, async, usually < 1 h)
  anthropic        Claude Messages API, direct concurrent calls (good for pilots)
  openai-batch     OpenAI Batch API (50% cheaper, async, within 24 h)
  openai           any OpenAI-compatible server: local llama.cpp (gpt-oss),
                   OpenRouter, DeepInfra, Together, ...

Examples:
  # look at a prompt without calling anything
  python generate_stories.py --dry_run

  # pilot: 40 requests x 5 stories with Claude Haiku 5.5
  python generate_stories.py --backend anthropic --model claude-haiku-5-5 --n_requests 40 \
      --out stories/pilot_haiku.jsonl

  # pilot with gpt-oss-120b on OpenRouter
  OPENAI_API_KEY=... python generate_stories.py --backend openai --model openai/gpt-oss-120b \
      --base_url https://openrouter.ai/api/v1 --n_requests 40 --price_in 0.04 --price_out 0.17 \
      --extra_body '{"reasoning": {"effort": "low"}}' --out stories/pilot_gptoss120b.jsonl

  # GPT-6 Luna via OpenAI Batch API with structured JSON output (50% off)
  OPENAI_API_KEY=... python generate_stories.py --backend openai-batch --model gpt-6-luna --json \
      --base_url https://api.openai.com/v1 --price_in 0.10 --price_out 0.50 \
      --extra_body '{"max_completion_tokens": 8000, "reasoning_effort": "low"}' --n_requests 100000

  # full run via Claude batches (resumable: rerun the same command after an interruption)
  python generate_stories.py --backend anthropic-batch --model claude-haiku-5-5 --n_requests 100000

Output: JSONL, one story per line:
  {"req_id": 17, "model": ..., "genre": ..., "audience": ..., "features": [...], "ending": ... | null,
   "words": [...], "text": ...}
"""
import argparse
import functools
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Prompt data: system prompt, genres, audiences, features and the word lists live in
# story_prompts.json, so they can be edited without touching the code.

DEFAULT_PROMPT_DATA = os.path.join(HERE, "story_prompts.json")

# Ukrainian never uses these letters; their presence means Russian text slipped in.
RUSSIAN_LETTERS = re.compile(r"[ыэъёЫЭЪЁ]")


@dataclass(frozen=True)
class PromptData:
    system: str
    genres: tuple
    audiences: tuple
    features: tuple
    endings: tuple             # mutually exclusive: at most one per request
    ending_probability: float  # share of requests that get an ending instruction
    nouns: tuple       # nominative singular (or plural-only nouns)
    verbs: tuple       # infinitives; the model inflects all words as needed
    adjectives: tuple  # masculine nominative
    fingerprint: str   # hash of the file content: tells prompt versions apart


def load_prompt_data(path: str) -> PromptData:
    """Load and validate the prompt data file; raises ValueError on a malformed file."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    list_fields = ["genres", "audiences", "features", "endings", "nouns", "verbs", "adjectives"]
    expected = {"system", "ending_probability", *list_fields}
    if raw.keys() != expected:
        raise ValueError(f"{path}: expected keys {sorted(expected)}, got {sorted(raw)}")
    if not isinstance(raw["system"], str) or not raw["system"].strip():
        raise ValueError(f"{path}: 'system' must be a non-empty string")
    p_end = raw["ending_probability"]
    if isinstance(p_end, bool) or not isinstance(p_end, (int, float)) or not 0 <= p_end <= 1:
        raise ValueError(f"{path}: 'ending_probability' must be a number in [0, 1]")
    lists = {}
    for name in list_fields:
        items = raw[name]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}: '{name}' must be a non-empty list")
        if not all(isinstance(x, str) and x.strip() for x in items):
            raise ValueError(f"{path}: '{name}' must contain only non-empty strings")
        dups = sorted({x for x in items if items.count(x) > 1})
        if dups:
            raise ValueError(f"{path}: duplicates in '{name}': {dups}")
        lists[name] = tuple(items)
    russian = [x for x in (raw["system"], *sum(lists.values(), ())) if RUSSIAN_LETTERS.search(x)]
    if russian:
        raise ValueError(f"{path}: non-Ukrainian letters in {russian}")
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return PromptData(system=raw["system"], ending_probability=float(p_end), **lists,
                      fingerprint=hashlib.sha256(canonical).hexdigest()[:12])


SEPARATOR = "==="

# Claude prices, $ per 1M tokens (input, output), standard rates; batch = 50%.
CLAUDE_PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


FORMAT_SEPARATOR = f"Не додавай заголовків, номерів чи пояснень. Розділяй тексти рядком, що містить лише {SEPARATOR}"
FORMAT_JSON = ("Поверни JSON з полем \"stories\": масив рядків, по одному тексту в кожному. "
               "Не додавай до текстів заголовків чи номерів.")

# structured output for OpenAI: guarantees a parseable list of stories
STORIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "stories",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"stories": {"type": "array", "items": {"type": "string"}}},
            "required": ["stories"],
            "additionalProperties": False,
        },
    },
}


def make_request(data: PromptData, req_id: int, seed: int, stories_per_request: int,
                 json_mode: bool = False) -> dict:
    """Deterministic random spec for one request: same data + req_id -> same prompt."""
    rng = random.Random(seed * 1_000_003 + req_id)
    genre = rng.choice(data.genres)
    audience = rng.choice(data.audiences)
    features = rng.sample(data.features, rng.choice([1, 1, 2]))
    ending = rng.choice(data.endings) if rng.random() < data.ending_probability else None
    stories = [[rng.choice(data.nouns), rng.choice(data.verbs), rng.choice(data.adjectives)]
               for _ in range(stories_per_request)]

    words_lines = "\n".join(
        f"Текст {i + 1}: «{n}», «{v}», «{a}»" for i, (n, v, a) in enumerate(stories)
    )
    prompt = f"""Напиши {stories_per_request} різних коротких текстів українською мовою.

Жанр: {genre}.
Читач: {audience}.

Вимоги:
- Використовуй лише прості слова та короткі речення, зрозумілі читачеві.
- Кожен текст — 120–250 слів, з різними персонажами й сюжетами.
- {" ".join(features + [ending] if ending else features)}
- Кожен текст має природно містити свої три слова (у будь-якій граматичній формі):
{words_lines}

{FORMAT_JSON if json_mode else FORMAT_SEPARATOR}"""
    return {"req_id": req_id, "genre": genre, "audience": audience,
            "features": features, "ending": ending, "stories": stories,
            "system": data.system, "prompt": prompt}


# leaked headers like "Текст 2", or a short title line ("Сніг, що сяяв") before the text
HEADER = re.compile(r"^\s*(?:(?:Текст|Історія|Казка)\s*\d+[.:]?|[^\n.!?»…]{1,60})\s*\n\s*\n", re.I)


def split_stories(spec: dict, text: str, model: str) -> list:
    try:
        parts = json.loads(text)["stories"]
    except (ValueError, KeyError, TypeError):
        parts = re.split(rf"^\s*{SEPARATOR}\s*$", text, flags=re.M)
    out = []
    for i, p in enumerate(parts):
        p = HEADER.sub("", p.strip(), count=1).strip()
        if len(p) < 300 or RUSSIAN_LETTERS.search(p):
            continue
        words = spec["stories"][i] if i < len(spec["stories"]) else []
        out.append({"req_id": spec["req_id"], "model": model, "genre": spec["genre"],
                    "audience": spec["audience"], "features": spec["features"],
                    "ending": spec["ending"], "words": words, "text": p})
    return out


# ---------------------------------------------------------------------------
# Batch bookkeeping shared by both batch backends

class BatchState:
    """Submitted batches, persisted in <out>.batches.json so an interrupted run resumes.

    Results are matched back to prompts by req_id, so a batch must be collected with
    the same prompt data it was submitted with: if the data file changed while a
    batch is still uncollected, refuse to run instead of mislabeling its stories.
    """

    def __init__(self, path: str, fingerprint: str):
        self.path = path
        self.fingerprint = fingerprint
        self.batches = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.batches = json.load(f)["batches"]
        stale = [b["id"] for b in self.pending() if b.get("prompt_data", fingerprint) != fingerprint]
        if stale:
            raise ValueError(f"batches {stale} were submitted with other prompt data; "
                             f"restore that version of the file to collect them")

    def pending(self) -> list:
        return [b for b in self.batches if not b["collected"]]

    def in_flight(self) -> set:
        return {rid for b in self.pending() for rid in b["req_ids"]}

    def add(self, batch_id: str, req_ids: list):
        self.batches.append({"id": batch_id, "req_ids": req_ids, "collected": False,
                             "prompt_data": self.fingerprint})
        self.save()

    def mark_collected(self, batch: dict):
        batch["collected"] = True
        self.save()

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"batches": self.batches}, f, indent=1)
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written state file


# ---------------------------------------------------------------------------
# Claude

def claude_params(model: str, spec: dict, max_tokens: int) -> dict:
    """Short creative writing needs no deep reasoning: minimize thinking tokens."""
    params = {"model": model, "max_tokens": max_tokens, "system": spec["system"],
              "messages": [{"role": "user", "content": spec["prompt"]}],
              "output_config": {"effort": "low"}}
    if model.startswith("claude-haiku-5"):
        params["thinking"] = {"type": "disabled"}
    elif model == "claude-sonnet-5-5":
        params["thinking"] = {"type": "between_tools"}
    # Opus 5.5 cannot disable thinking; low effort keeps it short.
    return params


def message_text(msg) -> str:
    return "".join(b.text for b in msg.content if b.type == "text")


def run_anthropic(specs, args, sink, stats):
    import anthropic
    client = anthropic.Anthropic()

    def call(spec):
        msg = client.messages.create(**claude_params(args.model, spec, args.max_tokens))
        return spec, msg

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(call, s) for s in specs]
        for fut in as_completed(futures):
            try:
                spec, msg = fut.result()
            except anthropic.APIError as e:
                print(f"request failed: {e}", file=sys.stderr)
                stats["failed"] += 1
                continue
            stats["in"] += msg.usage.input_tokens
            stats["out"] += msg.usage.output_tokens
            if msg.stop_reason == "refusal":
                stats["failed"] += 1
                continue
            sink(split_stories(spec, message_text(msg), args.model))


def run_anthropic_batch(specs, args, sink, stats, state: BatchState):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    # submit requests not waiting in an uncollected batch; requests that failed in an
    # already collected batch have no output lines, so a rerun resubmits them
    in_flight = state.in_flight()
    todo = [s for s in specs if s["req_id"] not in in_flight]
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=f"r{s['req_id']}",
                    params=MessageCreateParamsNonStreaming(
                        **claude_params(args.model, s, args.max_tokens)))
            for s in chunk
        ])
        state.add(batch.id, [s["req_id"] for s in chunk])
        print(f"submitted batch {batch.id} with {len(chunk)} requests")

    spec_by_id = {s["req_id"]: s for s in specs}
    for b in state.pending():
        while True:
            batch = client.messages.batches.retrieve(b["id"])
            if batch.processing_status == "ended":
                break
            c = batch.request_counts
            print(f"{b['id']}: processing={c.processing} succeeded={c.succeeded} errored={c.errored}")
            time.sleep(60)
        for res in client.messages.batches.results(b["id"]):
            rid = int(res.custom_id[1:])
            if res.result.type != "succeeded" or rid not in spec_by_id:
                stats["failed"] += 1
                continue
            msg = res.result.message
            stats["in"] += msg.usage.input_tokens
            stats["out"] += msg.usage.output_tokens
            if msg.stop_reason == "refusal":
                stats["failed"] += 1
                continue
            sink(split_stories(spec_by_id[rid], message_text(msg), args.model))
        state.mark_collected(b)


# ---------------------------------------------------------------------------
# OpenAI-compatible (llama.cpp server, OpenRouter, DeepInfra, ...)

def openai_body(spec, args, extra):
    body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 1.0,
            "messages": [{"role": "system", "content": spec["system"]},
                         {"role": "user", "content": spec["prompt"]}], **extra}
    # OpenAI reasoning models reject max_tokens/temperature; pass
    # --extra_body '{"max_completion_tokens": 8000}' for them
    if "max_completion_tokens" in extra:
        body.pop("max_tokens")
        body.pop("temperature")
    if args.json:
        body["response_format"] = STORIES_SCHEMA
    return body


def handle_openai_response(spec, resp, args, sink, stats):
    usage = resp.get("usage") or {}
    stats["in"] += usage.get("prompt_tokens", 0)
    stats["out"] += usage.get("completion_tokens", 0)  # includes reasoning tokens
    stats["reasoning"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    text = resp["choices"][0]["message"].get("content") or ""
    sink(split_stories(spec, text, args.model))


def openai_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"
    # retry dropped connections and 5xx/429 on GETs (status polls, result downloads);
    # POSTs are not retried here so a flaky upload can't create a duplicate paid batch
    retry = Retry(total=8, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def run_openai(specs, args, sink, stats):
    session = openai_session()
    url = args.base_url.rstrip("/") + "/chat/completions"
    extra = json.loads(args.extra_body) if args.extra_body else {}

    def call(spec):
        r = session.post(url, json=openai_body(spec, args, extra), timeout=1800)
        r.raise_for_status()
        return spec, r.json()

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(call, s) for s in specs]
        for fut in as_completed(futures):
            try:
                spec, resp = fut.result()
            except Exception as e:
                print(f"request failed: {e}", file=sys.stderr)
                stats["failed"] += 1
                continue
            handle_openai_response(spec, resp, args, sink, stats)


def run_openai_batch(specs, args, sink, stats, state: BatchState):
    """OpenAI Batch API: upload a JSONL file of requests, poll, download results."""
    session = openai_session()
    base = args.base_url.rstrip("/")
    extra = json.loads(args.extra_body) if args.extra_body else {}

    def ok(r):
        r.raise_for_status()
        return r

    in_flight = state.in_flight()
    todo = [s for s in specs if s["req_id"] not in in_flight]
    batch_size = min(args.batch_size, 50_000)  # OpenAI limit per batch
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        jsonl = "".join(json.dumps({"custom_id": f"r{s['req_id']}", "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": openai_body(s, args, extra)}, ensure_ascii=False) + "\n"
                        for s in chunk).encode("utf-8")
        f = ok(session.post(f"{base}/files", data={"purpose": "batch"},
                            files={"file": ("requests.jsonl", jsonl)})).json()
        batch = ok(session.post(f"{base}/batches", json={
            "input_file_id": f["id"], "endpoint": "/v1/chat/completions",
            "completion_window": "24h"})).json()
        state.add(batch["id"], [s["req_id"] for s in chunk])
        print(f"submitted batch {batch['id']} with {len(chunk)} requests")

    spec_by_id = {s["req_id"]: s for s in specs}
    for b in state.pending():
        while True:
            try:
                batch = ok(session.get(f"{base}/batches/{b['id']}")).json()
            except IOError as e:  # requests errors subclass IOError; keep polling
                print(f"{b['id']}: status check failed ({e}), retrying", file=sys.stderr)
                time.sleep(60)
                continue
            if batch["status"] in ("completed", "failed", "expired", "cancelled"):
                break
            c = batch.get("request_counts") or {}
            print(f"{b['id']}: {batch['status']} completed={c.get('completed')} "
                  f"failed={c.get('failed')} total={c.get('total')}")
            time.sleep(30)
        if batch["status"] == "failed":
            print(f"batch {b['id']} failed validation: {batch.get('errors')}", file=sys.stderr)
        if batch.get("error_file_id"):
            errors = ok(session.get(f"{base}/files/{batch['error_file_id']}/content")).text
            stats["failed"] += len(errors.splitlines())
            print(f"batch {b['id']}: {len(errors.splitlines())} failed requests, first: "
                  f"{errors.splitlines()[0][:300]}", file=sys.stderr)
        if batch.get("output_file_id"):
            output = ok(session.get(f"{base}/files/{batch['output_file_id']}/content")).text
            for line in output.splitlines():
                res = json.loads(line)
                rid = int(res["custom_id"][1:])
                resp = res.get("response") or {}
                if resp.get("status_code") != 200 or rid not in spec_by_id:
                    stats["failed"] += 1
                    continue
                handle_openai_response(spec_by_id[rid], resp["body"], args, sink, stats)
        state.mark_collected(b)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["anthropic-batch", "anthropic", "openai-batch", "openai"], default="anthropic")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--n_requests", type=int, default=20, help="total requests (req_ids 0..n-1)")
    p.add_argument("--stories_per_request", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max_tokens", type=int, default=8000)
    p.add_argument("--workers", type=int, default=8, help="parallel requests (direct backends)")
    p.add_argument("--batch_size", type=int, default=10000, help="requests per batch")
    p.add_argument("--json", action="store_true",
                   help="openai backends: structured JSON output instead of === separators")
    p.add_argument("--base_url", default="http://127.0.0.1:8080/v1", help="openai backend")
    p.add_argument("--extra_body", default="", help="openai backend: JSON merged into the request")
    p.add_argument("--price_in", type=float, help="$ per 1M input tokens, standard rate (openai backends)")
    p.add_argument("--price_out", type=float, help="$ per 1M output tokens, standard rate (openai backends)")
    p.add_argument("--prompt_data", default=DEFAULT_PROMPT_DATA,
                   help="JSON with the system prompt, genres, audiences, features and word lists")
    p.add_argument("--out", default=os.path.join(HERE, "stories", "stories.jsonl"))
    p.add_argument("--dry_run", action="store_true", help="print one prompt and exit")
    args = p.parse_args()

    try:
        data = load_prompt_data(args.prompt_data)
    except (OSError, ValueError) as e:  # JSONDecodeError is a ValueError
        sys.exit(f"bad prompt data: {e}")
    print(f"prompt data {args.prompt_data} ({data.fingerprint}): {len(data.nouns)} nouns, "
          f"{len(data.verbs)} verbs, {len(data.adjectives)} adjectives")

    if args.dry_run:
        print(make_request(data, 0, args.seed, args.stories_per_request, args.json)["prompt"])
        return

    runners = {"anthropic": run_anthropic, "anthropic-batch": run_anthropic_batch,
               "openai": run_openai, "openai-batch": run_openai_batch}
    run = runners[args.backend]
    if args.backend.endswith("-batch"):
        try:
            state = BatchState(args.out + ".batches.json", data.fingerprint)
        except ValueError as e:
            sys.exit(str(e))
        run = functools.partial(run, state=state)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["req_id"] for line in f}
    specs = [make_request(data, i, args.seed, args.stories_per_request, args.json)
             for i in range(args.n_requests) if i not in done]
    print(f"{len(done)} requests already done, {len(specs)} to go")

    stats = {"in": 0, "out": 0, "reasoning": 0, "failed": 0, "stories": 0, "chars": 0}
    out_f = open(args.out, "a", encoding="utf-8")

    def sink(stories):
        for s in stories:
            out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
            stats["stories"] += 1
            stats["chars"] += len(s["text"])
        out_f.flush()

    t0 = time.time()
    try:
        run(specs, args, sink, stats)
    finally:
        out_f.close()
        report(args, stats, time.time() - t0)


def report(args, stats, seconds):
    print(f"\nstories written: {stats['stories']}   failed requests: {stats['failed']}   "
          f"time: {seconds / 60:.1f} min")
    if not stats["stories"]:
        return
    print(f"tokens: in={stats['in']:,} out={stats['out']:,} (reasoning {stats['reasoning']:,})   "
          f"avg chars/story={stats['chars'] / stats['stories']:.0f}   "
          f"generator chars/output token={stats['chars'] / max(stats['out'], 1):.2f}")
    if args.model in CLAUDE_PRICES:
        pin, pout = CLAUDE_PRICES[args.model]
    elif args.price_in is not None and args.price_out is not None:
        pin, pout = args.price_in, args.price_out
    else:
        return
    if args.backend.endswith("-batch"):  # both providers: batch = 50% off
        pin, pout = pin / 2, pout / 2
    cost = (stats["in"] * pin + stats["out"] * pout) / 1e6
    # our 8k BPE tokenizer: assume ~4 chars per token on simple Ukrainian text
    our_tokens = stats["chars"] / 4
    print(f"cost: ${cost:.3f}   -> ~${cost / our_tokens * 100e6:,.0f} per 100M training tokens")


if __name__ == "__main__":
    main()

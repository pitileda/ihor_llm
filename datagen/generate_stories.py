"""
Generate a synthetic Ukrainian "TinyStories"-style dataset with an LLM.

Variety comes from randomizing every request (TinyStories recipe): each story gets
3 random words (noun, verb, adjective) it must use, plus a random genre, audience
and 1-2 narrative features. Several stories are generated per request to save cost.

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
  {"req_id": 17, "model": ..., "genre": ..., "audience": ..., "words": [...], "features": [...], "text": ...}
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Vocabulary for prompt randomization. ~130 x 75 x 75 = ~700k word triples.
# Verbs are infinitives, the model inflects all words as needed.

NOUNS = """
кошеня цуценя їжачок зайчик лисиця ведмедик білка сова жабка метелик бджола мурашка
равлик качка курча корова кінь коза мишка пташка рибка черепаха слон жирафа мавпа
дракон бабуся дідусь мама тато сестричка братик друг подруга вчителька лікар пекар
садівник рибалка пожежник м'яч лялька машинка кубики кулька книжка олівець малюнок
ліхтарик парасолька капелюх шарф рукавички черевики ковдра подушка ліжко вікно двері
сходи дах сад город ліс поле річка озеро море гора печера міст дорога стежка село
місто парк гойдалка пісочниця школа магазин ринок кухня торт пиріжок яблуко груша
вишня суниця морква гарбуз хліб молоко мед каша суп сонце місяць зірка хмара дощ
сніг вітер веселка грім калюжа листок квітка соняшник дерево гриб жолудь камінчик
мушля скриня ключ карта човен потяг літак велосипед санчата ракета годинник дзвіночок
свічка подарунок лист пісня гра свято сон таємниця
""".split()

VERBS = """
бігати стрибати літати плавати співати танцювати малювати читати писати рахувати
будувати копати садити поливати готувати пекти їсти пити спати прокидатися шукати
знайти загубити ховатися грати допомагати ділитися дарувати обіймати сміятися плакати
боятися мріяти чекати слухати дивитися рости лагодити прибирати мити одягатися гуляти
подорожувати лізти падати ловити кидати нести тягнути штовхати відкривати закривати
рятувати захищати вибачатися дякувати пробачити навчитися розповідати питати
відповідати кликати світити сяяти шепотіти кричати тремтіти ковзати кататися будити
збирати
""".split()

ADJECTIVES = """
маленький великий веселий сумний сміливий боязкий добрий хитрий лінивий працьовитий
швидкий повільний теплий холодний мокрий сухий м'який твердий блискучий яскравий
темний світлий червоний жовтий зелений синій білий чорний рудий золотий срібний
пухнастий колючий солодкий кислий смачний гарячий тихий гучний дивний чарівний старий
новий високий низький довгий короткий круглий важкий легкий чистий брудний голодний
сонний щасливий сердитий цікавий загублений зламаний улюблений таємничий крихітний
величезний ввічливий слухняний неслухняний розумний кмітливий дружній самотній щедрий
жадібний уважний гордий ніжний смішний
""".split()

GENRES = [
    "казка",
    "оповідання з повсякденного життя дитини",
    "байка про тварин",
    "коротка розповідь від першої особи",
    "лист другові або бабусі",
    "опис природи, пори року або місця",
    "сценка-діалог між персонажами",
    "пояснення для дитини, як або чому щось відбувається у світі",
]

AUDIENCES = ["дитина 3–5 років", "дитина 6–8 років"]

FEATURES = [
    "У тексті має бути діалог: персонажі розмовляють, пряма мова оформлена через тире.",
    "У сюжеті має бути несподіваний поворот.",
    "Текст має закінчуватися мораллю або уроком.",
    "Історія має сумний або гіркуватий кінець.",
    "У сюжеті має бути конфлікт, який персонажі розв'язують.",
    "Використай повтор, як у народних казках: щось трапляється тричі.",
]

SYSTEM = (
    "Ти — досвідчений український дитячий письменник. "
    "Пишеш грамотною літературною українською мовою, без русизмів і суржику."
)

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


def make_request(req_id: int, seed: int, stories_per_request: int, json_mode: bool = False) -> dict:
    """Deterministic random spec for one request: same req_id -> same prompt."""
    rng = random.Random(seed * 1_000_003 + req_id)
    genre = rng.choice(GENRES)
    audience = rng.choice(AUDIENCES)
    features = rng.sample(FEATURES, rng.choice([1, 1, 2]))
    stories = [[rng.choice(NOUNS), rng.choice(VERBS), rng.choice(ADJECTIVES)]
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
- {" ".join(features)}
- Кожен текст має природно містити свої три слова (у будь-якій граматичній формі):
{words_lines}

{FORMAT_JSON if json_mode else FORMAT_SEPARATOR}"""
    return {"req_id": req_id, "genre": genre, "audience": audience,
            "features": features, "stories": stories, "prompt": prompt}


# Ukrainian never uses these letters; their presence means Russian text slipped in.
RUSSIAN_LETTERS = re.compile(r"[ыэъёЫЭЪЁ]")


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
                    "words": words, "text": p})
    return out


# ---------------------------------------------------------------------------
# Claude

def claude_params(model: str, prompt: str, max_tokens: int) -> dict:
    """Short creative writing needs no deep reasoning: minimize thinking tokens."""
    params = {"model": model, "max_tokens": max_tokens, "system": SYSTEM,
              "messages": [{"role": "user", "content": prompt}],
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
        msg = client.messages.create(**claude_params(args.model, spec["prompt"], args.max_tokens))
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


def run_anthropic_batch(specs, args, sink, stats):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    state_path = args.out + ".batches.json"
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"batches": []}

    def save_state():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)

    # submit requests not waiting in an uncollected batch; requests that failed in an
    # already collected batch have no output lines, so a rerun resubmits them
    in_flight = {rid for b in state["batches"] if not b["collected"] for rid in b["req_ids"]}
    todo = [s for s in specs if s["req_id"] not in in_flight]
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=f"r{s['req_id']}",
                    params=MessageCreateParamsNonStreaming(
                        **claude_params(args.model, s["prompt"], args.max_tokens)))
            for s in chunk
        ])
        state["batches"].append({"id": batch.id, "req_ids": [s["req_id"] for s in chunk],
                                 "collected": False})
        save_state()
        print(f"submitted batch {batch.id} with {len(chunk)} requests")

    spec_by_id = {s["req_id"]: s for s in specs}
    for b in state["batches"]:
        if b["collected"]:
            continue
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
        b["collected"] = True
        save_state()


# ---------------------------------------------------------------------------
# OpenAI-compatible (llama.cpp server, OpenRouter, DeepInfra, ...)

def openai_body(spec, args, extra):
    body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 1.0,
            "messages": [{"role": "system", "content": SYSTEM},
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


def run_openai_batch(specs, args, sink, stats):
    """OpenAI Batch API: upload a JSONL file of requests, poll, download results."""
    session = openai_session()
    base = args.base_url.rstrip("/")
    extra = json.loads(args.extra_body) if args.extra_body else {}
    state_path = args.out + ".batches.json"
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"batches": []}

    def save_state():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)

    def ok(r):
        r.raise_for_status()
        return r

    in_flight = {rid for b in state["batches"] if not b["collected"] for rid in b["req_ids"]}
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
        state["batches"].append({"id": batch["id"], "req_ids": [s["req_id"] for s in chunk],
                                 "collected": False})
        save_state()
        print(f"submitted batch {batch['id']} with {len(chunk)} requests")

    spec_by_id = {s["req_id"]: s for s in specs}
    for b in state["batches"]:
        if b["collected"]:
            continue
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
        b["collected"] = True
        save_state()


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
    p.add_argument("--out", default=os.path.join(HERE, "stories", "stories.jsonl"))
    p.add_argument("--dry_run", action="store_true", help="print one prompt and exit")
    args = p.parse_args()

    if args.dry_run:
        print(make_request(0, args.seed, args.stories_per_request, args.json)["prompt"])
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["req_id"] for line in f}
    specs = [make_request(i, args.seed, args.stories_per_request, args.json)
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
        {"anthropic": run_anthropic, "anthropic-batch": run_anthropic_batch,
         "openai": run_openai, "openai-batch": run_openai_batch}[args.backend](specs, args, sink, stats)
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

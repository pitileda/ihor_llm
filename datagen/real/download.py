"""
Download the raw corpora listed in sources.json from the Hugging Face Hub.

Each source is pinned to the dataset's current commit, and goes to <out_dir>/<name>/
with the repo's own file layout. Downloads resume: rerunning skips finished files
and continues partial ones. <out_dir>/manifest.json records what was downloaded
(repo, commit, files, bytes), so a later prepare step knows exactly which data it used.

Usage:
    python download.py --out_dir /mnt/secure_ssd/llm/ua_data_set --list   # sizes only
    python download.py --out_dir /mnt/secure_ssd/llm/ua_data_set          # everything
    python download.py --out_dir ... --only wikipedia_uk wikisource_uk     # some sources
"""
import argparse
import fnmatch
import json
import os
import shutil
import sys
import time

from huggingface_hub import HfApi, snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCES = os.path.join(HERE, "sources.json")
REQUIRED_KEYS = {"repo", "include", "lang", "about"}


def load_sources(path: str) -> dict:
    """Load and validate sources.json; raises ValueError on a malformed file."""
    with open(path, encoding="utf-8") as f:
        sources = json.load(f)
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"{path}: expected a non-empty object of sources")
    for name, src in sources.items():
        missing = REQUIRED_KEYS - src.keys()
        if missing:
            raise ValueError(f"{path}: source '{name}' is missing {sorted(missing)}")
        if not isinstance(src["include"], list) or not src["include"]:
            raise ValueError(f"{path}: source '{name}': 'include' must be a non-empty list")
    return sources


def remote_files(api: HfApi, src: dict) -> tuple:
    """(commit sha, [(path, size)]) of the files a source would download."""
    info = api.dataset_info(src["repo"], files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings
             if any(fnmatch.fnmatch(s.rfilename, p) for p in src["include"])]
    return info.sha, files


def load_manifest(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(path: str, manifest: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="root folder for the raw corpora")
    p.add_argument("--sources", default=DEFAULT_SOURCES)
    p.add_argument("--only", nargs="+", help="source names to download (default: all)")
    p.add_argument("--list", action="store_true", help="print sources and sizes, download nothing")
    p.add_argument("--workers", type=int, default=8, help="parallel file downloads")
    args = p.parse_args()

    try:
        sources = load_sources(args.sources)
    except (OSError, ValueError) as e:
        sys.exit(f"bad sources file: {e}")
    unknown = set(args.only or []) - sources.keys()
    if unknown:
        sys.exit(f"unknown sources: {sorted(unknown)}; known: {sorted(sources)}")
    selected = {k: v for k, v in sources.items() if not args.only or k in args.only}

    api = HfApi()
    plan = {}
    for name, src in selected.items():
        sha, files = remote_files(api, src)
        if not files:
            sys.exit(f"{name}: no files in {src['repo']} match {src['include']}")
        plan[name] = (sha, files)
    total = sum(size for _, files in plan.values() for _, size in files)
    for name, (sha, files) in plan.items():
        size = sum(s for _, s in files)
        print(f"{name:18} {selected[name]['lang']:6} {len(files):4} files {size / 1e9:8.2f} GB  "
              f"{selected[name]['repo']}@{sha[:8]}")
    print(f"{'total':18} {'':6} {sum(len(f) for _, f in plan.values()):4} files {total / 1e9:8.2f} GB")
    if args.list:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    free = shutil.disk_usage(args.out_dir).free
    if free < total * 1.05:  # already downloaded files make this conservative
        print(f"warning: {free / 1e9:.0f} GB free in {args.out_dir}, sources need {total / 1e9:.0f} GB",
              file=sys.stderr)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = load_manifest(manifest_path)
    for name, (sha, files) in plan.items():
        src = selected[name]
        done = manifest.get(name, {})
        if done.get("revision") == sha and done.get("complete"):
            print(f"\n{name}: already complete, skipping")
            continue
        if done.get("revision") not in (None, sha):
            print(f"\n{name}: dataset changed upstream ({done['revision'][:8]} -> {sha[:8]}), "
                  f"updating", file=sys.stderr)
        print(f"\n{name}: downloading {len(files)} files, {sum(s for _, s in files) / 1e9:.2f} GB")
        t0 = time.time()
        snapshot_download(repo_id=src["repo"], repo_type="dataset", revision=sha,
                          allow_patterns=src["include"], local_dir=os.path.join(args.out_dir, name),
                          max_workers=args.workers)
        manifest[name] = {**src, "revision": sha, "files": [f for f, _ in files],
                          "bytes": sum(s for _, s in files), "complete": True,
                          "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        save_manifest(manifest_path, manifest)
        print(f"{name}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recursively scan asmrgay.com AList with rate-limit handling."""

import json
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import PurePosixPath

API = "http://asmrgay.com/api/fs/list"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus"}
LRC_EXT = ".lrc"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# Rate control
BASE_DELAY = 0.4       # seconds between requests
MAX_RETRIES = 5
BACKOFF_BASE = 3       # seconds for first retry on 429

stats = {
    "dirs_scanned": 0, "audio_files": 0, "lrc_files": 0, "other_files": 0,
    "audio_total_bytes": 0, "lrc_total_bytes": 0,
    "paired": 0, "lrc_orphan": 0, "requests": 0, "retries": 0, "errors": 0,
}
partition_stats = defaultdict(lambda: {"audio": 0, "lrc": 0, "paired": 0, "dirs": 0})
category_stats = defaultdict(lambda: {"audio": 0, "lrc": 0, "paired": 0})
sample_lrc_paths = []
MAX_SAMPLES = 50
lrc_depth_dist = defaultdict(int)
# Track works (3rd level dirs) that contain LRC
works_with_lrc = []


def list_dir(path, page=1, per_page=200):
    """Call AList API with retry on 429."""
    payload = json.dumps({
        "path": path, "password": "", "page": page,
        "per_page": per_page, "refresh": False,
    }).encode()

    for attempt in range(MAX_RETRIES + 1):
        stats["requests"] += 1
        req = urllib.request.Request(API, data=payload, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            if data["code"] != 200:
                return [], 0
            content = data["data"].get("content") or []
            total = data["data"].get("total", len(content))
            time.sleep(BASE_DELAY)
            return content, total
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** attempt)
                stats["retries"] += 1
                print(f"    [429] {path} — waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            stats["errors"] += 1
            return [], 0
        except Exception:
            stats["errors"] += 1
            return [], 0
    return [], 0


def scan_dir(path, depth=0, partition=None, category=None):
    """Recursively scan a directory."""
    page = 1
    all_items = []
    while True:
        items, total = list_dir(path, page=page)
        all_items.extend(items)
        if len(all_items) >= total or not items:
            break
        page += 1

    stats["dirs_scanned"] += 1
    if partition:
        partition_stats[partition]["dirs"] += 1

    files = [i for i in all_items if not i["is_dir"]]
    dirs = [i for i in all_items if i["is_dir"]]

    audio_stems = set()
    lrc_stems = set()

    for f in files:
        name = f["name"]
        ext = PurePosixPath(name).suffix.lower()
        stem = PurePosixPath(name).stem

        if ext in AUDIO_EXTS:
            audio_stems.add(stem)
            stats["audio_files"] += 1
            stats["audio_total_bytes"] += f.get("size", 0)
            if partition:
                partition_stats[partition]["audio"] += 1
            if category:
                category_stats[category]["audio"] += 1
        elif ext == LRC_EXT:
            lrc_stems.add(stem)
            stats["lrc_files"] += 1
            stats["lrc_total_bytes"] += f.get("size", 0)
            lrc_depth_dist[depth] += 1
            if partition:
                partition_stats[partition]["lrc"] += 1
            if category:
                category_stats[category]["lrc"] += 1
            if len(sample_lrc_paths) < MAX_SAMPLES:
                sample_lrc_paths.append(path + "/" + name)
        else:
            stats["other_files"] += 1

    paired = audio_stems & lrc_stems
    orphan_lrc = lrc_stems - audio_stems
    stats["paired"] += len(paired)
    stats["lrc_orphan"] += len(orphan_lrc)
    if partition:
        partition_stats[partition]["paired"] += len(paired)
    if category:
        category_stats[category]["paired"] += len(paired)

    if lrc_stems:
        works_with_lrc.append({
            "path": path,
            "audio": len(audio_stems),
            "lrc": len(lrc_stems),
            "paired": len(paired),
        })

    if stats["dirs_scanned"] % 100 == 0:
        print(f"  ... {stats['dirs_scanned']} dirs, {stats['audio_files']} audio, "
              f"{stats['lrc_files']} lrc, {stats['requests']} reqs, {stats['retries']} retries")

    for d in dirs:
        child_path = path + "/" + d["name"]
        cat = category if depth >= 1 else None
        if depth == 1:
            cat = d["name"]
        scan_dir(child_path, depth + 1, partition=partition, category=cat)


def fmt_bytes(b):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def main():
    t0 = time.time()

    # Get root structure
    root_items, _ = list_dir("/")
    partitions = [i["name"] for i in root_items if i["is_dir"] and i["name"].startswith("asmr")]
    print(f"Partitions: {partitions}\n")

    for part in partitions:
        print(f"=== Scanning /{part} ===")
        scan_dir(f"/{part}", depth=0, partition=part)
        print(f"  done. dirs={partition_stats[part]['dirs']} "
              f"audio={partition_stats[part]['audio']} lrc={partition_stats[part]['lrc']}\n")

    elapsed = time.time() - t0

    # Report
    print("=" * 70)
    print("SCAN RESULTS")
    print("=" * 70)
    print(f"Time elapsed         : {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"API requests         : {stats['requests']}  (retries: {stats['retries']})")
    print(f"Directories scanned  : {stats['dirs_scanned']}")
    print(f"Audio files          : {stats['audio_files']}  ({fmt_bytes(stats['audio_total_bytes'])})")
    print(f"LRC files            : {stats['lrc_files']}  ({fmt_bytes(stats['lrc_total_bytes'])})")
    print(f"Other files          : {stats['other_files']}")
    print(f"Audio+LRC paired     : {stats['paired']}")
    print(f"LRC orphans          : {stats['lrc_orphan']}")
    if stats['audio_files'] > 0:
        print(f"Audio→LRC match rate : {stats['paired']/stats['audio_files']*100:.1f}%")
    print(f"API errors (final)   : {stats['errors']}")

    print("\n--- Per Partition ---")
    for part in sorted(partition_stats):
        p = partition_stats[part]
        rate = f"{p['paired']/p['audio']*100:.1f}%" if p['audio'] > 0 else "N/A"
        print(f"  /{part:10s}  dirs={p['dirs']:>5d}  audio={p['audio']:>6d}  "
              f"lrc={p['lrc']:>6d}  paired={p['paired']:>6d}  match={rate}")

    print("\n--- Directories containing LRC files ---")
    for w in sorted(works_with_lrc, key=lambda x: x["lrc"], reverse=True):
        print(f"  lrc={w['lrc']:>4d}  audio={w['audio']:>4d}  paired={w['paired']:>4d}  {w['path']}")

    print("\n--- Top Categories by LRC count ---")
    top_cats = sorted(category_stats.items(), key=lambda x: x[1]["lrc"], reverse=True)[:30]
    for cat, c in top_cats:
        if c["lrc"] == 0 and c["audio"] < 100:
            continue
        rate = f"{c['paired']/c['audio']*100:.1f}%" if c['audio'] > 0 else "N/A"
        print(f"  {cat:40s}  audio={c['audio']:>5d}  lrc={c['lrc']:>5d}  paired={c['paired']:>5d}  match={rate}")

    print(f"\n--- LRC depth distribution ---")
    for depth in sorted(lrc_depth_dist):
        print(f"  depth {depth}: {lrc_depth_dist[depth]} files")

    # Save results
    results = {
        "stats": stats,
        "partition_stats": dict(partition_stats),
        "works_with_lrc": works_with_lrc,
        "sample_lrc_paths": sample_lrc_paths,
        "category_stats": {k: v for k, v in category_stats.items() if v["lrc"] > 0},
    }
    with open("scan_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to scan_results.json")
    print(f"Sample LRC paths: {len(sample_lrc_paths)}")


if __name__ == "__main__":
    main()

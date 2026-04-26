#!/usr/bin/env python3
"""
Focused scan: enumerate directories breadth-first with aggressive rate-limit
handling, collect LRC stats, then download+analyze samples.

Strategy:
  1. Get top-2 levels of each partition (partition → category → work list)
  2. For categories that look promising (汉化, DLsite, etc.), go deeper
  3. Track every .lrc we find; download a diverse sample set
"""

import json, time, urllib.request, urllib.error, re, os, sys
from collections import defaultdict
from pathlib import PurePosixPath

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

API_LIST = "http://asmrgay.com/api/fs/list"
API_GET  = "http://asmrgay.com/api/fs/get"
HDR = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
AUDIO_EXTS = {".mp3",".wav",".flac",".m4a",".ogg",".aac",".wma",".opus"}
DELAY = 0.5
BACKOFF = 4
MAX_RETRY = 5

# ---- API helpers ----

request_count = 0

def api(url, body):
    global request_count
    payload = json.dumps(body).encode()
    for attempt in range(MAX_RETRY + 1):
        request_count += 1
        req = urllib.request.Request(url, data=payload, headers=HDR)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read())
            time.sleep(DELAY)
            return d
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRY:
                w = BACKOFF * (2 ** attempt)
                print(f"  [429] retry in {w}s (attempt {attempt+1})...", flush=True)
                time.sleep(w)
                continue
            return None
        except Exception:
            return None
    return None

def ls(path, per_page=300):
    """List all items in path (handles pagination)."""
    page, items = 1, []
    while True:
        d = api(API_LIST, {"path": path, "page": page, "per_page": per_page})
        if not d or d.get("code") != 200 or not d["data"]:
            break
        batch = d["data"].get("content") or []
        items.extend(batch)
        if len(items) >= d["data"].get("total", 0) or not batch:
            break
        page += 1
    return items

def get_raw_url(path):
    d = api(API_GET, {"path": path, "password": ""})
    if d and d.get("code") == 200:
        return d["data"].get("raw_url")
    return None

def download(url):
    url = url.replace("https://", "http://")
    req = urllib.request.Request(url, headers={"User-Agent": HDR["User-Agent"]})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
        for enc in ("utf-8-sig", "utf-8", "shift_jis", "gbk", "latin-1"):
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, ValueError):
                continue
        return raw.decode("latin-1")
    except Exception as e:
        print(f"  [DL ERR] {e}")
        return None

# ---- Stats ----

class Stats:
    def __init__(self):
        self.dirs = 0
        self.audio = 0
        self.lrc = 0
        self.other = 0
        self.audio_bytes = 0
        self.lrc_bytes = 0
        self.paired = 0
        self.lrc_orphan = 0
        # per-partition
        self.part = defaultdict(lambda: {"audio":0,"lrc":0,"paired":0,"dirs":0})
        # per-category (2nd level)
        self.cat = defaultdict(lambda: {"audio":0,"lrc":0,"paired":0})
        # works with lrc
        self.works_lrc = []
        # all lrc paths
        self.lrc_paths = []

S = Stats()

def scan_files(path, items, partition, category):
    """Process files in a directory listing."""
    S.dirs += 1
    S.part[partition]["dirs"] += 1

    audio_stems, lrc_stems = set(), set()
    for f in items:
        if f["is_dir"]:
            continue
        name = f["name"]
        ext = PurePosixPath(name).suffix.lower()
        stem = PurePosixPath(name).stem
        size = f.get("size", 0)

        if ext in AUDIO_EXTS:
            audio_stems.add(stem)
            S.audio += 1; S.audio_bytes += size
            S.part[partition]["audio"] += 1
            S.cat[category]["audio"] += 1
        elif ext == ".lrc":
            lrc_stems.add(stem)
            S.lrc += 1; S.lrc_bytes += size
            S.part[partition]["lrc"] += 1
            S.cat[category]["lrc"] += 1
            S.lrc_paths.append(path + "/" + name)
        else:
            S.other += 1

    p = audio_stems & lrc_stems
    o = lrc_stems - audio_stems
    S.paired += len(p); S.lrc_orphan += len(o)
    S.part[partition]["paired"] += len(p)
    S.cat[category]["paired"] += len(p)
    if lrc_stems:
        S.works_lrc.append({"path": path, "audio": len(audio_stems),
                            "lrc": len(lrc_stems), "paired": len(p)})

# ---- Main scan logic ----

def scan_recursive(path, depth, max_depth, partition, category):
    """Generic recursive scan up to max_depth."""
    items = ls(path)
    if not items:
        return
    scan_files(path, items, partition, category)
    if depth >= max_depth:
        return
    dirs = [i for i in items if i["is_dir"]]
    for d in dirs:
        child = path + "/" + d["name"]
        cat = d["name"] if depth == 1 else category
        scan_recursive(child, depth+1, max_depth, partition, cat)


def main():
    t0 = time.time()

    # ---- Phase 1: Structure discovery ----
    print("=" * 60, flush=True)
    print("PHASE 1: Directory structure discovery", flush=True)
    print("=" * 60, flush=True)

    root = ls("/")
    parts = [i["name"] for i in root if i["is_dir"] and i["name"].startswith("asmr")]
    print(f"Partitions: {parts}\n", flush=True)

    # For each partition, get category list (level 1)
    part_cats = {}
    for p in parts:
        cats = ls(f"/{p}")
        cat_names = [c["name"] for c in cats if c["is_dir"]]
        file_count = sum(1 for c in cats if not c["is_dir"])
        part_cats[p] = cat_names
        print(f"  /{p}: {len(cat_names)} categories, {file_count} files", flush=True)

    # ---- Phase 2: Deep scan of promising areas ----
    print(f"\n{'='*60}", flush=True)
    print("PHASE 2: Deep scan (LRC-rich directories)", flush=True)
    print("=" * 60, flush=True)

    # Priority targets: 汉化 groups, DLsite, Japanese creators
    DEEP_SCAN_KEYWORDS = ["汉化", "チロル", "天知遥", "柚木", "野上", "清软",
                          "DLsite", "dlsite", "同人"]

    for p in parts:
        for cat in part_cats.get(p, []):
            # Decide scan depth
            is_priority = any(kw in cat for kw in DEEP_SCAN_KEYWORDS)
            # asmr5 is DLsite originals - always deep scan
            if p == "asmr5":
                is_priority = True

            if is_priority:
                max_d = 4  # deep: partition/category/work/track
                print(f"\n  [DEEP] /{p}/{cat}", flush=True)
            else:
                max_d = 3  # shallow: just peek for LRC
                # Don't print shallow scans to reduce noise

            scan_recursive(f"/{p}/{cat}", depth=2, max_depth=max_d,
                          partition=p, category=cat)

        print(f"\n  --- /{p} done: audio={S.part[p]['audio']} lrc={S.part[p]['lrc']} "
              f"paired={S.part[p]['paired']} ---", flush=True)

    elapsed = time.time() - t0

    # ---- Phase 3: Report ----
    print(f"\n{'='*60}", flush=True)
    print("SCAN RESULTS", flush=True)
    print("=" * 60, flush=True)

    def fmt(b):
        for u in ["B","KB","MB","GB","TB"]:
            if b < 1024: return f"{b:.1f} {u}"
            b /= 1024

    print(f"Time       : {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"API reqs   : {request_count}")
    print(f"Dirs       : {S.dirs}")
    print(f"Audio      : {S.audio}  ({fmt(S.audio_bytes)})")
    print(f"LRC        : {S.lrc}  ({fmt(S.lrc_bytes)})")
    print(f"Other      : {S.other}")
    print(f"Paired     : {S.paired}")
    print(f"LRC orphan : {S.lrc_orphan}")
    if S.audio > 0:
        print(f"Match rate : {S.paired/S.audio*100:.2f}%")

    print(f"\n--- Per Partition ---")
    for p in parts:
        d = S.part[p]
        r = f"{d['paired']/d['audio']*100:.1f}%" if d['audio'] > 0 else "N/A"
        print(f"  /{p:10s}  dirs={d['dirs']:>5}  audio={d['audio']:>6}  "
              f"lrc={d['lrc']:>6}  paired={d['paired']:>6}  match={r}")

    print(f"\n--- Categories with LRC (sorted by count) ---")
    cats_with_lrc = [(c, v) for c, v in S.cat.items() if v["lrc"] > 0]
    for cat, v in sorted(cats_with_lrc, key=lambda x: -x[1]["lrc"]):
        r = f"{v['paired']/v['audio']*100:.1f}%" if v['audio'] > 0 else "N/A"
        print(f"  {cat:45s} audio={v['audio']:>5}  lrc={v['lrc']:>5}  "
              f"paired={v['paired']:>5}  match={r}")

    print(f"\n--- Works with LRC (top 30) ---")
    for w in sorted(S.works_lrc, key=lambda x: -x["lrc"])[:30]:
        print(f"  lrc={w['lrc']:>3} audio={w['audio']:>3} paired={w['paired']:>3}  {w['path']}")

    # Save all data
    results = {
        "stats": {"dirs": S.dirs, "audio": S.audio, "lrc": S.lrc,
                  "audio_bytes": S.audio_bytes, "lrc_bytes": S.lrc_bytes,
                  "paired": S.paired, "lrc_orphan": S.lrc_orphan,
                  "requests": request_count, "elapsed_s": elapsed},
        "partition_stats": dict(S.part),
        "categories_with_lrc": cats_with_lrc,
        "works_with_lrc": S.works_lrc,
        "all_lrc_paths": S.lrc_paths,
    }
    with open("scan_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # ---- Phase 4: LRC quality sampling ----
    if S.lrc_paths:
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 3: LRC Quality Sampling ({min(30, len(S.lrc_paths))} samples)", flush=True)
        print("=" * 60, flush=True)

        os.makedirs("lrc_samples", exist_ok=True)

        # Evenly sample
        paths = S.lrc_paths
        n = min(30, len(paths))
        step = max(1, len(paths) // n)
        samples = paths[::step][:n]

        analyses = []
        for i, lpath in enumerate(samples):
            fname = lpath.split("/")[-1]
            print(f"  [{i+1}/{n}] {fname}", flush=True)
            raw_url = get_raw_url(lpath)
            if not raw_url:
                continue
            text = download(raw_url)
            if not text:
                continue

            # Save locally
            safe = f"sample_{i:03d}.lrc"
            with open(f"lrc_samples/{safe}", "w") as f:
                f.write(text)

            # Analyze
            a = analyze_lrc(text, lpath)
            analyses.append(a)

        # Quality report
        if analyses:
            print_quality_report(analyses)
            with open("lrc_analysis.json", "w") as f:
                json.dump(analyses, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDone. Total time: {time.time()-t0:.0f}s", flush=True)


def analyze_lrc(text, path=""):
    lines = text.strip().split("\n")
    r = {"path": path, "total_lines": len(lines), "timed_lines": 0,
         "empty_timed": 0, "has_offset": False, "ts_precision": "unknown",
         "langs": set(), "sfx": [], "gaps": []}

    ts_re = re.compile(r'\[(\d{2}):(\d{2})\.(\d{1,3})\]')
    timestamps = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'\[(ti|ar|al|by|offset|re|ve):', line):
            if "offset:" in line:
                r["has_offset"] = True
            continue

        m = ts_re.match(line)
        if m:
            r["timed_lines"] += 1
            mm, ss, frac = int(m.group(1)), int(m.group(2)), m.group(3)
            if len(frac) == 3:
                r["ts_precision"] = "ms"
            elif r["ts_precision"] != "ms":
                r["ts_precision"] = "cs"
            ts = mm*60 + ss + int(frac.ljust(3,'0')[:3])/1000
            timestamps.append(ts)

            txt = ts_re.sub("", line).strip()
            if not txt:
                r["empty_timed"] += 1
                continue

            # Language detect
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', txt):
                r["langs"].add("ja")
            if re.search(r'[\u4e00-\u9fff]', txt):
                r["langs"].add("zh")
            if re.search(r'[a-zA-Z]{3,}', txt):
                r["langs"].add("en")

            # Sound effects
            sfx = re.findall(r'\*[^*]+\*', txt)
            r["sfx"].extend(sfx)

    if len(timestamps) >= 2:
        r["gaps"] = [timestamps[i+1]-timestamps[i]
                     for i in range(len(timestamps)-1) if timestamps[i+1]>timestamps[i]]

    r["langs"] = sorted(r["langs"])
    return r


def print_quality_report(analyses):
    print(f"\n  Samples analyzed: {len(analyses)}")

    # Languages
    lc = defaultdict(int)
    for a in analyses:
        for l in a["langs"]:
            lc[l] += 1
    n = len(analyses)
    print(f"\n  --- Languages ---")
    for l in ["zh", "ja", "en"]:
        label = {"zh":"Chinese","ja":"Japanese","en":"English"}[l]
        print(f"    {label}: {lc[l]}/{n}")
    zh_only = sum(1 for a in analyses if "zh" in a["langs"] and "ja" not in a["langs"])
    ja_only = sum(1 for a in analyses if "ja" in a["langs"] and "zh" not in a["langs"])
    both = sum(1 for a in analyses if "zh" in a["langs"] and "ja" in a["langs"])
    neither = sum(1 for a in analyses if not a["langs"])
    print(f"    Chinese-only: {zh_only}  Japanese-only: {ja_only}  Both: {both}  Neither: {neither}")

    # Precision
    pc = defaultdict(int)
    for a in analyses:
        pc[a["ts_precision"]] += 1
    print(f"\n  --- Timestamp Precision ---")
    for p, c in sorted(pc.items(), key=lambda x:-x[1]):
        print(f"    {p}: {c}")

    # Gaps
    all_gaps = [g for a in analyses for g in a["gaps"]]
    if all_gaps:
        avg_gaps = [sum(a["gaps"])/len(a["gaps"]) for a in analyses if a["gaps"]]
        print(f"\n  --- Timing ---")
        print(f"    Mean line gap: {sum(all_gaps)/len(all_gaps):.1f}s")
        print(f"    Per-file avg gap range: {min(avg_gaps):.1f}s - {max(avg_gaps):.1f}s")

    # Lines
    tl = [a["timed_lines"] for a in analyses]
    et = [a["empty_timed"] for a in analyses]
    print(f"\n  --- Lines ---")
    print(f"    Timed lines/file: mean={sum(tl)/n:.0f}  min={min(tl)}  max={max(tl)}")
    print(f"    Empty timed/file: mean={sum(et)/n:.0f}")
    print(f"    Total timed lines: {sum(tl)}")

    # Sound effects
    all_sfx = [s for a in analyses for s in a["sfx"]]
    sfx_files = sum(1 for a in analyses if a["sfx"])
    print(f"\n  --- Sound Effects ---")
    print(f"    Files with SFX annotations: {sfx_files}/{n}")
    if all_sfx:
        sc = defaultdict(int)
        for s in all_sfx:
            sc[s] += 1
        print(f"    Top SFX:")
        for s, c in sorted(sc.items(), key=lambda x:-x[1])[:10]:
            print(f"      {s}: {c}")

    # Offset
    wo = sum(1 for a in analyses if a["has_offset"])
    print(f"\n  --- Metadata ---")
    print(f"    With offset tag: {wo}/{n}")


if __name__ == "__main__":
    main()

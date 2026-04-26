#!/usr/bin/env python3
"""
Smart scan: fast breadth-first enumeration.

Strategy:
  Phase A: Get work-level counts for ALL partitions (only go 2 levels deep)
  Phase B: Deep-scan ONLY 汉化-related categories for LRC pairing
  Phase C: Download + analyze LRC samples
"""

import json, time, urllib.request, urllib.error, re, os, sys
from collections import defaultdict
from pathlib import PurePosixPath

sys.stdout.reconfigure(line_buffering=True)

API = "http://asmrgay.com/api/fs/list"
API_GET = "http://asmrgay.com/api/fs/get"
HDR = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
AUDIO_EXTS = {".mp3",".wav",".flac",".m4a",".ogg",".aac",".wma",".opus"}
DELAY = 0.35
reqs = 0

def api(url, body, retries=5):
    global reqs
    payload = json.dumps(body).encode()
    for attempt in range(retries + 1):
        reqs += 1
        req = urllib.request.Request(url, data=payload, headers=HDR)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read())
            time.sleep(DELAY)
            return d
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                w = 4 * (2 ** attempt)
                print(f"    [429→{w}s]", end="", flush=True)
                time.sleep(w)
                continue
            return None
        except Exception:
            return None
    return None

def ls(path, per_page=300):
    page, items = 1, []
    while True:
        d = api(API, {"path": path, "page": page, "per_page": per_page})
        if not d or d.get("code") != 200 or not d.get("data"):
            break
        batch = d["data"].get("content") or []
        items.extend(batch)
        if len(items) >= d["data"].get("total", 0) or not batch:
            break
        page += 1
    return items

def classify_items(items):
    """Return (dirs, audio_files, lrc_files, other_files) from a listing."""
    dirs, audios, lrcs, others = [], [], [], []
    for i in items:
        if i["is_dir"]:
            dirs.append(i)
        else:
            ext = PurePosixPath(i["name"]).suffix.lower()
            if ext in AUDIO_EXTS:
                audios.append(i)
            elif ext == ".lrc":
                lrcs.append(i)
            else:
                others.append(i)
    return dirs, audios, lrcs, others

# ================================================================
# Phase A: Breadth-first overview
# ================================================================

def phase_a():
    """Get category-level overview for all partitions."""
    print("=" * 60)
    print("PHASE A: Partition & category overview")
    print("=" * 60)

    root = ls("/")
    parts = sorted([i["name"] for i in root if i["is_dir"] and i["name"].startswith("asmr")])
    print(f"Partitions: {parts}\n")

    overview = {}
    for p in parts:
        items = ls(f"/{p}")
        dirs, audios, lrcs, others = classify_items(items)
        cat_info = []
        for d in dirs:
            # Just list the category, count works inside
            sub = ls(f"/{p}/{d['name']}")
            sub_dirs, sub_a, sub_l, sub_o = classify_items(sub)
            cat_info.append({
                "name": d["name"],
                "works": len(sub_dirs),
                "top_audio": len(sub_a),
                "top_lrc": len(sub_l),
                "top_other": len(sub_o),
            })

        total_works = sum(c["works"] for c in cat_info)
        total_top_a = sum(c["top_audio"] for c in cat_info) + len(audios)
        total_top_l = sum(c["top_lrc"] for c in cat_info) + len(lrcs)

        overview[p] = {
            "categories": len(dirs),
            "total_works_l2": total_works,
            "top_audio": total_top_a,
            "top_lrc": total_top_l,
            "top_files": len(audios) + len(lrcs) + len(others),
            "cat_details": cat_info,
        }

        print(f"/{p}:", flush=True)
        print(f"  categories: {len(dirs)}  works(L2): {total_works}  "
              f"top_audio: {total_top_a}  top_lrc: {total_top_l}")
        for c in sorted(cat_info, key=lambda x: -x["works"])[:10]:
            lrc_tag = f"  ★LRC={c['top_lrc']}" if c['top_lrc'] > 0 else ""
            print(f"    {c['name']:35s}  works={c['works']:>5}"
                  f"  audio={c['top_audio']:>4}  lrc={c['top_lrc']:>4}{lrc_tag}")
        if len(cat_info) > 10:
            print(f"    ... and {len(cat_info)-10} more categories")
        print()

    return overview, parts

# ================================================================
# Phase B: Deep scan 汉化 directories for LRC
# ================================================================

def phase_b(overview, parts):
    """Deep scan categories likely to have LRC."""
    print("=" * 60)
    print("PHASE B: Deep scan LRC-bearing categories")
    print("=" * 60)

    # Identify targets: categories with top_lrc > 0, or known 汉化 groups
    HANHUA_KW = ["汉化", "翻译", "字幕"]
    targets = []

    for p in parts:
        for c in overview.get(p, {}).get("cat_details", []):
            is_hanhua = any(kw in c["name"] for kw in HANHUA_KW)
            has_lrc = c["top_lrc"] > 0
            if is_hanhua or has_lrc:
                targets.append((p, c["name"], c["works"]))

    print(f"Targets for deep scan: {len(targets)}")
    for p, c, w in targets:
        print(f"  /{p}/{c}  ({w} works)")

    all_lrc_paths = []
    all_paired = 0
    all_audio_deep = 0
    work_details = []

    for p, cat, num_works in targets:
        print(f"\n  Scanning /{p}/{cat} ({num_works} works)...", flush=True)
        cat_items = ls(f"/{p}/{cat}")
        cat_dirs, cat_a, cat_l, _ = classify_items(cat_items)

        # LRC at category level
        for l in cat_l:
            all_lrc_paths.append(f"/{p}/{cat}/{l['name']}")

        scanned = 0
        for work_dir in cat_dirs:
            work_path = f"/{p}/{cat}/{work_dir['name']}"
            work_items = ls(work_path)
            w_dirs, w_a, w_l, _ = classify_items(work_items)

            a_stems = {PurePosixPath(f["name"]).stem for f in w_a}
            l_stems = {PurePosixPath(f["name"]).stem for f in w_l}
            paired = len(a_stems & l_stems)

            all_audio_deep += len(w_a)
            all_paired += paired
            for l in w_l:
                all_lrc_paths.append(work_path + "/" + l["name"])

            if w_l:
                work_details.append({
                    "path": work_path,
                    "audio": len(w_a), "lrc": len(w_l), "paired": paired
                })

            # Go one more level if there are subdirs (e.g. "本編BGM＆SEなし/")
            for sd in w_dirs:
                sd_path = work_path + "/" + sd["name"]
                sd_items = ls(sd_path)
                _, sd_a, sd_l, _ = classify_items(sd_items)

                sa_stems = {PurePosixPath(f["name"]).stem for f in sd_a}
                sl_stems = {PurePosixPath(f["name"]).stem for f in sd_l}
                sp = len(sa_stems & sl_stems)

                all_audio_deep += len(sd_a)
                all_paired += sp
                for l in sd_l:
                    all_lrc_paths.append(sd_path + "/" + l["name"])

                if sd_l:
                    work_details.append({
                        "path": sd_path,
                        "audio": len(sd_a), "lrc": len(sd_l), "paired": sp
                    })

            scanned += 1
            if scanned % 20 == 0:
                print(f"    {scanned}/{num_works} works, {len(all_lrc_paths)} LRC found", flush=True)

        print(f"    done: {scanned} works scanned, {len(all_lrc_paths)} total LRC", flush=True)

    print(f"\n  DEEP SCAN SUMMARY:")
    print(f"    Total LRC files found : {len(all_lrc_paths)}")
    print(f"    Audio in LRC dirs     : {all_audio_deep}")
    print(f"    Paired (audio+LRC)    : {all_paired}")
    print(f"    Works with LRC        : {len(work_details)}")

    if work_details:
        print(f"\n    Top works by LRC count:")
        for w in sorted(work_details, key=lambda x: -x["lrc"])[:20]:
            print(f"      lrc={w['lrc']:>3} audio={w['audio']:>3} "
                  f"paired={w['paired']:>3}  {w['path']}")

    return all_lrc_paths, work_details

# ================================================================
# Phase C: LRC quality analysis
# ================================================================

def phase_c(lrc_paths):
    """Download and analyze sample LRC files."""
    print(f"\n{'='*60}")
    print(f"PHASE C: LRC Quality Analysis")
    print("=" * 60)

    if not lrc_paths:
        print("No LRC files found!")
        return

    os.makedirs("lrc_samples", exist_ok=True)

    n = min(30, len(lrc_paths))
    step = max(1, len(lrc_paths) // n)
    samples = lrc_paths[::step][:n]

    print(f"Sampling {n} of {len(lrc_paths)} LRC files\n")

    analyses = []
    for i, lpath in enumerate(samples):
        fname = lpath.split("/")[-1]
        print(f"  [{i+1}/{n}] {fname[:60]}", flush=True)
        d = api(API_GET, {"path": lpath, "password": ""})
        if not d or d.get("code") != 200:
            continue
        raw_url = d["data"].get("raw_url", "").replace("https://", "http://")
        if not raw_url:
            continue
        try:
            req = urllib.request.Request(raw_url, headers={"User-Agent": HDR["User-Agent"]})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            time.sleep(DELAY)
        except Exception as e:
            print(f"    [DL ERR] {e}")
            continue

        for enc in ("utf-8-sig","utf-8","shift_jis","gbk","latin-1"):
            try: text = raw.decode(enc); break
            except: continue
        else:
            text = raw.decode("latin-1")

        with open(f"lrc_samples/sample_{i:03d}.lrc", "w") as f:
            f.write(text)

        a = analyze(text, lpath)
        analyses.append(a)

    if not analyses:
        print("No samples downloaded!")
        return

    # ---- Report ----
    N = len(analyses)
    print(f"\n  Analyzed: {N} files\n")

    # Language
    lc = defaultdict(int)
    for a in analyses:
        for l in a["langs"]: lc[l] += 1
    print("  --- Language Distribution ---")
    for l in ["zh","ja","en"]:
        lb = {"zh":"Chinese","ja":"Japanese","en":"English"}[l]
        print(f"    {lb:10s}: {lc.get(l,0):>3}/{N} ({lc.get(l,0)/N*100:.0f}%)")

    zh_only = sum(1 for a in analyses if "zh" in a["langs"] and "ja" not in a["langs"])
    ja_only = sum(1 for a in analyses if "ja" in a["langs"] and "zh" not in a["langs"])
    both    = sum(1 for a in analyses if "zh" in a["langs"] and "ja" in a["langs"])
    print(f"    Chinese-only: {zh_only}   Japanese-only: {ja_only}   Both-CJK: {both}")

    # bilingual check: does any single line contain both kana AND hanzi?
    bi = sum(1 for a in analyses if a.get("bilingual_lines", 0) > 0)
    print(f"    Files with true bilingual lines (kana+hanzi same line): {bi}")

    # Precision
    pc = defaultdict(int)
    for a in analyses: pc[a["ts_prec"]] += 1
    print(f"\n  --- Timestamp Precision ---")
    for p, c in sorted(pc.items(), key=lambda x:-x[1]):
        print(f"    {p}: {c}")

    # Gaps
    all_avg = [a["avg_gap"] for a in analyses if a["avg_gap"]]
    if all_avg:
        print(f"\n  --- Timing (inter-line gap) ---")
        print(f"    Mean avg gap: {sum(all_avg)/len(all_avg):.1f}s")
        print(f"    Range: {min(all_avg):.1f}s — {max(all_avg):.1f}s")

    # Lines
    tl = [a["timed"] for a in analyses]
    el = [a["empty"] for a in analyses]
    dur = [a["duration"] for a in analyses if a["duration"]]
    print(f"\n  --- Content Volume ---")
    print(f"    Timed lines/file: mean={sum(tl)/N:.0f}  range={min(tl)}—{max(tl)}")
    print(f"    Empty lines/file: mean={sum(el)/N:.0f}")
    print(f"    Total timed lines: {sum(tl)}")
    if dur:
        print(f"    Estimated duration/file: mean={sum(dur)/len(dur)/60:.1f}min  "
              f"range={min(dur)/60:.1f}—{max(dur)/60:.1f}min")

    # SFX
    sfx_files = sum(1 for a in analyses if a["sfx_count"])
    all_sfx = []
    for a in analyses: all_sfx.extend(a["sfx_samples"])
    print(f"\n  --- Sound Effects ---")
    print(f"    Files with SFX: {sfx_files}/{N}")
    if all_sfx:
        sc = defaultdict(int)
        for s in all_sfx: sc[s] += 1
        print(f"    Top SFX: {', '.join(f'{s}({c})' for s,c in sorted(sc.items(),key=lambda x:-x[1])[:8])}")

    # Save
    with open("lrc_analysis.json", "w") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Analysis saved to lrc_analysis.json")

    return analyses


def analyze(text, path=""):
    lines = text.strip().split("\n")
    r = {"path": path, "timed": 0, "empty": 0, "ts_prec": "unknown",
         "langs": set(), "sfx_count": 0, "sfx_samples": [],
         "avg_gap": None, "duration": None, "bilingual_lines": 0}

    ts_re = re.compile(r'\[(\d{2}):(\d{2})\.(\d{1,3})\]')
    timestamps = []

    for line in lines:
        line = line.strip()
        if not line or re.match(r'\[(ti|ar|al|by|offset|re|ve):', line):
            continue
        m = ts_re.match(line)
        if not m:
            continue
        r["timed"] += 1
        mm, ss, frac = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(frac) == 3: r["ts_prec"] = "ms"
        elif r["ts_prec"] != "ms": r["ts_prec"] = "cs"
        ts = mm*60 + ss + int(frac.ljust(3,'0')[:3])/1000
        timestamps.append(ts)

        txt = ts_re.sub("", line).strip()
        if not txt:
            r["empty"] += 1
            continue

        has_kana = bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff]', txt))
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', txt))
        has_en = bool(re.search(r'[a-zA-Z]{3,}', txt))

        if has_kana: r["langs"].add("ja")
        if has_cjk: r["langs"].add("zh")
        if has_en: r["langs"].add("en")
        if has_kana and has_cjk:
            r["bilingual_lines"] += 1

        sfx = re.findall(r'\*[^*]+\*', txt)
        if sfx:
            r["sfx_count"] += len(sfx)
            r["sfx_samples"].extend(sfx[:3])

    if len(timestamps) >= 2:
        gaps = [timestamps[i+1]-timestamps[i] for i in range(len(timestamps)-1)
                if timestamps[i+1] > timestamps[i]]
        if gaps:
            r["avg_gap"] = sum(gaps)/len(gaps)
        r["duration"] = max(timestamps) - min(timestamps)

    r["langs"] = sorted(r["langs"])
    r["sfx_samples"] = r["sfx_samples"][:10]
    return r


def main():
    t0 = time.time()

    overview, parts = phase_a()
    lrc_paths, work_details = phase_b(overview, parts)
    analyses = phase_c(lrc_paths)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"API requests: {reqs}")

    # Save everything
    results = {
        "overview": {p: {k: v for k, v in d.items() if k != "cat_details"}
                     for p, d in overview.items()},
        "overview_details": overview,
        "lrc_count": len(lrc_paths),
        "lrc_paths": lrc_paths,
        "works_with_lrc": work_details,
        "elapsed": elapsed,
        "requests": reqs,
    }
    with open("scan_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"Results saved to scan_results.json")


if __name__ == "__main__":
    main()

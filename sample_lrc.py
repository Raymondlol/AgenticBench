#!/usr/bin/env python3
"""Download and analyze sample LRC files from known LRC-rich directories."""

import json
import time
import urllib.request
import urllib.error
import re
import os
from collections import defaultdict

API_LIST = "http://asmrgay.com/api/fs/list"
API_GET = "http://asmrgay.com/api/fs/get"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
DELAY = 0.5
MAX_RETRIES = 4

SAMPLE_DIR = "lrc_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)


def api_call(url, payload_dict):
    payload = json.dumps(payload_dict).encode()
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=payload, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            time.sleep(DELAY)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                wait = 3 * (2 ** attempt)
                print(f"  [429] waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [ERR {e.code}] {url}")
            return None
        except Exception as e:
            print(f"  [ERR] {e}")
            return None
    return None


def list_dir(path, page=1, per_page=200):
    data = api_call(API_LIST, {"path": path, "page": page, "per_page": per_page})
    if not data or data["code"] != 200:
        return []
    return data["data"].get("content") or []


def get_file_url(path):
    data = api_call(API_GET, {"path": path, "password": ""})
    if not data or data["code"] != 200:
        return None
    return data["data"].get("raw_url")


def download_lrc(path, local_name):
    """Download LRC file content."""
    raw_url = get_file_url(path)
    if not raw_url:
        return None
    # Use HTTP version of the download URL
    dl_url = raw_url.replace("https://", "http://")
    try:
        req = urllib.request.Request(dl_url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
        # Try UTF-8 first, then shift-jis, then latin-1
        for enc in ["utf-8-sig", "utf-8", "shift_jis", "latin-1"]:
            try:
                text = content.decode(enc)
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            text = content.decode("latin-1")

        local_path = os.path.join(SAMPLE_DIR, local_name)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(text)
        time.sleep(DELAY)
        return text
    except Exception as e:
        print(f"  [DL ERR] {e}")
        return None


def analyze_lrc(text, path=""):
    """Analyze a single LRC file."""
    lines = text.strip().split("\n")
    result = {
        "path": path,
        "total_lines": len(lines),
        "timed_lines": 0,
        "empty_timed_lines": 0,  # [mm:ss.xx] followed by nothing
        "has_offset": False,
        "has_metadata": False,
        "timestamp_precision": "unknown",
        "languages_detected": set(),
        "has_japanese": False,
        "has_chinese": False,
        "has_english": False,
        "has_bilingual_lines": False,  # single line with both ja + zh
        "sound_effects": [],
        "avg_gap_seconds": 0,
        "min_gap_seconds": 999,
        "max_gap_seconds": 0,
    }

    ts_pattern = re.compile(r'\[(\d{2}):(\d{2})\.(\d{1,3})\]')
    timestamps = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Metadata tags
        if re.match(r'\[(ti|ar|al|by|offset|re|ve):', line):
            result["has_metadata"] = True
            if line.startswith("[offset:"):
                result["has_offset"] = True
            continue

        # Timed lines
        ts_match = ts_pattern.match(line)
        if ts_match:
            result["timed_lines"] += 1
            m, s, cs = int(ts_match.group(1)), int(ts_match.group(2)), ts_match.group(3)
            # Detect precision
            if len(cs) == 3:
                result["timestamp_precision"] = "millisecond"
            elif len(cs) == 2:
                if result["timestamp_precision"] != "millisecond":
                    result["timestamp_precision"] = "centisecond"
            ts_sec = m * 60 + s + int(cs.ljust(3, '0')[:3]) / 1000
            timestamps.append(ts_sec)

            # Text after timestamp
            text_part = ts_pattern.sub("", line).strip()
            if not text_part:
                result["empty_timed_lines"] += 1
                continue

            # Language detection (simple heuristic)
            has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text_part))
            has_jp = bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text_part))
            has_en = bool(re.search(r'[a-zA-Z]{3,}', text_part))

            if has_jp:
                result["has_japanese"] = True
                result["languages_detected"].add("ja")
            if has_cjk:
                result["has_chinese"] = True
                result["languages_detected"].add("zh")
            if has_en:
                result["has_english"] = True
                result["languages_detected"].add("en")

            # Check bilingual in same line
            if has_jp and has_cjk and not has_jp:
                result["has_bilingual_lines"] = True

            # Sound effects
            sfx = re.findall(r'\*[^*]+\*', text_part)
            result["sound_effects"].extend(sfx)

    # Gap analysis
    if len(timestamps) >= 2:
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        gaps = [g for g in gaps if g > 0]
        if gaps:
            result["avg_gap_seconds"] = sum(gaps) / len(gaps)
            result["min_gap_seconds"] = min(gaps)
            result["max_gap_seconds"] = max(gaps)

    result["languages_detected"] = sorted(result["languages_detected"])
    return result


def main():
    # Known directories likely to have LRC files
    # From earlier exploration: 音声汉化 has LRC-paired audio
    targets = [
        "/asmr/音声汉化",
        "/asmr/橙澄子汉化组",
        "/asmr/风花雪月汉化组",
        "/asmr/大山チロル",
        "/asmr/天知遥",
        "/asmr/柚木つばめ",
        "/asmr/野上菜月",
        "/asmr/清软喵",
    ]

    lrc_paths = []

    # Phase 1: Find LRC files in target directories (2 levels deep)
    print("Phase 1: Finding LRC files in target directories...\n")
    for target in targets:
        print(f"Scanning {target}...")
        items = list_dir(target)
        if not items:
            print(f"  (empty or error)")
            continue

        for item in items[:30]:  # cap per target to avoid rate limiting
            if item["is_dir"]:
                sub_path = target + "/" + item["name"]
                sub_items = list_dir(sub_path)
                for si in sub_items:
                    if not si["is_dir"] and si["name"].lower().endswith(".lrc"):
                        lrc_paths.append(sub_path + "/" + si["name"])
                    elif si["is_dir"]:
                        # One more level
                        sub2_path = sub_path + "/" + si["name"]
                        sub2_items = list_dir(sub2_path)
                        for s2i in sub2_items:
                            if not s2i["is_dir"] and s2i["name"].lower().endswith(".lrc"):
                                lrc_paths.append(sub2_path + "/" + s2i["name"])
            elif item["name"].lower().endswith(".lrc"):
                lrc_paths.append(target + "/" + item["name"])

        print(f"  found {len(lrc_paths)} LRC files so far")

    print(f"\nTotal LRC paths found: {len(lrc_paths)}")

    # Phase 2: Download and analyze samples (up to 30)
    print(f"\nPhase 2: Downloading up to 30 samples for analysis...\n")
    # Take evenly spaced samples
    if len(lrc_paths) > 30:
        step = len(lrc_paths) // 30
        sample_paths = lrc_paths[::step][:30]
    else:
        sample_paths = lrc_paths[:30]

    analyses = []
    for i, lpath in enumerate(sample_paths):
        safe_name = f"sample_{i:03d}.lrc"
        print(f"  [{i+1}/{len(sample_paths)}] {lpath.split('/')[-1]}")
        text = download_lrc(lpath, safe_name)
        if text:
            analysis = analyze_lrc(text, lpath)
            analyses.append(analysis)

    # Phase 3: Aggregate analysis
    print(f"\n{'='*70}")
    print("LRC QUALITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Samples analyzed: {len(analyses)}")

    if not analyses:
        print("No samples collected!")
        return

    # Language distribution
    lang_counts = defaultdict(int)
    for a in analyses:
        for lang in a["languages_detected"]:
            lang_counts[lang] += 1
    print(f"\n--- Language Detection ---")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        label = {"ja": "Japanese", "zh": "Chinese", "en": "English"}.get(lang, lang)
        print(f"  {label}: {count}/{len(analyses)} ({count/len(analyses)*100:.0f}%)")

    ja_only = sum(1 for a in analyses if a["has_japanese"] and not a["has_chinese"])
    zh_only = sum(1 for a in analyses if a["has_chinese"] and not a["has_japanese"])
    both = sum(1 for a in analyses if a["has_chinese"] and a["has_japanese"])
    print(f"\n  Japanese only : {ja_only}")
    print(f"  Chinese only  : {zh_only}")
    print(f"  Both (ja+zh CJK)  : {both}")
    bilingual = sum(1 for a in analyses if a["has_bilingual_lines"])
    print(f"  Bilingual lines in same line: {bilingual}")

    # Timestamp precision
    prec_counts = defaultdict(int)
    for a in analyses:
        prec_counts[a["timestamp_precision"]] += 1
    print(f"\n--- Timestamp Precision ---")
    for prec, count in sorted(prec_counts.items(), key=lambda x: -x[1]):
        print(f"  {prec}: {count}")

    # Gap analysis
    gaps = [a["avg_gap_seconds"] for a in analyses if a["avg_gap_seconds"] > 0]
    if gaps:
        print(f"\n--- Timing Gaps ---")
        print(f"  Mean avg gap: {sum(gaps)/len(gaps):.1f}s")
        print(f"  Min avg gap : {min(gaps):.1f}s")
        print(f"  Max avg gap : {max(gaps):.1f}s")

    # Line counts
    timed = [a["timed_lines"] for a in analyses]
    empty = [a["empty_timed_lines"] for a in analyses]
    print(f"\n--- Line Statistics ---")
    print(f"  Mean timed lines per file : {sum(timed)/len(timed):.0f}")
    print(f"  Mean empty timed lines    : {sum(empty)/len(empty):.0f}")
    print(f"  Total timed lines sampled : {sum(timed)}")

    # Sound effects
    all_sfx = []
    for a in analyses:
        all_sfx.extend(a["sound_effects"])
    if all_sfx:
        sfx_counts = defaultdict(int)
        for s in all_sfx:
            sfx_counts[s] += 1
        print(f"\n--- Sound Effect Annotations (top 15) ---")
        for sfx, count in sorted(sfx_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {sfx}: {count}")

    # Metadata/offset
    with_offset = sum(1 for a in analyses if a["has_offset"])
    with_meta = sum(1 for a in analyses if a["has_metadata"])
    print(f"\n--- Metadata ---")
    print(f"  With [offset:] tag  : {with_offset}/{len(analyses)}")
    print(f"  With metadata tags  : {with_meta}/{len(analyses)}")

    # Save full results
    for a in analyses:
        a["sound_effects"] = a["sound_effects"][:10]  # trim for JSON
    with open("lrc_analysis.json", "w") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nFull analysis saved to lrc_analysis.json")
    print(f"Sample files saved to {SAMPLE_DIR}/")


if __name__ == "__main__":
    main()

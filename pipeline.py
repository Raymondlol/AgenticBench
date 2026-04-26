#!/usr/bin/env python3
"""
ASMR-Data sampling pipeline.

Phase 1: Crawl all subtitle work metadata → meta/all_works.jsonl
Phase 2: Filter & select 30 official CN works → meta/selected.json
Phase 3: Download VTT first, then audio → chinese_asr/{work_id}/
"""

import json, time, urllib.request, urllib.error, os, sys, re, hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

# ── Config ──────────────────────────────────────────────────────────
API        = "https://api.asmr-200.com/api"
HDR        = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
BASE_DIR   = Path("/Users/raymond/WorkSpace/ASMR-Data")
META_DIR   = BASE_DIR / "meta"
ASR_DIR    = BASE_DIR / "chinese_asr"
STATE_FILE = META_DIR / "pipeline_state.json"

DELAY       = 1.0        # base delay between requests
BACKOFF     = 5          # initial backoff on 429
MAX_RETRY   = 6
PAGE_SIZE   = 20
TARGET_N    = 150        # ~50h pilot dataset (~30 existing + ~120 new)
MIN_DUR     = 15 * 60   # 15 min in seconds
MAX_DUR     = 50 * 60   # 50 min (relaxed from 40)
MIN_VTT_LINES = 100

# Diversity caps (to avoid acoustic homogeneity within series/circle/CV)
MAX_PER_SERIES = 1      # at most 1 work per original_workno series
MAX_PER_CIRCLE = 2      # at most 2 works per circle (studio)
MAX_PER_CV     = 2      # at most 2 works per primary CV
RANDOM_SEED    = 42     # deterministic shuffle

# ── HTTP helpers ────────────────────────────────────────────────────

request_count = 0

def api_get(url, retries=MAX_RETRY):
    """GET JSON with retry + exponential backoff on 429."""
    global request_count
    for attempt in range(retries + 1):
        request_count += 1
        req = urllib.request.Request(url, headers=HDR)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            time.sleep(DELAY)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = BACKOFF * (2 ** attempt)
                print(f"    [429] retry in {wait}s (attempt {attempt+1})", flush=True)
                time.sleep(wait)
                continue
            print(f"    [HTTP {e.code}] {url[:100]}", flush=True)
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"    [ERR] {e}", flush=True)
            return None
    return None


def download_file(url, dest, max_size=None):
    """Download file with resume support. Returns True on success."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Resume: if file exists and looks complete, skip
    if dest.exists() and dest.stat().st_size > 0:
        return True

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers=HDR)

    # Resume partial download
    existing = 0
    if tmp.exists():
        existing = tmp.stat().st_size
        req.add_header("Range", f"bytes={existing}-")

    for attempt in range(MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if existing > 0 and r.status == 206 else "wb"
                if mode == "wb":
                    existing = 0
                total = int(r.headers.get("Content-Length", 0)) + existing

                if max_size and total > max_size:
                    print(f"    [SKIP] too large: {total/1024/1024:.0f}MB", flush=True)
                    return False

                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)

            tmp.rename(dest)
            time.sleep(DELAY)
            return True

        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRY:
                wait = BACKOFF * (2 ** attempt)
                print(f"    [429] retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"    [DL HTTP {e.code}]", flush=True)
            return False
        except Exception as e:
            if attempt < MAX_RETRY:
                time.sleep(2)
                continue
            print(f"    [DL ERR] {e}", flush=True)
            return False
    return False


# ── State management ────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"phase": "init", "crawl_page": 0, "downloaded": []}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ── Phase 1: Crawl metadata ────────────────────────────────────────

def phase1_crawl(state):
    """Crawl all subtitle works metadata."""
    print("=" * 60)
    print("PHASE 1: Crawl subtitle work metadata")
    print("=" * 60)

    works_file = META_DIR / "all_works.jsonl"
    start_page = state.get("crawl_page", 0) + 1

    # Count existing
    existing_ids = set()
    if works_file.exists():
        with open(works_file) as f:
            for line in f:
                w = json.loads(line)
                existing_ids.add(w["id"])
        print(f"  Resuming from page {start_page}, {len(existing_ids)} works already crawled")

    # Get total
    d = api_get(f"{API}/works?subtitle=1&page=1&limit=1")
    if not d:
        print("  Failed to get total count!")
        return False
    total = d["pagination"]["totalCount"]
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"  Total subtitle works: {total} ({total_pages} pages)")

    with open(works_file, "a") as f:
        for page in range(start_page, total_pages + 1):
            d = api_get(f"{API}/works?subtitle=1&order=release&sort=desc&page={page}&limit={PAGE_SIZE}")
            if not d or not d.get("works"):
                print(f"  [WARN] Empty page {page}, skipping")
                continue

            new = 0
            for w in d["works"]:
                if w["id"] not in existing_ids:
                    # Extract key fields
                    record = {
                        "id": w["id"],
                        "source_id": w.get("source_id", ""),
                        "title": w.get("title", ""),
                        "release": w.get("release", ""),
                        "duration": w.get("duration", 0),
                        "dl_count": w.get("dl_count", 0),
                        "has_subtitle": w.get("has_subtitle", False),
                        "nsfw": w.get("nsfw", False),
                        "circle_name": w.get("name", ""),
                        "vas": [v.get("name", "") for v in w.get("vas", [])],
                        "tags": [t.get("name", "") for t in w.get("tags", [])],
                        "translation_lang": w.get("translation_info", {}).get("lang"),
                        "original_workno": w.get("translation_info", {}).get("original_workno"),
                        "is_original": w.get("translation_info", {}).get("is_original", False),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    existing_ids.add(w["id"])
                    new += 1

            if page % 20 == 0:
                print(f"  page {page}/{total_pages}  total={len(existing_ids)}  (+{new})", flush=True)
                f.flush()

            state["crawl_page"] = page
            if page % 50 == 0:
                save_state(state)

    state["crawl_page"] = total_pages
    state["phase"] = "crawled"
    save_state(state)
    print(f"  Done: {len(existing_ids)} works saved to all_works.jsonl")
    return True


# ── Phase 2: Filter & select ───────────────────────────────────────

def phase2_select():
    """Select 30 official CN works for download."""
    print(f"\n{'='*60}")
    print("PHASE 2: Filter & select works")
    print("=" * 60)

    works_file = META_DIR / "all_works.jsonl"
    works = []
    with open(works_file) as f:
        for line in f:
            works.append(json.loads(line))

    print(f"  Total works loaded: {len(works)}")

    # Split by type
    official_cn = [w for w in works if w.get("translation_lang") in ("CHI_HANS", "CHI_HANT")]
    crowd_lrc = [w for w in works if w.get("translation_lang") not in ("CHI_HANS", "CHI_HANT")]
    print(f"  Official CN editions: {len(official_cn)}")
    print(f"  Crowd-sourced LRC:    {len(crowd_lrc)}")

    # Filter official CN: duration in [MIN_DUR, MAX_DUR]
    candidates = [w for w in official_cn if MIN_DUR <= w["duration"] <= MAX_DUR]
    print(f"  After duration filter ({MIN_DUR//60}-{MAX_DUR//60}min): {len(candidates)}")

    # Random shuffle (deterministic) to break popularity bias and diversify acoustic conditions
    import random
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(candidates)
    print(f"  Shuffled with seed={RANDOM_SEED} for acoustic diversity")

    # Identify existing downloaded works (preserve them, they're already on disk)
    existing_ids = set()
    if ASR_DIR.exists():
        for d in ASR_DIR.iterdir():
            if d.is_dir() and d.name.isdigit():
                existing_ids.add(int(d.name))
    print(f"  Existing downloaded works: {len(existing_ids)} (will be preserved)")

    def update_counts(w, series_count, circle_count, cv_count):
        series = w.get("original_workno") or w["source_id"]
        circle = w["circle_name"] or "unknown"
        primary_cv = w["vas"][0] if w["vas"] else "unknown"
        series_count[series] = series_count.get(series, 0) + 1
        circle_count[circle] = circle_count.get(circle, 0) + 1
        cv_count[primary_cv] = cv_count.get(primary_cv, 0) + 1

    # Greedy diversity selection: cap works per series/circle/CV
    selected = []
    series_count = {}
    circle_count = {}
    cv_count = {}

    # Pass 1: include all already-downloaded works (don't waste them)
    for w in candidates:
        if w["id"] in existing_ids:
            selected.append(w)
            update_counts(w, series_count, circle_count, cv_count)

    # Pass 2: add new works respecting diversity caps
    for w in candidates:
        if len(selected) >= TARGET_N:
            break
        if w["id"] in existing_ids:
            continue  # already added in pass 1
        series = w.get("original_workno") or w["source_id"]
        circle = w["circle_name"] or "unknown"
        primary_cv = w["vas"][0] if w["vas"] else "unknown"

        if series_count.get(series, 0) >= MAX_PER_SERIES:
            continue
        if circle_count.get(circle, 0) >= MAX_PER_CIRCLE:
            continue
        if cv_count.get(primary_cv, 0) >= MAX_PER_CV:
            continue

        selected.append(w)
        update_counts(w, series_count, circle_count, cv_count)

    used_cvs = set(cv_count.keys())
    used_circles = set(circle_count.keys())

    print(f"  Selected: {len(selected)} works")
    print(f"  Unique CVs: {len(used_cvs)}")
    print(f"  Unique circles: {len(used_circles)}")
    total_dur = sum(w["duration"] for w in selected)
    print(f"  Total duration: {total_dur/3600:.1f}h")

    # Summary
    print(f"\n  Selected works:")
    for i, w in enumerate(selected):
        cvs = ", ".join(w["vas"][:3]) or "N/A"
        print(f"    {i+1:2d}. {w['source_id']}  {w['duration']//60:>2d}min  dl={w['dl_count']:>5d}  "
              f"CV=[{cvs}]  {w['title'][:40]}")

    # Save selection
    sel_file = META_DIR / "selected.json"
    with open(sel_file, "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {sel_file}")

    # Also save crowd LRC summary
    crowd_file = META_DIR / "crowd_lrc_works.json"
    with open(crowd_file, "w") as f:
        json.dump(crowd_lrc[:200], f, ensure_ascii=False, indent=2)

    return selected


# ── Phase 3: Download ───────────────────────────────────────────────

def extract_files(tracks_data):
    """Walk tracks tree and extract audio + subtitle files."""
    files = []
    def walk(node, folder=""):
        if isinstance(node, list):
            for item in node:
                walk(item, folder)
        elif isinstance(node, dict):
            t = node.get("type", "")
            title = node.get("title", "")

            if t in ("audio", "text") and title:
                ext = Path(title).suffix.lower()
                is_sub = ext in (".vtt", ".lrc", ".srt", ".ass")
                is_audio = ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg")

                if is_sub or is_audio:
                    # Prefer download URL, fall back to stream
                    url = node.get("mediaDownloadUrl") or node.get("mediaStreamUrl", "")
                    files.append({
                        "title": title,
                        "type": "subtitle" if is_sub else "audio",
                        "ext": ext,
                        "url": url,
                        "size": node.get("size", 0),
                        "duration": node.get("duration", 0),
                        "folder": folder,
                    })

            if node.get("type") == "folder":
                child_folder = folder + "/" + title if folder else title
                for child in node.get("children", []):
                    walk(child, child_folder)
            else:
                for child in node.get("children", []):
                    walk(child, folder)
    walk(tracks_data)
    return files


def pick_best_audio(files):
    """For each subtitle file, pick matching audio. Prefer mp3 over wav (smaller)."""
    subs = [f for f in files if f["type"] == "subtitle"]
    audios = [f for f in files if f["type"] == "audio"]

    # Build stem → audio map
    audio_map = defaultdict(list)
    for a in audios:
        stem = Path(a["title"]).stem
        # Handle cases like "Track1.mp3.vtt" → stem is "Track1.mp3"
        if stem.endswith((".mp3", ".wav", ".flac", ".m4a")):
            stem = Path(stem).stem
        audio_map[stem].append(a)

    pairs = []
    used_audio_urls = set()

    for sub in subs:
        stem = Path(sub["title"]).stem
        # "Track1.mp3.vtt" → "Track1"
        if stem.endswith((".mp3", ".wav", ".flac", ".m4a")):
            stem = Path(stem).stem

        candidates = audio_map.get(stem, [])
        if not candidates:
            # Try fuzzy match: strip common suffixes
            for astem, alist in audio_map.items():
                if astem.startswith(stem) or stem.startswith(astem):
                    candidates = alist
                    break

        if candidates:
            # Prefer mp3 (smaller), then m4a, then wav
            pref = {".mp3": 0, ".m4a": 1, ".flac": 2, ".wav": 3}
            candidates.sort(key=lambda a: pref.get(a["ext"], 9))
            audio = candidates[0]
            if audio["url"] not in used_audio_urls:
                pairs.append((sub, audio))
                used_audio_urls.add(audio["url"])
        else:
            pairs.append((sub, None))

    return pairs


def sanitize(name, max_len=80):
    """Make filename filesystem-safe."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    return name[:max_len]


def phase3_download(selected, state):
    """Download VTT + audio for selected works."""
    print(f"\n{'='*60}")
    print("PHASE 3: Download")
    print("=" * 60)

    downloaded = set(state.get("downloaded", []))

    for idx, work in enumerate(selected):
        wid = work["id"]
        sid = work["source_id"]

        if str(wid) in downloaded:
            print(f"  [{idx+1}/{len(selected)}] {sid} — already done, skipping")
            continue

        print(f"\n  [{idx+1}/{len(selected)}] {sid}  {work['title'][:50]}", flush=True)

        work_dir = ASR_DIR / str(wid)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Get tracks
        tracks = api_get(f"{API}/tracks/{wid}")
        if not tracks:
            print(f"    [SKIP] failed to get tracks")
            continue

        files = extract_files(tracks)
        pairs = pick_best_audio(files)
        subs = [p[0] for p in pairs]
        valid_pairs = [(s, a) for s, a in pairs if a is not None]

        print(f"    files: {len(files)} total, {len(subs)} subs, {len(valid_pairs)} paired")

        if not subs:
            print(f"    [SKIP] no subtitle files found")
            continue

        # Step 1: Download subtitles first
        sub_ok = 0
        total_lines = 0
        for sub, _ in pairs:
            dest = work_dir / sanitize(sub["title"])
            if sub["url"]:
                ok = download_file(sub["url"], dest, max_size=1024*1024)  # 1MB max
                if ok:
                    sub_ok += 1
                    # Count lines
                    try:
                        text = dest.read_text(encoding="utf-8-sig", errors="replace")
                        total_lines += len(re.findall(r'\d{2}:\d{2}', text))
                    except:
                        pass

        print(f"    subtitles: {sub_ok} downloaded, ~{total_lines} timed lines")

        # Quality gate: skip if too few lines
        if total_lines < MIN_VTT_LINES:
            print(f"    [SKIP] only {total_lines} lines (need {MIN_VTT_LINES}+)")
            # Clean up
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
            continue

        # Step 2: Download audio (mp3 preferred, limit 500MB per work)
        audio_ok = 0
        audio_bytes = 0
        MAX_WORK_SIZE = 500 * 1024 * 1024  # 500MB

        for sub, audio in valid_pairs:
            if audio is None:
                continue
            if audio_bytes + audio["size"] > MAX_WORK_SIZE:
                print(f"    [CAP] size limit reached ({audio_bytes/1024/1024:.0f}MB)")
                break

            dest = work_dir / sanitize(audio["title"])
            url = audio["url"]
            # Prefer stream URL for mp3 (often points to m4a/mp3 on fast CDN)
            if audio.get("ext") == ".wav" and "streamLowQualityUrl" in str(tracks):
                # Try to find low quality stream
                pass

            ok = download_file(url, dest, max_size=MAX_WORK_SIZE - audio_bytes)
            if ok:
                audio_ok += 1
                audio_bytes += dest.stat().st_size

        print(f"    audio: {audio_ok} downloaded ({audio_bytes/1024/1024:.1f}MB)")

        # Save per-work metadata
        work_meta = {
            **work,
            "download_time": datetime.now().isoformat(),
            "subtitle_files": [sanitize(p[0]["title"]) for p in pairs if p[0]],
            "audio_files": [sanitize(p[1]["title"]) for p in valid_pairs if p[1]],
            "total_subtitle_lines": total_lines,
            "total_audio_bytes": audio_bytes,
        }
        (work_dir / "metadata.json").write_text(
            json.dumps(work_meta, ensure_ascii=False, indent=2))

        downloaded.add(str(wid))
        state["downloaded"] = list(downloaded)
        save_state(state)
        print(f"    ✓ done")

    return downloaded


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    META_DIR.mkdir(parents=True, exist_ok=True)
    ASR_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    print(f"Pipeline state: {state.get('phase', 'init')}")
    print(f"Target dir: {BASE_DIR}\n")

    # Phase 1
    if state.get("phase") not in ("crawled", "selected", "done"):
        if not phase1_crawl(state):
            print("Phase 1 failed!")
            return

    # Phase 2
    selected = phase2_select()
    state["phase"] = "selected"
    save_state(state)

    if not selected:
        print("No works selected!")
        return

    # Phase 3
    downloaded = phase3_download(selected, state)

    state["phase"] = "done"
    save_state(state)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"API requests: {request_count}")
    print(f"Works downloaded: {len(downloaded)}")
    print(f"Output: {ASR_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Find genuine Chinese-dubbed ASMR works from asmr.one:
  1. Filter candidates from all_works.jsonl (Chinese title, is_original, has_subtitle)
  2. For each candidate, fetch track listing from API
  3. Download first 30s of first audio track
  4. Run whisper language detection
  5. Save results
"""

import json, sys, time, os, tempfile, struct
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

META_FILE   = Path("/Users/raymond/WorkSpace/ASMR-Data/meta/all_works.jsonl")
RESULTS_DIR = Path("/Users/raymond/WorkSpace/ASMR-Data/meta")
STATE_FILE  = RESULTS_DIR / "lang_detect_state.json"
API_BASE    = "https://api.asmr-200.com/api"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# ── Step 1: Find candidates ────────────────────────────────────────

def find_candidates():
    """Find is_original=True works with Chinese-dominant titles and subtitles."""
    import re
    candidates = []
    for line in open(META_FILE):
        w = json.loads(line)
        if not w.get("has_subtitle"):
            continue
        title = w.get("title", "")
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', title))
        jp_kana = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', title))

        # Chinese-dominant title: many Chinese chars, few kana
        if cn_chars >= 5 and jp_kana <= 3:
            candidates.append(w)

    return candidates


# ── Step 2: Get first audio URL from API ───────────────────────────

def api_get(path):
    url = f"{API_BASE}/{path}"
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())


def get_first_audio_url(work_id):
    """Get the URL of the first audio file from a work's track listing."""
    tracks = api_get(f"tracks/{work_id}")

    audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

    def walk(items):
        for item in items:
            if item.get("type") == "folder":
                result = walk(item.get("children", []))
                if result:
                    return result
            else:
                title = item.get("title", "")
                ext = os.path.splitext(title)[1].lower()
                if ext in audio_exts:
                    url = item.get("mediaStreamUrl") or item.get("mediaDownloadUrl")
                    if url:
                        return url, title
    return walk(tracks) if isinstance(tracks, list) else None


# ── Step 3: Download first N seconds ──────────────────────────────

def download_partial(url, max_bytes=500_000):
    """Download first ~500KB of audio (enough for language detection)."""
    req = urllib.request.Request(url, headers={
        **HEADERS,
        "Range": f"bytes=0-{max_bytes}",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read()
    except Exception:
        # Some servers don't support Range, download full but limit
        req2 = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req2, timeout=30)
        data = resp.read(max_bytes + 1)
        return data


# ── Step 4: Detect language with whisper ───────────────────────────

_whisper_loaded = False

def detect_language(audio_path):
    """Use whisper to detect language of audio file."""
    global _whisper_loaded
    import mlx_whisper

    if not _whisper_loaded:
        print("  Loading whisper model...", flush=True)
        _whisper_loaded = True

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        verbose=False,
    )
    return result.get("language", "unknown"), result.get("text", "")[:100]


# ── State management ──────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"checked": {}, "chinese_works": []}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ── Main ──────────────────────────────────────────────────────────

def main():
    state = load_state()
    candidates = find_candidates()
    print(f"Candidates: {len(candidates)}")
    print(f"Already checked: {len(state['checked'])}")

    remaining = [w for w in candidates if w["source_id"] not in state["checked"]]
    print(f"Remaining: {len(remaining)}\n")

    chinese_count = sum(1 for v in state["checked"].values() if v.get("lang") in ("zh", "chinese", "Chinese"))

    for i, w in enumerate(remaining):
        sid = w["source_id"]
        wid = str(w["id"])
        title = w["title"][:55]

        print(f"[{len(state['checked'])+1}/{len(candidates)}] {sid} {title}", flush=True)

        # Get audio URL
        try:
            result = get_first_audio_url(wid)
            if not result:
                state["checked"][sid] = {"lang": "no_audio", "title": w["title"]}
                save_state(state)
                print(f"  -> no audio found", flush=True)
                continue
            audio_url, audio_name = result
        except Exception as e:
            state["checked"][sid] = {"lang": "api_error", "error": str(e), "title": w["title"]}
            save_state(state)
            print(f"  -> API error: {e}", flush=True)
            time.sleep(1)
            continue

        # Download partial audio
        try:
            audio_data = download_partial(audio_url)
        except Exception as e:
            state["checked"][sid] = {"lang": "download_error", "error": str(e), "title": w["title"]}
            save_state(state)
            print(f"  -> download error: {e}", flush=True)
            time.sleep(1)
            continue

        # Save to temp file and detect language
        ext = os.path.splitext(audio_name)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        try:
            lang, text_preview = detect_language(tmp_path)
        except Exception as e:
            lang = "detect_error"
            text_preview = str(e)
        finally:
            os.unlink(tmp_path)

        state["checked"][sid] = {
            "lang": lang,
            "title": w["title"],
            "audio_name": audio_name,
            "text_preview": text_preview,
            "id": w["id"],
            "duration": w.get("duration", 0),
            "dl_count": w.get("dl_count", 0),
        }

        if lang in ("zh", "chinese", "Chinese"):
            chinese_count += 1
            state["chinese_works"].append(sid)
            print(f"  -> ✓ CHINESE  [{text_preview[:40]}]", flush=True)
        else:
            print(f"  -> {lang}  [{text_preview[:40]}]", flush=True)

        save_state(state)

        # Rate limit
        time.sleep(0.3)

    # Summary
    print(f"\n{'='*60}")
    print(f"  LANGUAGE DETECTION SUMMARY")
    print(f"{'='*60}")
    lang_counts = {}
    for v in state["checked"].values():
        lang = v.get("lang", "?")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang:>12s}: {count}")

    print(f"\n  Chinese audio works: {chinese_count}")
    if state["chinese_works"]:
        print(f"\n  --- Chinese Works ---")
        for sid in state["chinese_works"]:
            info = state["checked"][sid]
            print(f"  {sid}  dur={info.get('duration',0)//60}min  {info['title'][:50]}")

    save_state(state)

if __name__ == "__main__":
    main()

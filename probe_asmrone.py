#!/usr/bin/env python3
"""
Probe asmr.one at scale:
  1. Sample works with subtitles across different time periods
  2. Download LRC samples and detect language
  3. Check audio download accessibility
  4. Estimate total usable data volume
"""

import json, time, urllib.request, urllib.error, re, sys, os
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

API = "https://api.asmr-200.com/api"
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
DELAY = 0.3

def get(url):
    req = urllib.request.Request(url, headers=HDR)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            time.sleep(DELAY)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3 * (2**attempt))
                continue
            return None
        except:
            return None
    return None

def fetch_raw(url):
    """Fetch raw bytes from URL."""
    req = urllib.request.Request(url, headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read()
        time.sleep(DELAY)
    except:
        return None

def detect_lang(text):
    """Detect language of LRC text content."""
    # Strip timestamps and metadata
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or re.match(r'\[(ti|ar|al|by|offset|re|ve):', line):
            continue
        # Remove timestamp
        line = re.sub(r'\[\d{2}:\d{2}\.\d{1,3}\]', '', line).strip()
        if line:
            lines.append(line)

    full = ' '.join(lines)
    if not full:
        return "empty"

    has_kana = bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff]', full))
    has_cjk = bool(re.search(r'[\u4e00-\u9fff]', full))
    has_hangul = bool(re.search(r'[\uac00-\ud7af]', full))
    has_latin = bool(re.search(r'[a-zA-Z]{3,}', full))

    # Count characters
    kana_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', full))
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', full))

    langs = []
    if has_kana:
        langs.append("ja")
    if has_cjk:
        # Distinguish Chinese from Japanese kanji usage
        if kana_count > 0 and kana_count > cjk_count * 0.1:
            if "ja" not in langs:
                langs.append("ja")
        else:
            langs.append("zh")
    if has_hangul:
        langs.append("ko")
    if has_latin and not has_kana and not has_cjk:
        langs.append("en")

    if not langs:
        return "unknown"
    return "+".join(langs)


def find_lrc_urls(tracks_data):
    """Extract all LRC stream URLs from tracks data."""
    results = []
    def walk(node):
        if isinstance(node, list):
            for i in node: walk(i)
        elif isinstance(node, dict):
            if node.get('type') == 'text' and node.get('title', '').lower().endswith('.lrc'):
                url = node.get('mediaStreamUrl', '')
                results.append({
                    'title': node['title'],
                    'url': url,
                    'size': node.get('size', 0),
                    'duration': node.get('duration', 0),
                })
            for c in node.get('children', []):
                walk(c)
    walk(tracks_data)
    return results


def find_audio_urls(tracks_data):
    """Extract audio file info."""
    results = []
    def walk(node):
        if isinstance(node, list):
            for i in node: walk(i)
        elif isinstance(node, dict):
            if node.get('type') == 'audio':
                results.append({
                    'title': node['title'],
                    'stream': node.get('mediaStreamUrl', ''),
                    'download': node.get('mediaDownloadUrl', ''),
                    'size': node.get('size', 0),
                    'duration': node.get('duration', 0),
                })
            for c in node.get('children', []):
                walk(c)
    walk(tracks_data)
    return results


def main():
    t0 = time.time()
    os.makedirs("asmrone_samples", exist_ok=True)

    # ---- Phase 1: Scale statistics ----
    print("=" * 60)
    print("PHASE 1: Scale statistics")
    print("=" * 60)

    # Total works
    d = get(f"{API}/works?page=1&limit=1")
    total = d['pagination']['totalCount'] if d else 0
    print(f"Total works: {total}")

    # Works with subtitles
    d = get(f"{API}/works?subtitle=1&page=1&limit=1")
    sub_total = d['pagination']['totalCount'] if d else 0
    print(f"Works with subtitles: {sub_total} ({sub_total/total*100:.1f}%)")

    # Sample across different pages to estimate time distribution
    print(f"\nSubtitle works by era (sampled):")
    for page_offset, label in [(1, "newest"), (sub_total//40, "middle"), (sub_total//20 - 1, "oldest")]:
        d = get(f"{API}/works?subtitle=1&order=release&sort=desc&page={page_offset}&limit=5")
        if d and d.get('works'):
            dates = [w['release'] for w in d['works']]
            dls = [w['dl_count'] for w in d['works']]
            durs = [w.get('duration',0) for w in d['works']]
            print(f"  {label:8s}: dates={dates[0]}..{dates[-1]}  "
                  f"avg_dl={sum(dls)//len(dls)}  avg_dur={sum(durs)//len(durs)//60}min")

    # ---- Phase 2: Language sampling ----
    print(f"\n{'='*60}")
    print("PHASE 2: LRC language sampling (50 works)")
    print("=" * 60)

    # Sample 50 works evenly distributed
    sample_pages = []
    total_pages = sub_total // 20
    step = max(1, total_pages // 50)
    for i in range(0, 50):
        sample_pages.append(1 + i * step)

    lang_stats = defaultdict(int)
    lrc_line_counts = []
    lrc_durations = []
    audio_counts = []
    audio_accessible = 0
    audio_tested = 0
    work_samples = []

    for idx, page in enumerate(sample_pages):
        d = get(f"{API}/works?subtitle=1&order=release&sort=desc&page={page}&limit=1")
        if not d or not d.get('works'):
            continue
        work = d['works'][0]
        wid = work['id']

        # Get tracks
        tracks = get(f"{API}/tracks/{wid}")
        if not tracks:
            continue

        lrcs = find_lrc_urls(tracks)
        audios = find_audio_urls(tracks)
        audio_counts.append(len(audios))

        if not lrcs:
            continue

        # Download first LRC
        lrc = lrcs[0]
        raw = fetch_raw(lrc['url'])
        if not raw:
            continue

        # Try encodings
        text = None
        for enc in ("utf-8-sig", "utf-8", "gbk", "shift_jis", "latin-1"):
            try:
                text = raw.decode(enc)
                # Verify: if we get too many replacement chars, try next
                if text.count('\ufffd') > len(text) * 0.1:
                    continue
                # Quick sanity: does it have timestamp lines?
                if re.search(r'\[\d{2}:\d{2}\.\d{1,3}\]', text):
                    break
            except:
                continue
        if not text:
            text = raw.decode('latin-1')

        lang = detect_lang(text)
        lang_stats[lang] += 1

        # Count lines
        timed = len(re.findall(r'\[\d{2}:\d{2}\.\d{1,3}\]', text))
        lrc_line_counts.append(timed)
        if lrc.get('duration'):
            lrc_durations.append(lrc['duration'])

        # Save sample
        with open(f"asmrone_samples/sample_{idx:03d}_{wid}.lrc", "wb") as f:
            f.write(raw)

        # Test audio accessibility (first 10 only)
        if audio_tested < 10 and audios:
            audio_tested += 1
            test_url = audios[0].get('stream') or audios[0].get('download')
            if test_url:
                try:
                    req = urllib.request.Request(test_url, method='HEAD', headers=HDR)
                    with urllib.request.urlopen(req, timeout=10) as r:
                        if r.status == 200:
                            audio_accessible += 1
                except:
                    pass

        work_samples.append({
            "id": wid, "source_id": work.get('source_id'),
            "release": work.get('release'), "lang": lang,
            "lrc_count": len(lrcs), "audio_count": len(audios),
            "timed_lines": timed, "duration": lrc.get('duration', 0),
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/50] sampled. lang dist so far: {dict(lang_stats)}", flush=True)

    # ---- Phase 3: Report ----
    print(f"\n{'='*60}")
    print("RESULTS")
    print("=" * 60)

    print(f"\n--- Scale ---")
    print(f"  Total works in DB       : {total:,}")
    print(f"  Works with subtitles    : {sub_total:,} ({sub_total/total*100:.1f}%)")
    est_tracks = sum(audio_counts) / len(audio_counts) * sub_total if audio_counts else 0
    print(f"  Avg audio tracks/work   : {sum(audio_counts)/len(audio_counts):.1f}" if audio_counts else "")
    print(f"  Est. total audio tracks : ~{est_tracks:,.0f}")
    if lrc_durations:
        avg_dur = sum(lrc_durations) / len(lrc_durations)
        est_hours = avg_dur * est_tracks / 3600
        print(f"  Avg track duration      : {avg_dur/60:.1f}min")
        print(f"  Est. total audio hours  : ~{est_hours:,.0f}h")

    print(f"\n--- Language Distribution ({sum(lang_stats.values())} samples) ---")
    for lang, count in sorted(lang_stats.items(), key=lambda x: -x[1]):
        pct = count / sum(lang_stats.values()) * 100
        label = {"zh": "Chinese only", "ja": "Japanese", "ja+zh": "Japanese+Chinese",
                 "en": "English", "unknown": "Unknown/encoding error", "empty": "Empty",
                 "ko": "Korean"}.get(lang, lang)
        print(f"  {label:30s}: {count:>3} ({pct:.0f}%)")

    print(f"\n--- LRC Content ---")
    if lrc_line_counts:
        print(f"  Timed lines/file: mean={sum(lrc_line_counts)/len(lrc_line_counts):.0f}  "
              f"range={min(lrc_line_counts)}—{max(lrc_line_counts)}")
        est_total_lines = sum(lrc_line_counts) / len(lrc_line_counts) * sub_total
        print(f"  Est. total timed lines  : ~{est_total_lines:,.0f}")

    print(f"\n--- Audio Accessibility ---")
    print(f"  Tested: {audio_tested}  Accessible: {audio_accessible}")

    # Save
    with open("asmrone_results.json", "w") as f:
        json.dump({
            "total_works": total, "subtitle_works": sub_total,
            "lang_stats": dict(lang_stats), "samples": work_samples,
            "audio_accessible": audio_accessible, "audio_tested": audio_tested,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to asmrone_results.json")
    print(f"Samples saved to asmrone_samples/")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

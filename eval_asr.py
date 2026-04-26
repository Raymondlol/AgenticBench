#!/usr/bin/env python3
"""
ASR Evaluation Pipeline:
  1. Transcribe each audio file with mlx-whisper
  2. Parse VTT reference text
  3. Compute CER per file and per work
  4. Save incremental results after every file
  5. Generate summary report
"""

import json, re, os, sys, time, unicodedata
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR    = Path("/Users/raymond/WorkSpace/ASMR-Data/chinese_asr")
RESULTS_DIR = Path("/Users/raymond/WorkSpace/ASMR-Data/eval_results")
STATE_FILE  = RESULTS_DIR / "eval_state.json"
REPORT_FILE = RESULTS_DIR / "report.json"
MODEL_NAME  = "mlx-community/whisper-large-v3-turbo"  # fast + accurate
LANGUAGE    = "zh"
AUDIO_EXTS  = {".mp3", ".wav", ".flac", ".m4a"}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── VTT Parser ──────────────────────────────────────────────────────

def parse_vtt(path):
    """Parse WebVTT file, return list of {start, end, text}."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    segments = []
    # Match VTT cue blocks: timestamp line + text lines
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n(.+?)(?=\n\n|\n\d+\n|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        start_str, end_str, content = m.group(1), m.group(2), m.group(3)
        # Clean text
        content = re.sub(r'<[^>]+>', '', content)  # strip HTML tags
        content = content.strip()
        if content:
            segments.append({
                "start": ts_to_sec(start_str),
                "end": ts_to_sec(end_str),
                "text": content,
            })
    return segments


def ts_to_sec(ts):
    """Convert HH:MM:SS.mmm to seconds."""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def segments_to_text(segments):
    """Join segment texts into a single reference string."""
    return "".join(seg["text"] for seg in segments)


# ── Text normalization ──────────────────────────────────────────────

def normalize_text(text):
    """Normalize text for CER comparison."""
    # NFKC normalize (fullwidth → halfwidth, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Remove punctuation (Chinese and English)
    text = re.sub(r'[，。！？、；：""''「」『』（）【】《》…—～·\s]', '', text)
    text = re.sub(r'[,.!?;:\'"()\[\]{}<>\-_=+/\\|@#$%^&*`~]', '', text)
    # Remove digits (timestamps sometimes leak)
    # Keep Chinese chars, kana, latin letters
    text = re.sub(r'\d+', '', text)
    # Lowercase
    text = text.lower()
    # Remove remaining whitespace
    text = re.sub(r'\s+', '', text)
    return text


# ── CER calculation ─────────────────────────────────────────────────

def compute_cer(hypothesis, reference):
    """Compute Character Error Rate using jiwer."""
    from jiwer import cer
    # jiwer's cer works at word level by default; for Chinese we need char level
    # Trick: insert spaces between every character
    ref_chars = " ".join(list(reference))
    hyp_chars = " ".join(list(hypothesis))

    if not ref_chars.strip():
        return 1.0 if hyp_chars.strip() else 0.0

    return cer(ref_chars, hyp_chars)


# ── State management ────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed_files": {}, "model": MODEL_NAME}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ── Transcription ───────────────────────────────────────────────────

_model_loaded = False

def transcribe(audio_path):
    """Transcribe audio file using mlx-whisper."""
    global _model_loaded
    import mlx_whisper

    if not _model_loaded:
        print(f"  Loading model {MODEL_NAME}...", flush=True)
        _model_loaded = True

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=MODEL_NAME,
        language=LANGUAGE,
        verbose=False,
    )
    return result


# ── Main pipeline ───────────────────────────────────────────────────

def find_pairs(work_dir):
    """Find (audio, vtt) pairs in a work directory."""
    pairs = []
    for f in sorted(work_dir.iterdir()):
        if f.suffix.lower() in AUDIO_EXTS:
            # Find matching VTT: "track.mp3" → "track.mp3.vtt"
            vtt = Path(str(f) + ".vtt")
            if not vtt.exists():
                # Try stem match: "track.mp3" → "track.vtt"
                vtt = f.with_suffix(".vtt")
            if vtt.exists():
                pairs.append((f, vtt))
    return pairs


def eval_file(audio_path, vtt_path, state):
    """Evaluate a single audio file. Returns result dict."""
    key = str(audio_path)
    if key in state["completed_files"]:
        return state["completed_files"][key]

    t0 = time.time()

    # Parse reference
    segments = parse_vtt(vtt_path)
    ref_raw = segments_to_text(segments)
    ref_norm = normalize_text(ref_raw)

    if len(ref_norm) < 5:
        result = {
            "audio": audio_path.name,
            "vtt": vtt_path.name,
            "status": "skip_empty_ref",
            "ref_chars": len(ref_norm),
        }
        state["completed_files"][key] = result
        save_state(state)
        return result

    # Transcribe
    try:
        asr_result = transcribe(audio_path)
        hyp_raw = asr_result.get("text", "")
        hyp_segments = asr_result.get("segments", [])
    except Exception as e:
        result = {
            "audio": audio_path.name,
            "vtt": vtt_path.name,
            "status": "transcribe_error",
            "error": str(e),
        }
        state["completed_files"][key] = result
        save_state(state)
        return result

    hyp_norm = normalize_text(hyp_raw)
    elapsed = time.time() - t0

    # CER
    cer_val = compute_cer(hyp_norm, ref_norm)

    result = {
        "audio": audio_path.name,
        "vtt": vtt_path.name,
        "status": "ok",
        "ref_chars": len(ref_norm),
        "hyp_chars": len(hyp_norm),
        "cer": round(cer_val, 4),
        "ref_raw_preview": ref_raw[:200],
        "hyp_raw_preview": hyp_raw[:200],
        "ref_norm_preview": ref_norm[:100],
        "hyp_norm_preview": hyp_norm[:100],
        "duration_s": round(elapsed, 1),
        "asr_segments": len(hyp_segments),
        "ref_segments": len(segments),
    }

    # Save per-file result
    state["completed_files"][key] = result
    save_state(state)

    # Also save detailed per-work result
    per_work_dir = RESULTS_DIR / audio_path.parent.name
    per_work_dir.mkdir(exist_ok=True)
    detail = {
        **result,
        "ref_text": ref_raw,
        "hyp_text": hyp_raw,
        "ref_segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments],
        "hyp_segments": [{"start": s.get("start",0), "end": s.get("end",0), "text": s.get("text","")}
                        for s in hyp_segments],
    }
    detail_file = per_work_dir / (audio_path.stem + "_eval.json")
    detail_file.write_text(json.dumps(detail, ensure_ascii=False, indent=2))

    return result


def generate_report(state):
    """Generate summary report from all completed evaluations."""
    results = state["completed_files"]
    ok_results = [r for r in results.values() if r.get("status") == "ok"]

    if not ok_results:
        print("No successful evaluations!")
        return

    # Per-work aggregation
    work_stats = {}
    for key, r in results.items():
        work_id = Path(key).parent.name
        if work_id not in work_stats:
            work_stats[work_id] = {"files": [], "total_ref_chars": 0,
                                    "total_hyp_chars": 0, "weighted_cer_sum": 0}
        ws = work_stats[work_id]
        ws["files"].append(r)
        if r.get("status") == "ok":
            ws["total_ref_chars"] += r["ref_chars"]
            ws["total_hyp_chars"] += r["hyp_chars"]
            ws["weighted_cer_sum"] += r["cer"] * r["ref_chars"]

    # Overall stats
    total_chars = sum(r["ref_chars"] for r in ok_results)
    total_hyp = sum(r["hyp_chars"] for r in ok_results)
    weighted_cer = sum(r["cer"] * r["ref_chars"] for r in ok_results) / total_chars if total_chars else 0
    macro_cer = sum(r["cer"] for r in ok_results) / len(ok_results)
    cers = sorted([r["cer"] for r in ok_results])
    median_cer = cers[len(cers) // 2]
    total_time = sum(r.get("duration_s", 0) for r in ok_results)

    # Per-work summary
    work_summary = []
    for wid, ws in sorted(work_stats.items()):
        ok_files = [f for f in ws["files"] if f.get("status") == "ok"]
        if not ok_files:
            continue
        w_cer = ws["weighted_cer_sum"] / ws["total_ref_chars"] if ws["total_ref_chars"] else 0
        # Load work metadata
        meta_path = DATA_DIR / wid / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        work_summary.append({
            "work_id": wid,
            "source_id": meta.get("source_id", ""),
            "title": meta.get("title", "")[:60],
            "cv": meta.get("vas", []),
            "files_ok": len(ok_files),
            "files_total": len(ws["files"]),
            "ref_chars": ws["total_ref_chars"],
            "cer": round(w_cer, 4),
        })

    report = {
        "model": MODEL_NAME,
        "language": LANGUAGE,
        "eval_time": datetime.now().isoformat(),
        "summary": {
            "total_files": len(results),
            "ok_files": len(ok_results),
            "total_ref_chars": total_chars,
            "total_hyp_chars": total_hyp,
            "weighted_cer": round(weighted_cer, 4),
            "macro_avg_cer": round(macro_cer, 4),
            "median_cer": round(median_cer, 4),
            "min_cer": round(min(cers), 4),
            "max_cer": round(max(cers), 4),
            "p25_cer": round(cers[len(cers)//4], 4),
            "p75_cer": round(cers[3*len(cers)//4], 4),
            "total_eval_time_s": round(total_time, 1),
            "total_works": len(work_summary),
        },
        "per_work": sorted(work_summary, key=lambda w: w["cer"]),
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # Print report
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"ASR EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Model          : {MODEL_NAME}")
    print(f"Language       : {LANGUAGE}")
    print(f"Files evaluated: {s['ok_files']} / {s['total_files']}")
    print(f"Total ref chars: {s['total_ref_chars']:,}")
    print(f"Eval time      : {s['total_eval_time_s']:.0f}s ({s['total_eval_time_s']/60:.1f}min)")
    print(f"\n--- CER Summary ---")
    print(f"  Weighted CER   : {s['weighted_cer']:.2%}")
    print(f"  Macro avg CER  : {s['macro_avg_cer']:.2%}")
    print(f"  Median CER     : {s['median_cer']:.2%}")
    print(f"  Min / Max      : {s['min_cer']:.2%} / {s['max_cer']:.2%}")
    print(f"  P25 / P75      : {s['p25_cer']:.2%} / {s['p75_cer']:.2%}")

    print(f"\n--- Per Work (sorted by CER) ---")
    for w in report["per_work"]:
        cv = ", ".join(w["cv"][:2]) if w["cv"] else "N/A"
        print(f"  CER={w['cer']:.2%}  chars={w['ref_chars']:>5}  "
              f"files={w['files_ok']}  CV=[{cv}]  {w['title'][:40]}")

    # CER distribution histogram
    print(f"\n--- CER Distribution ---")
    buckets = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, float('inf')]
    for i in range(len(buckets)-1):
        lo, hi = buckets[i], buckets[i+1]
        count = sum(1 for c in cers if lo <= c < hi)
        bar = "█" * count
        label = f"{lo:.0%}-{hi:.0%}" if hi != float('inf') else f"{lo:.0%}+"
        print(f"  {label:>8s} | {bar} {count}")

    print(f"\nFull report: {REPORT_FILE}")
    return report


def main():
    state = load_state()
    completed_count = len([v for v in state["completed_files"].values() if v.get("status") == "ok"])
    print(f"Resuming: {completed_count} files already done\n")

    # Enumerate all work dirs
    work_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Works: {len(work_dirs)}")

    total_pairs = 0
    all_pairs = []
    for wd in work_dirs:
        pairs = find_pairs(wd)
        total_pairs += len(pairs)
        all_pairs.extend(pairs)

    print(f"Audio+VTT pairs: {total_pairs}")
    remaining = sum(1 for a, v in all_pairs if str(a) not in state["completed_files"])
    print(f"Remaining: {remaining}\n")

    # Process
    done = 0
    for work_dir in work_dirs:
        pairs = find_pairs(work_dir)
        if not pairs:
            continue

        # Load work metadata for display
        meta_path = work_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        title = meta.get("title", "")[:50]
        wid = work_dir.name

        # Check if all files in this work are done
        all_done = all(str(a) in state["completed_files"] for a, v in pairs)
        if all_done:
            continue

        print(f"[{wid}] {title}", flush=True)

        for audio, vtt in pairs:
            if str(audio) in state["completed_files"]:
                r = state["completed_files"][str(audio)]
                status = r.get("status", "?")
                cer_str = f"CER={r['cer']:.2%}" if status == "ok" else status
                print(f"  ✓ {audio.name[:50]}  ({cer_str}) [cached]", flush=True)
                continue

            print(f"  → {audio.name[:50]}...", end="", flush=True)
            r = eval_file(audio, vtt, state)
            done += 1

            if r["status"] == "ok":
                print(f"  CER={r['cer']:.2%}  ({r['ref_chars']}chars, {r['duration_s']:.0f}s)", flush=True)
            else:
                print(f"  [{r['status']}]", flush=True)

    # Generate report
    generate_report(state)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Segment-level ASR Evaluation Pipeline:
  1. Parse VTT subtitles for per-segment timestamps + reference text
  2. Slice audio by subtitle timestamps (no VAD needed)
  3. Transcribe each segment individually with mlx-whisper
  4. Compute per-segment CER + decompose into S/D/I
  5. Aggregate per-file and per-work statistics
  6. Compare with baseline (whole-file) and VAD results
"""

import json, re, os, sys, time, unicodedata
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR     = Path("/Users/raymond/WorkSpace/ASMR-Data/chinese_asr")
RESULTS_DIR  = Path("/Users/raymond/WorkSpace/ASMR-Data/eval_results_seg")
BASELINE_DIR = Path("/Users/raymond/WorkSpace/ASMR-Data/eval_results")
VAD_DIR      = Path("/Users/raymond/WorkSpace/ASMR-Data/eval_results_vad")
STATE_FILE   = RESULTS_DIR / "eval_state.json"
REPORT_FILE  = RESULTS_DIR / "report.json"

ASR_MODEL    = "mlx-community/whisper-large-v3-turbo"
LANGUAGE     = "zh"
AUDIO_EXTS   = {".mp3", ".wav", ".flac", ".m4a"}

# Segment padding: add small buffer around subtitle timestamps
# to avoid cutting off speech at boundaries
PAD_BEFORE_S = 0.15  # 150ms before segment start
PAD_AFTER_S  = 0.15  # 150ms after segment end
MIN_SEG_DUR  = 0.5   # skip segments shorter than 500ms

# Merge nearby segments to avoid whisper hallucination on very short clips
MERGE_GAP_S  = 1.0   # merge segments with gap < 1s
MIN_MERGED_DUR = 1.5  # ensure merged segments are at least 1.5s

# Repetition detection
MAX_HYP_RATIO = 5.0   # if hyp_chars > ref_chars * this, flag as hallucination

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── VTT Parser ──────────────────────────────────────────────────────

def parse_vtt(path):
    """Parse WebVTT file, return list of {start, end, text}."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    segments = []
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n(.+?)(?=\n\n|\n\d+\n|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        content = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        if content:
            segments.append({
                "start": _ts(m.group(1)),
                "end": _ts(m.group(2)),
                "text": content,
            })
    return segments

def _ts(s):
    p = s.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

def merge_short_segments(segments):
    """Merge adjacent subtitle segments that are too close together or too short.
    This prevents whisper hallucination on very short audio clips."""
    if not segments:
        return segments

    merged = [dict(segments[0])]  # copy first segment

    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        prev_dur = prev["end"] - prev["start"]

        # Merge if: gap is small, or previous segment is too short
        if gap < MERGE_GAP_S or prev_dur < MIN_MERGED_DUR:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"] + seg["text"]
        else:
            merged.append(dict(seg))

    # Final pass: merge any remaining too-short segments with neighbors
    if len(merged) > 1:
        final = [merged[0]]
        for seg in merged[1:]:
            if (seg["end"] - seg["start"]) < MIN_MERGED_DUR:
                final[-1]["end"] = seg["end"]
                final[-1]["text"] = final[-1]["text"] + seg["text"]
            else:
                final.append(seg)
        merged = final

    return merged

# ── Text normalization ──────────────────────────────────────────────

def normalize(text):
    """Normalize text for CER comparison."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[，。！？、；：""''「」『』（）【】《》…—～·\s]', '', text)
    text = re.sub(r'[,.!?;:\'\"()\[\]{}<>\-_=+/\\|@#$%^&*`~]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', '', text)
    return text

# ── Edit distance with S/D/I decomposition ─────────────────────────

def edit_distance_sdi(hyp, ref):
    """
    Compute character-level edit distance and return (S, D, I, C).
      S = substitutions (ref char replaced by different hyp char)
      D = deletions     (ref char missing in hyp)
      I = insertions    (hyp char not in ref)
      C = correct       (matching chars)
    CER = (S + D + I) / len(ref)
    """
    n = len(ref)
    m = len(hyp)

    # dp[i][j] = (cost, S, D, I, C) to align ref[:i] with hyp[:j]
    dp = [[(0, 0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    # Base cases
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0, 0)  # delete all ref chars
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j, 0)  # insert all hyp chars

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                # Match
                cost, s, d, ins, c = dp[i-1][j-1]
                dp[i][j] = (cost, s, d, ins, c + 1)
            else:
                # Substitution
                sub_cost, ss, sd, si, sc = dp[i-1][j-1]
                sub = (sub_cost + 1, ss + 1, sd, si, sc)

                # Deletion (ref char dropped)
                del_cost, ds, dd, di, dc = dp[i-1][j]
                delete = (del_cost + 1, ds, dd + 1, di, dc)

                # Insertion (extra hyp char)
                ins_cost, is_, id_, ii, ic = dp[i][j-1]
                insert = (ins_cost + 1, is_, id_, ii + 1, ic)

                dp[i][j] = min(sub, delete, insert, key=lambda x: x[0])

    cost, s, d, i, c = dp[n][m]
    return {"S": s, "D": d, "I": i, "C": c, "total_errors": s + d + i, "ref_len": n}

# ── Audio loading ──────────────────────────────────────────────────

_audio_cache = {}

def load_audio(audio_path, sr=16000):
    """Load audio file, return (samples_np, sample_rate)."""
    key = str(audio_path)
    if key in _audio_cache:
        return _audio_cache[key]

    import librosa
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    _audio_cache[key] = audio

    # Keep cache small: only hold 1 file at a time
    if len(_audio_cache) > 1:
        oldest = next(iter(_audio_cache))
        if oldest != key:
            del _audio_cache[oldest]

    return audio

def slice_audio(audio_np, start_s, end_s, sr=16000):
    """Slice audio array by time range with padding."""
    total_dur = len(audio_np) / sr
    start = max(0, start_s - PAD_BEFORE_S)
    end = min(total_dur, end_s + PAD_AFTER_S)
    return audio_np[int(start * sr):int(end * sr)]

# ── ASR ─────────────────────────────────────────────────────────────

_asr_loaded = False

def detect_repetition(text, min_pattern=1, max_pattern=20):
    """Detect if text is dominated by a repeating pattern (whisper loop).
    Returns (is_repetitive, cleaned_text)."""
    if len(text) < 10:
        return False, text

    # Check for repeating patterns of various lengths
    for plen in range(min_pattern, min(max_pattern, len(text) // 3) + 1):
        pattern = text[:plen]
        if not pattern.strip():
            continue
        # Count how many times this pattern repeats from the start
        count = 0
        pos = 0
        while pos + plen <= len(text) and text[pos:pos+plen] == pattern:
            count += 1
            pos += plen
        if count >= 5 and pos >= len(text) * 0.6:
            # This is a repetition loop - return just one instance
            return True, pattern

    return False, text

def transcribe_segment(audio_np, sr=16000):
    """Transcribe a numpy audio segment with mlx-whisper."""
    global _asr_loaded
    import mlx_whisper
    import tempfile, soundfile as sf

    if not _asr_loaded:
        print(f"  Loading ASR model {ASR_MODEL}...", flush=True)
        _asr_loaded = True

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_np, sr)
        tmp_path = f.name

    try:
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=ASR_MODEL,
            language=LANGUAGE,
            verbose=False,
            condition_on_previous_text=False,  # reduce hallucination carryover
        )
        text = result.get("text", "")

        # Detect and clean repetition loops
        is_rep, cleaned = detect_repetition(normalize(text))
        if is_rep:
            # Return empty - this segment is hallucinated
            return "", True

        return text, False
    finally:
        os.unlink(tmp_path)

# ── State management ───────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed_files": {}, "model": ASR_MODEL}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

# ── Eval per file ──────────────────────────────────────────────────

def eval_file(audio_path, vtt_path, state):
    key = str(audio_path)
    if key in state["completed_files"]:
        return state["completed_files"][key]

    t0 = time.time()

    # Parse reference segments
    ref_segs = parse_vtt(vtt_path)
    if not ref_segs:
        result = {"audio": audio_path.name, "status": "skip_no_segments"}
        state["completed_files"][key] = result
        save_state(state)
        return result

    # Load audio once
    try:
        audio_np = load_audio(audio_path)
    except Exception as e:
        result = {"audio": audio_path.name, "status": "audio_load_error", "error": str(e)}
        state["completed_files"][key] = result
        save_state(state)
        return result

    audio_dur = len(audio_np) / 16000

    # Merge short/close segments to avoid whisper hallucination
    merged_segs = merge_short_segments(ref_segs)
    orig_seg_count = len(ref_segs)

    # Process each merged segment
    seg_results = []
    total_S, total_D, total_I, total_C = 0, 0, 0, 0
    total_ref_chars = 0
    total_hyp_chars = 0
    skipped = 0
    hallucinated = 0

    for i, seg in enumerate(merged_segs):
        ref_raw = seg["text"]
        ref_norm = normalize(ref_raw)

        # Skip segments with too little text
        if len(ref_norm) < 2:
            skipped += 1
            continue

        seg_dur = seg["end"] - seg["start"]
        if seg_dur < MIN_SEG_DUR:
            skipped += 1
            continue

        # Slice audio for this segment
        seg_audio = slice_audio(audio_np, seg["start"], seg["end"])

        if len(seg_audio) < 1600:  # < 100ms at 16kHz
            skipped += 1
            continue

        # ASR
        try:
            hyp_raw, is_hallucinated = transcribe_segment(seg_audio)
        except Exception as e:
            seg_results.append({
                "idx": i, "status": "asr_error",
                "start": seg["start"], "end": seg["end"],
                "ref": ref_raw, "error": str(e),
            })
            continue

        hyp_norm = normalize(hyp_raw)

        # Check for hallucination: hyp massively longer than ref
        if is_hallucinated or (len(ref_norm) > 0 and len(hyp_norm) > len(ref_norm) * MAX_HYP_RATIO):
            hallucinated += 1
            # Treat as all-deletion (ASR produced garbage, count ref as missed)
            seg_results.append({
                "idx": i,
                "status": "hallucinated",
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "dur": round(seg_dur, 3),
                "ref": ref_raw,
                "hyp": hyp_raw[:50] + "..." if len(hyp_raw) > 50 else hyp_raw,
                "ref_norm": ref_norm,
                "ref_chars": len(ref_norm),
                "hyp_chars": len(hyp_norm),
            })
            # Count as deletions for S/D/I stats
            total_D += len(ref_norm)
            total_ref_chars += len(ref_norm)
            continue

        # Edit distance with S/D/I
        sdi = edit_distance_sdi(hyp_norm, ref_norm)
        seg_cer = sdi["total_errors"] / sdi["ref_len"] if sdi["ref_len"] > 0 else 0.0

        total_S += sdi["S"]
        total_D += sdi["D"]
        total_I += sdi["I"]
        total_C += sdi["C"]
        total_ref_chars += len(ref_norm)
        total_hyp_chars += len(hyp_norm)

        seg_results.append({
            "idx": i,
            "status": "ok",
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "dur": round(seg_dur, 3),
            "ref": ref_raw,
            "hyp": hyp_raw,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "ref_chars": len(ref_norm),
            "hyp_chars": len(hyp_norm),
            "cer": round(seg_cer, 4),
            "S": sdi["S"],
            "D": sdi["D"],
            "I": sdi["I"],
            "C": sdi["C"],
        })

    elapsed = time.time() - t0

    ok_segs = [s for s in seg_results if s.get("status") == "ok"]
    hal_segs = [s for s in seg_results if s.get("status") == "hallucinated"]

    if not ok_segs and not hal_segs:
        result = {"audio": audio_path.name, "status": "skip_all_failed",
                  "total_segments": orig_seg_count, "merged_segments": len(merged_segs),
                  "skipped": skipped}
        state["completed_files"][key] = result
        save_state(state)
        return result

    # File-level aggregation
    file_cer = (total_S + total_D + total_I) / total_ref_chars if total_ref_chars > 0 else 0.0
    seg_cers = [s["cer"] for s in ok_segs] if ok_segs else [1.0]

    # Speech coverage: total subtitle time / audio duration
    subtitle_time = sum(s["end"] - s["start"] for s in ref_segs)
    speech_coverage = subtitle_time / audio_dur if audio_dur > 0 else 0

    result = {
        "audio": audio_path.name,
        "vtt": vtt_path.name,
        "status": "ok",
        # CER
        "cer": round(file_cer, 4),
        "seg_cer_mean": round(sum(seg_cers) / len(seg_cers), 4),
        "seg_cer_median": round(sorted(seg_cers)[len(seg_cers) // 2], 4),
        "seg_cer_min": round(min(seg_cers), 4),
        "seg_cer_max": round(max(seg_cers), 4),
        # S/D/I
        "S": total_S,
        "D": total_D,
        "I": total_I,
        "C": total_C,
        "S_rate": round(total_S / total_ref_chars, 4) if total_ref_chars else 0,
        "D_rate": round(total_D / total_ref_chars, 4) if total_ref_chars else 0,
        "I_rate": round(total_I / total_ref_chars, 4) if total_ref_chars else 0,
        # Counts
        "ref_chars": total_ref_chars,
        "hyp_chars": total_hyp_chars,
        "orig_segments": orig_seg_count,
        "merged_segments": len(merged_segs),
        "eval_segments": len(ok_segs),
        "hallucinated_segments": hallucinated,
        "skipped_segments": skipped,
        # Timing
        "audio_dur": round(audio_dur, 1),
        "subtitle_time": round(subtitle_time, 1),
        "speech_coverage": round(speech_coverage, 4),
        "duration_s": round(elapsed, 1),
    }

    # Save detailed per-file with all segment results
    per_work = RESULTS_DIR / audio_path.parent.name
    per_work.mkdir(exist_ok=True)
    detail = {
        **result,
        "segments": seg_results,
    }
    (per_work / (audio_path.stem + "_eval.json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2))

    state["completed_files"][key] = result
    save_state(state)
    return result

# ── Report ─────────────────────────────────────────────────────────

def load_comparison_data(state_path):
    """Load CER data from a previous eval for comparison."""
    if not state_path.exists():
        return {}
    st = json.loads(state_path.read_text())
    out = {}
    for key, r in st.get("completed_files", {}).items():
        if r.get("status") == "ok":
            out[r["audio"]] = r["cer"]
    return out

def generate_report(state):
    results = state["completed_files"]
    ok = [r for r in results.values() if r.get("status") == "ok"]

    if not ok:
        print("No successful evaluations!")
        return

    # ── Aggregate ──
    total_ref = sum(r["ref_chars"] for r in ok)
    total_S = sum(r["S"] for r in ok)
    total_D = sum(r["D"] for r in ok)
    total_I = sum(r["I"] for r in ok)
    total_C = sum(r["C"] for r in ok)

    weighted_cer = (total_S + total_D + total_I) / total_ref if total_ref else 0
    cers = sorted([r["cer"] for r in ok])
    macro_cer = sum(cers) / len(cers)
    median_cer = cers[len(cers) // 2]
    total_time = sum(r.get("duration_s", 0) for r in ok)
    avg_coverage = sum(r.get("speech_coverage", 0) for r in ok) / len(ok)

    # ── Per-work ──
    work_map = {}
    for key, r in results.items():
        wid = Path(key).parent.name
        work_map.setdefault(wid, []).append(r)

    work_summary = []
    for wid, files in sorted(work_map.items()):
        ok_f = [f for f in files if f.get("status") == "ok"]
        if not ok_f:
            continue
        chars = sum(f["ref_chars"] for f in ok_f)
        wS = sum(f["S"] for f in ok_f)
        wD = sum(f["D"] for f in ok_f)
        wI = sum(f["I"] for f in ok_f)
        wC = sum(f["C"] for f in ok_f)
        w_cer = (wS + wD + wI) / chars if chars else 0
        w_cov = sum(f.get("speech_coverage", 0) for f in ok_f) / len(ok_f)
        meta_path = DATA_DIR / wid / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        work_summary.append({
            "work_id": wid, "source_id": meta.get("source_id", ""),
            "title": meta.get("title", "")[:60],
            "cv": meta.get("vas", []),
            "files_ok": len(ok_f), "ref_chars": chars,
            "cer": round(w_cer, 4),
            "S": wS, "D": wD, "I": wI, "C": wC,
            "S_rate": round(wS / chars, 4) if chars else 0,
            "D_rate": round(wD / chars, 4) if chars else 0,
            "I_rate": round(wI / chars, 4) if chars else 0,
            "speech_coverage": round(w_cov, 4),
        })

    # ── Comparison data ──
    baseline_cers = load_comparison_data(BASELINE_DIR / "eval_state.json")
    vad_cers = load_comparison_data(VAD_DIR / "eval_state.json")

    # ── Build report ──
    report = {
        "model": ASR_MODEL, "language": LANGUAGE,
        "method": "segment-level (VTT timestamp slicing)",
        "params": {
            "pad_before_s": PAD_BEFORE_S,
            "pad_after_s": PAD_AFTER_S,
            "min_seg_dur": MIN_SEG_DUR,
        },
        "eval_time": datetime.now().isoformat(),
        "summary": {
            "total_files": len(results),
            "ok_files": len(ok),
            "total_ref_chars": total_ref,
            "total_segments_orig": sum(r.get("orig_segments", 0) for r in ok),
            "total_segments_merged": sum(r.get("merged_segments", 0) for r in ok),
            "total_segments_eval": sum(r["eval_segments"] for r in ok),
            "total_segments_hallucinated": sum(r.get("hallucinated_segments", 0) for r in ok),
            "total_segments_skip": sum(r["skipped_segments"] for r in ok),
            # CER
            "weighted_cer": round(weighted_cer, 4),
            "macro_avg_cer": round(macro_cer, 4),
            "median_cer": round(median_cer, 4),
            "min_cer": round(min(cers), 4),
            "max_cer": round(max(cers), 4),
            "p25_cer": round(cers[len(cers) // 4], 4),
            "p75_cer": round(cers[3 * len(cers) // 4], 4),
            # S/D/I global
            "total_S": total_S,
            "total_D": total_D,
            "total_I": total_I,
            "total_C": total_C,
            "S_rate": round(total_S / total_ref, 4) if total_ref else 0,
            "D_rate": round(total_D / total_ref, 4) if total_ref else 0,
            "I_rate": round(total_I / total_ref, 4) if total_ref else 0,
            "accuracy": round(total_C / total_ref, 4) if total_ref else 0,
            # Coverage
            "avg_speech_coverage": round(avg_coverage, 4),
            "total_eval_time_s": round(total_time, 1),
        },
        "per_work": sorted(work_summary, key=lambda w: w["cer"]),
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # ── Print ──
    s = report["summary"]
    print(f"\n{'='*64}")
    print(f"  SEGMENT-LEVEL ASR EVALUATION REPORT")
    print(f"{'='*64}")
    print(f"  Model       : {ASR_MODEL}")
    print(f"  Method      : VTT timestamp slicing → per-segment ASR")
    print(f"  Files       : {s['ok_files']} / {s['total_files']}")
    print(f"  Segments    : {s['total_segments_orig']} orig → {s['total_segments_merged']} merged → "
          f"{s['total_segments_eval']} eval + {s.get('total_segments_hallucinated',0)} halluc + {s['total_segments_skip']} skip")
    print(f"  Ref chars   : {s['total_ref_chars']:,}")
    print(f"  Eval time   : {s['total_eval_time_s']:.0f}s ({s['total_eval_time_s']/60:.1f}min)")

    print(f"\n  --- CER ---")
    print(f"  Weighted     : {s['weighted_cer']:.2%}")
    print(f"  Macro avg    : {s['macro_avg_cer']:.2%}")
    print(f"  Median       : {s['median_cer']:.2%}")
    print(f"  Min / Max    : {s['min_cer']:.2%} / {s['max_cer']:.2%}")
    print(f"  P25 / P75    : {s['p25_cer']:.2%} / {s['p75_cer']:.2%}")

    print(f"\n  --- Error Decomposition (global) ---")
    print(f"  Substitutions (S) : {total_S:>6}  ({s['S_rate']:.2%} of ref)")
    print(f"  Deletions     (D) : {total_D:>6}  ({s['D_rate']:.2%} of ref)")
    print(f"  Insertions    (I) : {total_I:>6}  ({s['I_rate']:.2%} of ref)")
    print(f"  Correct       (C) : {total_C:>6}  ({s['accuracy']:.2%} accuracy)")
    print(f"  ─────────────────────────────────")
    print(f"  CER = (S+D+I)/N   : {s['weighted_cer']:.2%}")

    # ── 3-way comparison ──
    seg_cers_by_name = {r["audio"]: r["cer"] for r in ok}

    def compare(name, other_cers):
        matched = [(seg_cers_by_name[a], other_cers[a])
                    for a in seg_cers_by_name if a in other_cers]
        if not matched:
            return
        seg_avg = sum(s for s, _ in matched) / len(matched)
        other_avg = sum(o for _, o in matched) / len(matched)
        better = sum(1 for s, o in matched if s < o)
        print(f"\n  --- vs {name} ({len(matched)} common files) ---")
        print(f"  {name:16s} avg CER : {other_avg:.2%}")
        print(f"  Segment-level  avg CER : {seg_avg:.2%}")
        diff = other_avg - seg_avg
        print(f"  Improvement            : {diff:+.2%} ({diff/other_avg*100:+.0f}% relative)" if other_avg else "")
        print(f"  Segment better         : {better}/{len(matched)}")

    compare("Baseline", baseline_cers)
    compare("VAD+ASR", vad_cers)

    # ── Per work ──
    print(f"\n  --- Per Work (sorted by CER) ---")
    for w in report["per_work"]:
        cv = ", ".join(w["cv"][:2]) if w["cv"] else "N/A"
        print(f"    CER={w['cer']:.2%}  S={w['S_rate']:.0%}/D={w['D_rate']:.0%}/I={w['I_rate']:.0%}"
              f"  cov={w['speech_coverage']:.0%}  CV=[{cv[:16]}]  {w['title'][:30]}")

    # ── CER distribution ──
    print(f"\n  --- CER Distribution ---")
    buckets = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, float('inf')]
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = sum(1 for c in cers if lo <= c < hi)
        bar = "█" * count
        label = f"{lo:.0%}-{hi:.0%}" if hi != float('inf') else f"{lo:.0%}+"
        print(f"    {label:>8s} | {bar} {count}")

    # ── S/D/I distribution across files ──
    print(f"\n  --- S/D/I Dominance per File ---")
    s_dom = sum(1 for r in ok if r["S"] >= r["D"] and r["S"] >= r["I"])
    d_dom = sum(1 for r in ok if r["D"] > r["S"] and r["D"] >= r["I"])
    i_dom = sum(1 for r in ok if r["I"] > r["S"] and r["I"] > r["D"])
    print(f"    S-dominant : {s_dom} files  (ASR hears wrong chars)")
    print(f"    D-dominant : {d_dom} files  (ASR misses chars)")
    print(f"    I-dominant : {i_dom} files  (ASR hallucinates extra chars)")

    print(f"\n  Report: {REPORT_FILE}")
    return report

# ── Main ───────────────────────────────────────────────────────────

def find_pairs(work_dir):
    pairs = []
    for f in sorted(work_dir.iterdir()):
        if f.suffix.lower() in AUDIO_EXTS:
            vtt = Path(str(f) + ".vtt")
            if not vtt.exists():
                vtt = f.with_suffix(".vtt")
            if vtt.exists():
                pairs.append((f, vtt))
    return pairs

def main():
    state = load_state()
    done_count = len([v for v in state["completed_files"].values() if v.get("status") == "ok"])
    print(f"Resuming: {done_count} files already done\n")

    work_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    all_pairs = []
    for wd in work_dirs:
        all_pairs.extend(find_pairs(wd))

    remaining = sum(1 for a, v in all_pairs if str(a) not in state["completed_files"])
    print(f"Works: {len(work_dirs)}  Pairs: {len(all_pairs)}  Remaining: {remaining}\n")

    for work_dir in work_dirs:
        pairs = find_pairs(work_dir)
        if not pairs:
            continue

        all_done = all(str(a) in state["completed_files"] for a, v in pairs)
        if all_done:
            continue

        meta_path = work_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        wid = work_dir.name
        print(f"[{wid}] {meta.get('title','')[:50]}", flush=True)

        for audio, vtt in pairs:
            if str(audio) in state["completed_files"]:
                r = state["completed_files"][str(audio)]
                if r.get("status") == "ok":
                    print(f"  . {audio.name[:40]}  CER={r['cer']:.2%} "
                          f"S={r['S_rate']:.0%}/D={r['D_rate']:.0%}/I={r['I_rate']:.0%} [cached]",
                          flush=True)
                else:
                    print(f"  . {audio.name[:40]}  [{r.get('status','?')}] [cached]", flush=True)
                continue

            print(f"  > {audio.name[:40]}...", end="", flush=True)
            r = eval_file(audio, vtt, state)

            if r["status"] == "ok":
                print(f"  CER={r['cer']:.2%}  S={r['S_rate']:.0%}/D={r['D_rate']:.0%}/I={r['I_rate']:.0%}"
                      f"  segs={r['eval_segments']}  ({r['ref_chars']}ch, {r['duration_s']:.0f}s)",
                      flush=True)
            else:
                print(f"  [{r['status']}]", flush=True)

    generate_report(state)

if __name__ == "__main__":
    main()

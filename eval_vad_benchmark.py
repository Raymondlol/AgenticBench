#!/usr/bin/env python3
"""
VAD Benchmark: Evaluate Whisper-VAD-ASMR-onnx accuracy using VTT timestamps as ground truth.

Metrics:
  - Frame-level: Precision, Recall, F1 (at 20ms resolution)
  - Segment-level: IoU between predicted and reference segments
  - Boundary: onset/offset error distribution
  - Coverage: how much of VTT speech the VAD captures
"""

import json, sys, time, os, re
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/Users/raymond/WorkSpace/ASMRGAY/vad_model")
sys.stdout.reconfigure(line_buffering=True)

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR    = Path("/Users/raymond/WorkSpace/ASMR-Data/chinese_asr")
RESULTS_DIR = Path("/Users/raymond/WorkSpace/ASMR-Data/eval_vad_bench")
STATE_FILE  = RESULTS_DIR / "eval_state.json"
REPORT_FILE = RESULTS_DIR / "report.json"

VAD_MODEL   = "/Users/raymond/WorkSpace/ASMRGAY/vad_model/model.onnx"
AUDIO_EXTS  = {".mp3", ".wav", ".flac", ".m4a"}

# VAD params (same as previous eval)
VAD_THRESHOLD = 0.5
MIN_SPEECH_MS = 300
MIN_SILENCE_MS = 200
SPEECH_PAD_MS = 50

# Frame resolution
FRAME_MS = 20  # VAD outputs 20ms frames
SR = 16000

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── VTT Parser ──────────────────────────────────────────────────────

def parse_vtt(path):
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
            })
    return segments

def _ts(s):
    p = s.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

# ── Frame-level ground truth from VTT ──────────────────────────────

def vtt_to_frames(vtt_segs, total_dur_s):
    """Convert VTT segments to frame-level binary labels (20ms resolution)."""
    n_frames = int(total_dur_s * 1000 / FRAME_MS) + 1
    labels = np.zeros(n_frames, dtype=np.int8)
    for seg in vtt_segs:
        start_frame = int(seg["start"] * 1000 / FRAME_MS)
        end_frame = int(seg["end"] * 1000 / FRAME_MS)
        labels[start_frame:min(end_frame + 1, n_frames)] = 1
    return labels

def vad_to_frames(vad_segs, total_dur_s):
    """Convert VAD output segments to frame-level binary predictions."""
    n_frames = int(total_dur_s * 1000 / FRAME_MS) + 1
    preds = np.zeros(n_frames, dtype=np.int8)
    for seg in vad_segs:
        start_frame = int(seg["start"] * 1000 / FRAME_MS)
        end_frame = int(seg["end"] * 1000 / FRAME_MS)
        preds[start_frame:min(end_frame + 1, n_frames)] = 1
    return preds

# ── Metrics ────────────────────────────────────────────────────────

def frame_metrics(labels, preds):
    """Compute frame-level precision, recall, F1."""
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tn = int(np.sum((labels == 0) & (preds == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }

def segment_iou(ref_segs, pred_segs):
    """Compute segment-level IoU between reference and predicted segments."""
    if not ref_segs or not pred_segs:
        return {"mean_iou": 0, "matched": 0, "unmatched_ref": len(ref_segs), "unmatched_pred": len(pred_segs)}

    ious = []
    matched_ref = set()
    matched_pred = set()

    for i, ref in enumerate(ref_segs):
        best_iou = 0
        best_j = -1
        for j, pred in enumerate(pred_segs):
            # Compute overlap
            overlap_start = max(ref["start"], pred["start"])
            overlap_end = min(ref["end"], pred["end"])
            overlap = max(0, overlap_end - overlap_start)

            union_start = min(ref["start"], pred["start"])
            union_end = max(ref["end"], pred["end"])
            union = union_end - union_start

            iou = overlap / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou > 0.1:  # minimum IoU threshold to count as match
            ious.append(best_iou)
            matched_ref.add(i)
            matched_pred.add(best_j)

    return {
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0,
        "median_iou": round(float(np.median(ious)), 4) if ious else 0,
        "matched": len(ious),
        "unmatched_ref": len(ref_segs) - len(matched_ref),
        "unmatched_pred": len(pred_segs) - len(matched_pred),
        "total_ref": len(ref_segs),
        "total_pred": len(pred_segs),
    }

def boundary_errors(ref_segs, pred_segs, tolerance_s=0.5):
    """Compute onset/offset boundary errors between matched segments."""
    onset_errors = []
    offset_errors = []

    for ref in ref_segs:
        # Find closest pred segment by onset
        best_onset_err = float('inf')
        best_offset_err = float('inf')
        for pred in pred_segs:
            onset_err = abs(pred["start"] - ref["start"])
            offset_err = abs(pred["end"] - ref["end"])
            # Match by overlap
            overlap = max(0, min(ref["end"], pred["end"]) - max(ref["start"], pred["start"]))
            if overlap > 0:
                if onset_err < best_onset_err:
                    best_onset_err = onset_err
                    best_offset_err = offset_err

        if best_onset_err < 10:  # sanity limit
            onset_errors.append(best_onset_err)
            offset_errors.append(best_offset_err)

    if not onset_errors:
        return {"onset_mean": 0, "offset_mean": 0, "onset_median": 0, "offset_median": 0, "count": 0}

    return {
        "onset_mean": round(float(np.mean(onset_errors)), 4),
        "onset_median": round(float(np.median(onset_errors)), 4),
        "onset_p90": round(float(np.percentile(onset_errors, 90)), 4),
        "offset_mean": round(float(np.mean(offset_errors)), 4),
        "offset_median": round(float(np.median(offset_errors)), 4),
        "offset_p90": round(float(np.percentile(offset_errors, 90)), 4),
        "within_200ms": round(sum(1 for e in onset_errors if e <= 0.2) / len(onset_errors), 4),
        "within_500ms": round(sum(1 for e in onset_errors if e <= 0.5) / len(onset_errors), 4),
        "count": len(onset_errors),
    }

# ── VAD ────────────────────────────────────────────────────────────

_vad = None

def get_vad():
    global _vad
    if _vad is None:
        from inference import WhisperVADOnnxWrapper
        _vad = WhisperVADOnnxWrapper(VAD_MODEL, num_threads=4)
    return _vad

def run_vad(audio_path):
    from inference import get_speech_timestamps, load_audio
    audio = load_audio(str(audio_path))
    model = get_vad()
    segs = get_speech_timestamps(
        audio, model,
        threshold=VAD_THRESHOLD,
        return_seconds=True,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )
    total_dur = len(audio) / SR
    return segs, total_dur

# ── State ──────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed_files": {}}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

# ── Eval per file ──────────────────────────────────────────────────

def eval_file(audio_path, vtt_path, state):
    key = str(audio_path)
    if key in state["completed_files"]:
        return state["completed_files"][key]

    t0 = time.time()

    # Parse VTT ground truth
    ref_segs = parse_vtt(vtt_path)
    if not ref_segs:
        result = {"audio": audio_path.name, "status": "skip_no_ref"}
        state["completed_files"][key] = result
        save_state(state)
        return result

    # Run VAD
    try:
        vad_segs, total_dur = run_vad(audio_path)
    except Exception as e:
        result = {"audio": audio_path.name, "status": "vad_error", "error": str(e)}
        state["completed_files"][key] = result
        save_state(state)
        return result

    elapsed = time.time() - t0

    # Frame-level comparison
    labels = vtt_to_frames(ref_segs, total_dur)
    preds = vad_to_frames(vad_segs, total_dur)

    # Ensure same length
    min_len = min(len(labels), len(preds))
    labels = labels[:min_len]
    preds = preds[:min_len]

    fm = frame_metrics(labels, preds)
    si = segment_iou(ref_segs, vad_segs)
    be = boundary_errors(ref_segs, vad_segs)

    # Coverage stats
    ref_speech_s = sum(s["end"] - s["start"] for s in ref_segs)
    vad_speech_s = sum(s["end"] - s["start"] for s in vad_segs)

    result = {
        "audio": audio_path.name,
        "status": "ok",
        "audio_dur": round(total_dur, 1),
        "ref_segments": len(ref_segs),
        "vad_segments": len(vad_segs),
        "ref_speech_s": round(ref_speech_s, 1),
        "vad_speech_s": round(vad_speech_s, 1),
        "ref_speech_ratio": round(ref_speech_s / total_dur, 4) if total_dur else 0,
        "vad_speech_ratio": round(vad_speech_s / total_dur, 4) if total_dur else 0,
        "frame": fm,
        "segment_iou": si,
        "boundary": be,
        "eval_time": round(elapsed, 1),
    }

    state["completed_files"][key] = result
    save_state(state)
    return result

# ── Report ─────────────────────────────────────────────────────────

def generate_report(state):
    ok = [r for r in state["completed_files"].values() if r.get("status") == "ok"]
    if not ok:
        print("No results!")
        return

    # Aggregate frame metrics
    total_tp = sum(r["frame"]["tp"] for r in ok)
    total_fp = sum(r["frame"]["fp"] for r in ok)
    total_fn = sum(r["frame"]["fn"] for r in ok)
    total_tn = sum(r["frame"]["tn"] for r in ok)

    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    g_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    g_f1 = 2 * g_prec * g_recall / (g_prec + g_recall) if (g_prec + g_recall) else 0
    g_acc = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)

    # Per-file F1 distribution
    f1s = sorted([r["frame"]["f1"] for r in ok])
    precs = [r["frame"]["precision"] for r in ok]
    recalls = [r["frame"]["recall"] for r in ok]

    # Aggregate segment IoU
    ious = [r["segment_iou"]["mean_iou"] for r in ok if r["segment_iou"]["mean_iou"] > 0]

    # Aggregate boundary
    all_onset_mean = [r["boundary"]["onset_mean"] for r in ok if r["boundary"]["count"] > 0]
    all_offset_mean = [r["boundary"]["offset_mean"] for r in ok if r["boundary"]["count"] > 0]
    all_within_200 = [r["boundary"]["within_200ms"] for r in ok if r["boundary"]["count"] > 0]
    all_within_500 = [r["boundary"]["within_500ms"] for r in ok if r["boundary"]["count"] > 0]

    # Speech ratio comparison
    ref_ratios = [r["ref_speech_ratio"] for r in ok]
    vad_ratios = [r["vad_speech_ratio"] for r in ok]

    # Per-work aggregation
    work_map = {}
    for key, r in state["completed_files"].items():
        if r.get("status") != "ok":
            continue
        wid = Path(key).parent.name
        work_map.setdefault(wid, []).append(r)

    work_summary = []
    for wid, files in sorted(work_map.items()):
        w_tp = sum(f["frame"]["tp"] for f in files)
        w_fp = sum(f["frame"]["fp"] for f in files)
        w_fn = sum(f["frame"]["fn"] for f in files)
        w_prec = w_tp / (w_tp + w_fp) if (w_tp + w_fp) else 0
        w_rec = w_tp / (w_tp + w_fn) if (w_tp + w_fn) else 0
        w_f1 = 2 * w_prec * w_rec / (w_prec + w_rec) if (w_prec + w_rec) else 0
        meta_path = DATA_DIR / wid / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        work_summary.append({
            "work_id": wid,
            "title": meta.get("title", "")[:50],
            "cv": meta.get("vas", []),
            "files": len(files),
            "f1": round(w_f1, 4),
            "precision": round(w_prec, 4),
            "recall": round(w_rec, 4),
        })

    report = {
        "model": "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx",
        "vad_params": {
            "threshold": VAD_THRESHOLD,
            "min_speech_ms": MIN_SPEECH_MS,
            "min_silence_ms": MIN_SILENCE_MS,
            "speech_pad_ms": SPEECH_PAD_MS,
        },
        "eval_time": datetime.now().isoformat(),
        "dataset": "asmr.one Chinese-subtitle ASMR (Japanese audio), 30 works",
        "ground_truth": "VTT subtitle timestamps (human-annotated)",
        "summary": {
            "total_files": len(ok),
            "total_frames": total_tp + total_fp + total_fn + total_tn,
            "total_hours": round(sum(r["audio_dur"] for r in ok) / 3600, 2),
            # Frame-level (micro-averaged)
            "frame_precision": round(g_prec, 4),
            "frame_recall": round(g_recall, 4),
            "frame_f1": round(g_f1, 4),
            "frame_accuracy": round(g_acc, 4),
            # Per-file F1 distribution
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_median": round(float(np.median(f1s)), 4),
            "f1_min": round(min(f1s), 4),
            "f1_max": round(max(f1s), 4),
            "f1_p25": round(float(np.percentile(f1s, 25)), 4),
            "f1_p75": round(float(np.percentile(f1s, 75)), 4),
            # Segment IoU
            "mean_segment_iou": round(float(np.mean(ious)), 4) if ious else 0,
            # Boundary
            "onset_mean_s": round(float(np.mean(all_onset_mean)), 4) if all_onset_mean else 0,
            "onset_median_s": round(float(np.median(all_onset_mean)), 4) if all_onset_mean else 0,
            "offset_mean_s": round(float(np.mean(all_offset_mean)), 4) if all_offset_mean else 0,
            "within_200ms": round(float(np.mean(all_within_200)), 4) if all_within_200 else 0,
            "within_500ms": round(float(np.mean(all_within_500)), 4) if all_within_500 else 0,
            # Speech ratio
            "ref_speech_ratio_mean": round(float(np.mean(ref_ratios)), 4),
            "vad_speech_ratio_mean": round(float(np.mean(vad_ratios)), 4),
        },
        "per_work": sorted(work_summary, key=lambda w: -w["f1"]),
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # Print
    s = report["summary"]
    print(f"\n{'='*64}")
    print(f"  VAD BENCHMARK REPORT")
    print(f"  Model: Whisper-VAD-ASMR-onnx")
    print(f"  Ground truth: VTT subtitle timestamps")
    print(f"{'='*64}")
    print(f"  Files: {s['total_files']}  |  Hours: {s['total_hours']}")
    print(f"  Frames: {s['total_frames']:,} ({FRAME_MS}ms each)")

    print(f"\n  --- Frame-level (micro-averaged) ---")
    print(f"  Precision : {s['frame_precision']:.2%}")
    print(f"  Recall    : {s['frame_recall']:.2%}")
    print(f"  F1        : {s['frame_f1']:.2%}")
    print(f"  Accuracy  : {s['frame_accuracy']:.2%}")

    print(f"\n  --- Per-file F1 distribution ---")
    print(f"  Mean={s['f1_mean']:.2%}  Median={s['f1_median']:.2%}  "
          f"Min={s['f1_min']:.2%}  Max={s['f1_max']:.2%}")
    print(f"  P25={s['f1_p25']:.2%}  P75={s['f1_p75']:.2%}")

    print(f"\n  --- Segment IoU ---")
    print(f"  Mean IoU : {s['mean_segment_iou']:.2%}")

    print(f"\n  --- Boundary Accuracy ---")
    print(f"  Onset  mean : {s['onset_mean_s']*1000:.0f}ms")
    print(f"  Offset mean : {s['offset_mean_s']*1000:.0f}ms")
    print(f"  Within 200ms: {s['within_200ms']:.0%}")
    print(f"  Within 500ms: {s['within_500ms']:.0%}")

    print(f"\n  --- Speech Ratio ---")
    print(f"  VTT (ground truth) : {s['ref_speech_ratio_mean']:.0%}")
    print(f"  VAD (predicted)    : {s['vad_speech_ratio_mean']:.0%}")

    print(f"\n  --- Per Work (by F1) ---")
    for w in report["per_work"]:
        cv = ", ".join(w["cv"][:1]) if w["cv"] else "N/A"
        print(f"    F1={w['f1']:.2%}  P={w['precision']:.2%}  R={w['recall']:.2%}  "
              f"CV=[{cv[:14]}]  {w['title'][:30]}")

    # F1 distribution histogram
    print(f"\n  --- F1 Distribution ---")
    buckets = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = sum(1 for f in f1s if lo <= f < hi)
        bar = "█" * count
        print(f"    {lo:.0%}-{hi:.0%}  | {bar} {count}")

    print(f"\n  Report: {REPORT_FILE}")

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
                    print(f"  . {audio.name[:40]}  F1={r['frame']['f1']:.2%} [cached]", flush=True)
                continue

            print(f"  > {audio.name[:40]}...", end="", flush=True)
            r = eval_file(audio, vtt, state)

            if r["status"] == "ok":
                fm = r["frame"]
                print(f"  F1={fm['f1']:.2%}  P={fm['precision']:.2%}  R={fm['recall']:.2%}  "
                      f"ref={r['ref_segments']}seg  vad={r['vad_segments']}seg  ({r['eval_time']:.0f}s)",
                      flush=True)
            else:
                print(f"  [{r['status']}]", flush=True)

    generate_report(state)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
VAD + ASR Evaluation Pipeline:
  1. VAD cuts speech segments (Whisper-VAD-ASMR-onnx)
  2. Whisper transcribes speech-only audio
  3. CER computed against VTT reference
  4. Incremental saves per file
  5. Comparison with baseline (no-VAD) results
"""

import json, re, os, sys, time, unicodedata
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/Users/raymond/WorkSpace/ASMRGAY/vad_model")
sys.stdout.reconfigure(line_buffering=True)

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR     = Path("/Users/raymond/WorkSpace/ASMR-Data/chinese_asr")
RESULTS_DIR  = Path("/Users/raymond/WorkSpace/ASMR-Data/eval_results_vad")
BASELINE_DIR = Path("/Users/raymond/WorkSpace/ASMR-Data/eval_results")
STATE_FILE   = RESULTS_DIR / "eval_state.json"
REPORT_FILE  = RESULTS_DIR / "report.json"

VAD_MODEL    = "/Users/raymond/WorkSpace/ASMRGAY/vad_model/model.onnx"
ASR_MODEL    = "mlx-community/whisper-large-v3-turbo"
LANGUAGE     = "zh"
AUDIO_EXTS   = {".mp3", ".wav", ".flac", ".m4a"}

# VAD params
VAD_THRESHOLD = 0.5
MIN_SPEECH_MS = 300
MIN_SILENCE_MS = 200
SPEECH_PAD_MS = 50

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
            segments.append({"start": _ts(m.group(1)), "end": _ts(m.group(2)), "text": content})
    return segments

def _ts(s):
    p = s.split(":")
    return int(p[0])*3600 + int(p[1])*60 + float(p[2])

def segments_to_text(segs):
    return "".join(s["text"] for s in segs)

# ── Text normalization ──────────────────────────────────────────────

def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[，。！？、；：""''「」『』（）【】《》…—～·\s]', '', text)
    text = re.sub(r'[,.!?;:\'\"()\[\]{}<>\-_=+/\\|@#$%^&*`~]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', '', text)
    return text

# ── CER ─────────────────────────────────────────────────────────────

def compute_cer(hyp, ref):
    from jiwer import cer
    ref_c = " ".join(list(ref))
    hyp_c = " ".join(list(hyp))
    if not ref_c.strip():
        return 1.0 if hyp_c.strip() else 0.0
    return cer(ref_c, hyp_c)

# ── State ───────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed_files": {}, "model": ASR_MODEL, "vad_model": VAD_MODEL}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

# ── VAD ─────────────────────────────────────────────────────────────

_vad = None

def get_vad():
    global _vad
    if _vad is None:
        from inference import WhisperVADOnnxWrapper
        _vad = WhisperVADOnnxWrapper(VAD_MODEL, num_threads=4)
    return _vad

def run_vad(audio_path):
    """Run VAD and return speech segments + speech-only audio."""
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

    total_dur = len(audio) / 16000
    total_speech = sum(s["end"] - s["start"] for s in segs)

    # Extract speech-only audio
    sr = 16000
    chunks = []
    for s in segs:
        start_sample = int(s["start"] * sr)
        end_sample = min(int(s["end"] * sr), len(audio))
        chunks.append(audio[start_sample:end_sample])

    if chunks:
        speech_audio = np.concatenate(chunks)
    else:
        speech_audio = np.array([], dtype=np.float32)

    return segs, speech_audio, total_dur, total_speech

# ── ASR ─────────────────────────────────────────────────────────────

_asr_loaded = False

def transcribe_audio(audio_np, sr=16000):
    """Transcribe numpy audio array with mlx-whisper."""
    global _asr_loaded
    import mlx_whisper
    import tempfile, soundfile as sf

    if not _asr_loaded:
        print(f"  Loading ASR model {ASR_MODEL}...", flush=True)
        _asr_loaded = True

    # mlx-whisper needs a file path, write temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_np, sr)
        tmp_path = f.name

    try:
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=ASR_MODEL,
            language=LANGUAGE,
            verbose=False,
        )
        return result
    finally:
        os.unlink(tmp_path)

# ── Eval per file ───────────────────────────────────────────────────

def eval_file(audio_path, vtt_path, state):
    key = str(audio_path)
    if key in state["completed_files"]:
        return state["completed_files"][key]

    t0 = time.time()

    # Parse reference
    ref_segs = parse_vtt(vtt_path)
    ref_raw = segments_to_text(ref_segs)
    ref_norm = normalize(ref_raw)

    if len(ref_norm) < 5:
        result = {"audio": audio_path.name, "status": "skip_empty_ref", "ref_chars": len(ref_norm)}
        state["completed_files"][key] = result
        save_state(state)
        return result

    # VAD
    try:
        vad_segs, speech_audio, total_dur, speech_dur = run_vad(audio_path)
    except Exception as e:
        result = {"audio": audio_path.name, "status": "vad_error", "error": str(e)}
        state["completed_files"][key] = result
        save_state(state)
        return result

    speech_ratio = speech_dur / total_dur if total_dur > 0 else 0

    if len(speech_audio) < 1600:  # < 0.1s
        result = {
            "audio": audio_path.name, "status": "skip_no_speech",
            "total_dur": round(total_dur, 1), "speech_dur": round(speech_dur, 1),
            "vad_segments": len(vad_segs),
        }
        state["completed_files"][key] = result
        save_state(state)
        return result

    # ASR on speech-only audio
    try:
        asr_result = transcribe_audio(speech_audio)
        hyp_raw = asr_result.get("text", "")
    except Exception as e:
        result = {"audio": audio_path.name, "status": "asr_error", "error": str(e)}
        state["completed_files"][key] = result
        save_state(state)
        return result

    hyp_norm = normalize(hyp_raw)
    elapsed = time.time() - t0

    cer_val = compute_cer(hyp_norm, ref_norm)

    result = {
        "audio": audio_path.name,
        "vtt": vtt_path.name,
        "status": "ok",
        "ref_chars": len(ref_norm),
        "hyp_chars": len(hyp_norm),
        "cer": round(cer_val, 4),
        "total_dur": round(total_dur, 1),
        "speech_dur": round(speech_dur, 1),
        "speech_ratio": round(speech_ratio, 4),
        "vad_segments": len(vad_segs),
        "asr_segments": len(asr_result.get("segments", [])),
        "ref_norm_preview": ref_norm[:100],
        "hyp_norm_preview": hyp_norm[:100],
        "duration_s": round(elapsed, 1),
    }

    # Save detailed per-file
    per_work = RESULTS_DIR / audio_path.parent.name
    per_work.mkdir(exist_ok=True)
    detail = {
        **result,
        "ref_text": ref_raw,
        "hyp_text": hyp_raw,
    }
    (per_work / (audio_path.stem + "_eval.json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2))

    state["completed_files"][key] = result
    save_state(state)
    return result

# ── Report ──────────────────────────────────────────────────────────

def generate_report(state):
    results = state["completed_files"]
    ok = [r for r in results.values() if r.get("status") == "ok"]

    if not ok:
        print("No successful evaluations!")
        return

    cers = sorted([r["cer"] for r in ok])
    total_chars = sum(r["ref_chars"] for r in ok)
    weighted_cer = sum(r["cer"] * r["ref_chars"] for r in ok) / total_chars if total_chars else 0
    macro_cer = sum(cers) / len(cers)
    median_cer = cers[len(cers) // 2]
    total_time = sum(r.get("duration_s", 0) for r in ok)
    avg_speech_ratio = sum(r.get("speech_ratio", 0) for r in ok) / len(ok)

    # Per-work
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
        w_cer = sum(f["cer"] * f["ref_chars"] for f in ok_f) / chars if chars else 0
        w_sr = sum(f.get("speech_ratio", 0) for f in ok_f) / len(ok_f)
        meta_path = DATA_DIR / wid / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        work_summary.append({
            "work_id": wid, "source_id": meta.get("source_id", ""),
            "title": meta.get("title", "")[:60],
            "cv": meta.get("vas", []),
            "files_ok": len(ok_f), "ref_chars": chars,
            "cer": round(w_cer, 4), "speech_ratio": round(w_sr, 4),
        })

    # Load baseline for comparison
    baseline_state = BASELINE_DIR / "eval_state.json"
    baseline_cers = {}
    if baseline_state.exists():
        bs = json.loads(baseline_state.read_text())
        for key, r in bs.get("completed_files", {}).items():
            if r.get("status") == "ok":
                baseline_cers[r["audio"]] = r["cer"]

    report = {
        "model": ASR_MODEL, "vad_model": VAD_MODEL, "language": LANGUAGE,
        "vad_params": {
            "threshold": VAD_THRESHOLD, "min_speech_ms": MIN_SPEECH_MS,
            "min_silence_ms": MIN_SILENCE_MS, "speech_pad_ms": SPEECH_PAD_MS,
        },
        "eval_time": datetime.now().isoformat(),
        "summary": {
            "total_files": len(results),
            "ok_files": len(ok),
            "total_ref_chars": total_chars,
            "weighted_cer": round(weighted_cer, 4),
            "macro_avg_cer": round(macro_cer, 4),
            "median_cer": round(median_cer, 4),
            "min_cer": round(min(cers), 4),
            "max_cer": round(max(cers), 4),
            "p25_cer": round(cers[len(cers)//4], 4),
            "p75_cer": round(cers[3*len(cers)//4], 4),
            "avg_speech_ratio": round(avg_speech_ratio, 4),
            "total_eval_time_s": round(total_time, 1),
        },
        "per_work": sorted(work_summary, key=lambda w: w["cer"]),
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # Print
    s = report["summary"]
    print(f"\n{'='*64}")
    print(f"  VAD + ASR EVALUATION REPORT")
    print(f"{'='*64}")
    print(f"  ASR model   : {ASR_MODEL}")
    print(f"  VAD model   : Whisper-VAD-ASMR-onnx")
    print(f"  Files eval  : {s['ok_files']} / {s['total_files']}")
    print(f"  Ref chars   : {s['total_ref_chars']:,}")
    print(f"  Avg speech% : {s['avg_speech_ratio']:.0%}")
    print(f"  Eval time   : {s['total_eval_time_s']:.0f}s ({s['total_eval_time_s']/60:.1f}min)")

    print(f"\n  --- CER (VAD+ASR) ---")
    print(f"  Weighted     : {s['weighted_cer']:.2%}")
    print(f"  Macro avg    : {s['macro_avg_cer']:.2%}")
    print(f"  Median       : {s['median_cer']:.2%}")
    print(f"  Min / Max    : {s['min_cer']:.2%} / {s['max_cer']:.2%}")
    print(f"  P25 / P75    : {s['p25_cer']:.2%} / {s['p75_cer']:.2%}")

    # Comparison with baseline
    if baseline_cers:
        matched = []
        for r in ok:
            bl = baseline_cers.get(r["audio"])
            if bl is not None:
                matched.append((r["cer"], bl))
        if matched:
            vad_avg = sum(v for v, _ in matched) / len(matched)
            bl_avg = sum(b for _, b in matched) / len(matched)
            improved = sum(1 for v, b in matched if v < b)
            print(f"\n  --- Baseline Comparison ({len(matched)} matched files) ---")
            print(f"  Baseline avg CER : {bl_avg:.2%}")
            print(f"  VAD+ASR avg CER  : {vad_avg:.2%}")
            print(f"  Improvement      : {bl_avg - vad_avg:.2%} ({(bl_avg-vad_avg)/bl_avg*100:.0f}% relative)")
            print(f"  Files improved   : {improved}/{len(matched)}")

    print(f"\n  --- Per Work (sorted by CER) ---")
    for w in report["per_work"]:
        cv = ", ".join(w["cv"][:2]) if w["cv"] else "N/A"
        print(f"    CER={w['cer']:.2%}  spch={w['speech_ratio']:.0%}  "
              f"chars={w['ref_chars']:>5}  CV=[{cv[:20]}]  {w['title'][:35]}")

    # Distribution
    print(f"\n  --- CER Distribution ---")
    buckets = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, float('inf')]
    for i in range(len(buckets)-1):
        lo, hi = buckets[i], buckets[i+1]
        count = sum(1 for c in cers if lo <= c < hi)
        bar = "#" * count
        label = f"{lo:.0%}-{hi:.0%}" if hi != float('inf') else f"{lo:.0%}+"
        print(f"    {label:>8s} | {bar} {count}")

    print(f"\n  Report: {REPORT_FILE}")
    return report

# ── Main ────────────────────────────────────────────────────────────

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
                cer_str = f"CER={r['cer']:.2%}" if r.get("status") == "ok" else r.get("status","?")
                print(f"  . {audio.name[:45]}  ({cer_str}) [cached]", flush=True)
                continue

            print(f"  > {audio.name[:45]}...", end="", flush=True)
            r = eval_file(audio, vtt, state)

            if r["status"] == "ok":
                print(f"  CER={r['cer']:.2%}  spch={r['speech_ratio']:.0%}  "
                      f"({r['ref_chars']}ch, {r['duration_s']:.0f}s)", flush=True)
            else:
                print(f"  [{r['status']}]", flush=True)

    generate_report(state)

if __name__ == "__main__":
    main()

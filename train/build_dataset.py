#!/usr/bin/env python3
"""
Build HuggingFace Arrow dataset from raw (audio, VTT) pairs.

Walks ASMR-Data/chinese_asr/{work_id}/ folders, parses each VTT for
subtitle timestamps, slices audio with librosa, and writes an Arrow
dataset with columns: (audio, zh_text, work_id, source_id, segment_idx,
start_s, end_s, dur_s).

Usage:
  python train/build_dataset.py [--config train/config.yaml]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

# Reuse VTT parser logic from eval_seg_asr.py (copied here for portability)
VTT_PATTERN = re.compile(
    r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n(.+?)(?=\n\n|\n\d+\n|\Z)',
    re.DOTALL,
)

# Annotation-only line filter: e.g. [喘息], (雨音), (息), [SE], etc.
ANNOTATION_ONLY = re.compile(r'^[\[（(【].*[\]）)】]$')

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a"}


def ts_to_sec(ts: str) -> float:
    p = ts.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])


def parse_vtt(path: Path) -> list[dict]:
    """Parse WebVTT, returning [{start, end, text}] list."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    segments = []
    for m in VTT_PATTERN.finditer(text):
        content = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        # Strip HTML/styling tags but keep text
        content = unicodedata.normalize("NFKC", content)
        if not content:
            continue
        # Drop annotation-only lines (e.g. "[喘息]")
        if ANNOTATION_ONLY.match(content):
            continue
        segments.append({
            "start": ts_to_sec(m.group(1)),
            "end": ts_to_sec(m.group(2)),
            "text": content,
        })
    return segments


def find_pairs(work_dir: Path) -> list[tuple[Path, Path]]:
    """Return list of (audio_path, vtt_path) pairs."""
    pairs = []
    for f in sorted(work_dir.iterdir()):
        if f.suffix.lower() not in AUDIO_EXTS:
            continue
        vtt = Path(str(f) + ".vtt")
        if not vtt.exists():
            vtt = f.with_suffix(".vtt")
        if vtt.exists():
            pairs.append((f, vtt))
    return pairs


def build_segments(
    work_dir: Path,
    pad_before: float,
    pad_after: float,
    min_dur: float,
    max_dur: float,
    sample_rate: int,
) -> list[dict]:
    """Yield segment dicts for one work, lazily loading audio per file."""
    import librosa

    metadata_path = work_dir / "metadata.json"
    meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    source_id = meta.get("source_id", work_dir.name)

    segments = []
    for audio_path, vtt_path in find_pairs(work_dir):
        try:
            audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        except Exception as e:
            print(f"  [audio_load_error] {audio_path.name}: {e}", flush=True)
            continue

        total_dur = len(audio) / sample_rate
        ref_segs = parse_vtt(vtt_path)

        for seg_idx, seg in enumerate(ref_segs):
            seg_dur = seg["end"] - seg["start"]
            # Clamp duration
            if seg_dur < 0.3:  # too short to be useful
                continue
            # If segment is too long, truncate to max_dur
            end_s = min(seg["end"], seg["start"] + max_dur)
            seg_dur = end_s - seg["start"]
            if seg_dur < min_dur:
                continue

            # Slice audio with padding
            start = max(0, seg["start"] - pad_before)
            end = min(total_dur, end_s + pad_after)
            samples = audio[int(start * sample_rate):int(end * sample_rate)]

            if len(samples) < min_dur * sample_rate:
                continue

            segments.append({
                "audio": {"array": samples, "sampling_rate": sample_rate},
                "zh_text": seg["text"],
                "work_id": work_dir.name,
                "source_id": source_id,
                "audio_file": audio_path.name,
                "segment_idx": seg_idx,
                "start_s": float(seg["start"]),
                "end_s": float(end_s),
                "dur_s": float(seg_dur),
            })

    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--data-root", default=None,
                        help="Override: path to ASMR-Data/chinese_asr/")
    parser.add_argument("--max-works", type=int, default=None,
                        help="Limit to first N works (for testing)")
    parser.add_argument("--shard-size", type=int, default=500,
                        help="Examples per Arrow shard")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seg_cfg = cfg["segmentation"]

    data_root = Path(args.data_root) if args.data_root else Path(
        "/Users/raymond/WorkSpace/ASMR-Data/chinese_asr")
    out_path = Path(cfg["dataset"]["local_path"])

    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    work_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()])
    if args.max_works:
        work_dirs = work_dirs[: args.max_works]

    print(f"Found {len(work_dirs)} work directories", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Segmentation: pad±({seg_cfg['pad_before_s']},{seg_cfg['pad_after_s']})s, "
          f"dur in [{seg_cfg['min_dur_s']},{seg_cfg['max_dur_s']}]s, "
          f"sr={seg_cfg['sample_rate']}",
          flush=True)

    # Lazy import datasets to avoid hard-failing when not installed locally
    from datasets import Dataset, Audio, Features, Value

    out_path.mkdir(parents=True, exist_ok=True)

    all_segments = []
    total_dur_s = 0.0

    for i, wd in enumerate(work_dirs):
        print(f"[{i+1}/{len(work_dirs)}] {wd.name}...", end="", flush=True)
        try:
            segs = build_segments(
                wd,
                pad_before=seg_cfg["pad_before_s"],
                pad_after=seg_cfg["pad_after_s"],
                min_dur=seg_cfg["min_dur_s"],
                max_dur=seg_cfg["max_dur_s"],
                sample_rate=seg_cfg["sample_rate"],
            )
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue

        if not segs:
            print(" 0 segs", flush=True)
            continue

        all_segments.extend(segs)
        work_dur = sum(s["dur_s"] for s in segs)
        total_dur_s += work_dur
        print(f" {len(segs)} segs ({work_dur/60:.1f}min)", flush=True)

    if not all_segments:
        print("No segments built. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal: {len(all_segments)} segments, {total_dur_s/3600:.2f}h", flush=True)

    # Build HF Dataset
    features = Features({
        "audio": Audio(sampling_rate=seg_cfg["sample_rate"]),
        "zh_text": Value("string"),
        "work_id": Value("string"),
        "source_id": Value("string"),
        "audio_file": Value("string"),
        "segment_idx": Value("int32"),
        "start_s": Value("float32"),
        "end_s": Value("float32"),
        "dur_s": Value("float32"),
    })

    ds = Dataset.from_list(all_segments, features=features)
    print(f"Dataset built: {ds}", flush=True)

    # Save to disk
    print(f"Saving to {out_path}...", flush=True)
    ds.save_to_disk(str(out_path), max_shard_size="500MB")
    print(f"Done. Shards: {sorted(p.name for p in out_path.iterdir())}", flush=True)


if __name__ == "__main__":
    main()

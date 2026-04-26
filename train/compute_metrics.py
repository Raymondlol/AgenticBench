#!/usr/bin/env python3
"""
Shared metric computation: BLEU, chrF, BERTScore for ja->zh ST evaluation.

For Chinese targets, we use sacrebleu with tokenize="zh" for BLEU/chrF.
BERTScore uses lang="zh" with bert-base-chinese.

Also exports a text normalizer (ported from eval_seg_asr.py) used to
strip punctuation before metric computation, so models that produce
slightly different punctuation aren't penalized.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List


PUNCT_RE = re.compile(
    r'[，。！？、；：""\'\'「」『』（）【】《》…—～·\s]|'
    r'[,.!?;:\'\"()\[\]{}<>\-_=+/\\|@#$%^&*`~]'
)


def normalize_zh(text: str, strip_punct: bool = True) -> str:
    """NFKC normalize, optionally strip punctuation, lowercase, dedupe whitespace."""
    text = unicodedata.normalize("NFKC", text)
    if strip_punct:
        text = PUNCT_RE.sub("", text)
    text = text.lower()
    text = re.sub(r'\s+', " ", text).strip()
    return text


def compute_bleu(hyps: List[str], refs: List[str]) -> dict:
    """Corpus BLEU using sacrebleu with Chinese tokenization."""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
    return {
        "bleu": round(bleu.score, 4),
        "bleu_precisions": [round(p, 4) for p in bleu.precisions],
        "bleu_brevity_penalty": round(bleu.bp, 4),
        "bleu_sys_len": bleu.sys_len,
        "bleu_ref_len": bleu.ref_len,
    }


def compute_chrf(hyps: List[str], refs: List[str], word_order: int = 2) -> dict:
    """Corpus chrF (chrF++ when word_order=2)."""
    import sacrebleu
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=word_order)
    return {"chrf": round(chrf.score, 4)}


def compute_bertscore(hyps: List[str], refs: List[str], lang: str = "zh",
                       model_type: str | None = None,
                       batch_size: int = 32, device: str | None = None) -> dict:
    """BERTScore F1 in [0, 1]. Reports mean F1, P, R."""
    from bert_score import score as bs_score
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F = bs_score(
        hyps, refs,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    return {
        "bertscore_p": round(float(P.mean()), 4),
        "bertscore_r": round(float(R.mean()), 4),
        "bertscore_f1": round(float(F.mean()), 4),
    }


def compute_all_metrics(hyps: List[str], refs: List[str],
                         normalize: bool = True,
                         skip_bertscore: bool = False) -> dict:
    """Run all metrics. Pads hyps to match refs length if mismatched."""
    assert len(hyps) == len(refs), \
        f"Length mismatch: hyps={len(hyps)} refs={len(refs)}"

    if normalize:
        hyps_n = [normalize_zh(h) for h in hyps]
        refs_n = [normalize_zh(r) for r in refs]
    else:
        hyps_n = hyps
        refs_n = refs

    out = {}
    out.update(compute_bleu(hyps_n, refs_n))
    out.update(compute_chrf(hyps_n, refs_n))
    if not skip_bertscore:
        out.update(compute_bertscore(hyps_n, refs_n, lang="zh"))
    out["n_examples"] = len(hyps)
    return out


if __name__ == "__main__":
    # Quick smoke test
    hyps = ["你好世界", "今天天气真好", "啊嗯哥哥"]
    refs = ["你好，世界！", "今天天气很好。", "嗯，哥哥..."]
    m = compute_all_metrics(hyps, refs, skip_bertscore=True)
    print(m)

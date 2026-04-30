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


## ── ASMR-domain vocabulary hit-rate ────────────────────────────────
# Pairs that fan-translation conventions agree on (Japanese term -> Chinese term).
# We count: of all (ja_kw, zh_kw) pairs where ja_kw appears in the *reference*
# zh translation (i.e. the human translator chose this term), what fraction
# of the time the hypothesis also chose the same/equivalent CN term?
NSFW_VOCAB_PAIRS = [
    # body parts (ja-style fan translation -> common CN fan translation)
    ("おまんこ", ["小穴", "屄"]),
    ("まんこ",   ["小穴", "屄"]),
    ("ちんちん", ["小鸡鸡", "小肉棒"]),
    ("ちんぽ",   ["肉棒", "鸡巴", "鸡鸡"]),
    ("おちんちん", ["鸡鸡", "肉棒"]),
    # actions
    ("射精",     ["射精", "射"]),
    ("中出し",   ["内射", "中出"]),
    ("精液",     ["精液"]),
    # sounds typical of ASMR
    ("オホ",     ["哦吼", "嗯哦"]),
    ("舐め",     ["舔"]),
    # sensations
    ("気持ちいい", ["舒服", "好爽"]),
    ("イク",     ["要去了", "要射了", "高潮", "去了"]),
]


def nsfw_vocab_recall(hyps: List[str], refs: List[str]) -> dict:
    """For samples where the reference contains a known CN ASMR term,
    check how often the hypothesis preserves it. Returns precision/recall
    over the term universe."""
    matched = 0      # ref has term + hyp has equivalent term
    ref_has = 0      # ref has term (denominator for recall)
    hyp_has = 0      # hyp has term (denominator for precision)
    by_term = {}     # diagnostic
    for hyp, ref in zip(hyps, refs):
        for _, zh_alts in NSFW_VOCAB_PAIRS:
            ref_match = any(z in ref for z in zh_alts)
            hyp_match = any(z in hyp for z in zh_alts)
            if ref_match:
                ref_has += 1
                key = zh_alts[0]
                by_term.setdefault(key, {"ref_has": 0, "matched": 0})
                by_term[key]["ref_has"] += 1
                if hyp_match:
                    matched += 1
                    by_term[key]["matched"] += 1
            if hyp_match:
                hyp_has += 1
    recall = matched / ref_has if ref_has else 0.0
    precision = matched / hyp_has if hyp_has else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # Per-term recall for diagnostic
    by_term_summary = {
        k: {"recall": round(v["matched"] / v["ref_has"], 3) if v["ref_has"] else 0,
            "n_ref": v["ref_has"]}
        for k, v in by_term.items()
    }
    return {
        "nsfw_vocab_precision": round(precision, 4),
        "nsfw_vocab_recall": round(recall, 4),
        "nsfw_vocab_f1": round(f1, 4),
        "nsfw_vocab_n_ref": ref_has,
        "nsfw_vocab_n_hyp": hyp_has,
        "nsfw_vocab_per_term": by_term_summary,
    }


def compute_all_metrics(hyps: List[str], refs: List[str],
                         normalize: bool = True,
                         skip_bertscore: bool = True) -> dict:
    """Run pilot metrics: BLEU + chrF + ASMR vocab hit-rate.

    BERTScore is intentionally OFF by default — bert-base-chinese is
    trained on clean Wikipedia and doesn't differentiate ASMR vocabulary
    well, so the score has poor signal for our task. chrF is the
    primary metric (char-level, no tokenization needed for CN).
    """
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
    # NSFW vocab metric uses RAW (un-stripped) text so we don't lose terms
    # to punctuation removal. This is intentional.
    out.update(nsfw_vocab_recall(hyps, refs))

    if not skip_bertscore:
        try:
            out.update(compute_bertscore(hyps_n, refs_n, lang="zh"))
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[WARN] BERTScore failed: {err_msg}", flush=True)
            out["bertscore_error"] = err_msg

    out["n_examples"] = len(hyps)
    return out


if __name__ == "__main__":
    # Quick smoke test
    hyps = ["你好世界", "今天天气真好", "啊嗯哥哥"]
    refs = ["你好，世界！", "今天天气很好。", "嗯，哥哥..."]
    m = compute_all_metrics(hyps, refs, skip_bertscore=True)
    print(m)

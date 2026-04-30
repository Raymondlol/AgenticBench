# Agent Brief — Execute the Pilot Study

**This is your starting prompt.** Copy/paste the section below into your first Claude Code message, then let yourself run autonomously until the pilot completes.

---

## Your task

Execute the full pilot study end-to-end on this 4× RTX 4090 vast.ai instance. Goal: produce `out/phase4_compare.md` with a 6-way comparison table (chrF / BLEU / NSFW vocab recall) and push results to git.

**Read `CLAUDE.md` for full context. Do not skip it.**

## Acceptance criteria (when you can stop)

You are DONE when ALL of these are true:

1. ✅ `out/phase1_baselines/zeroshot_results.json` exists (system A)
2. ✅ `out/phase1_baselines/cascade_generic_results.json` exists (system B)
3. ✅ `out/phase1_baselines/cascade_anime_results.json` exists (system B', already done)
4. ✅ `out/phase1.5_sakura/cascade_sakura_results.json` exists (system B'')
5. ✅ `out/phase2a_sakura_lora/best/` contains a usable LoRA adapter (system C-Cascade trained)
6. ✅ `out/phase2b_sm4t_lora/best/` contains a usable LoRA adapter (system C-E2E trained)
7. ✅ `out/phase3_ft_eval/ft_cascade_results.json` exists (C-Cascade evaluated)
8. ✅ `out/phase3_ft_eval/ft_e2e_results.json` exists (C-E2E evaluated)
9. ✅ `out/phase4_compare.md` contains the 6-way comparison table
10. ✅ Everything in `out/` committed locally (don't `git push` — no creds; the human will pull manually)
11. ✅ First line of `out/STATUS.md` is `DONE`

## Suggested execution order

```
Step 1 — Re-run failed baselines (parallel on GPUs 0+1)
   - eval_e2e.py --zero-shot (system A)
   - eval_cascade.py --variant generic (system B)
   - on GPUs 2+3 simultaneously, start downloading Sakura models in background
     so they're ready when needed

Step 2 — Sakura B'' baseline (any free GPU)
   - eval_cascade.py --variant anime --use-sakura
   - Output to out/phase1.5_sakura/cascade_sakura_results.json

Step 3 — Train both LoRAs (probably sequential, both want 4 GPUs DDP)
   - Train QLoRA Sakura first (smaller, faster):
     accelerate launch train/train_qwen_lora.py --use-sakura
       --output-dir out/phase2a_sakura_lora
   - Then LoRA SeamlessM4T:
     accelerate launch train/train_lora.py
       (this writes to out/seamless-lora-pilot/best by default; symlink or
        copy to out/phase2b_sm4t_lora/best)

Step 4 — Eval both FT models (parallel on 2 GPUs)
   - eval_cascade.py --variant anime --use-sakura --mt-lora-ckpt ...
   - eval_e2e.py --ckpt out/phase2b_sm4t_lora/best

Step 5 — Generate phase4_compare.md by reading all six results JSONs
   and producing a markdown table. Also include 5 random sample translations
   from the test split for human review.

Step 6 — git add out/ ; git commit ; git push
```

## Optimization principles (THIS IS IMPORTANT)

The naive serial execution wastes GPUs. To minimize wall time:

- **While baselines (Step 1) run on GPUs 0+1**, use GPUs 2+3 to download Sakura models AND start running B'' (Step 2).
- **Don't wait for all Step 1+2 to finish before starting training.** Once any 4 GPUs are free, immediately start training the next LoRA.
- **After C-Cascade training finishes on 4 GPUs**, you can launch C-E2E training on 4 GPUs while doing nothing else. But you can ALSO run FT-Cascade eval on 1-2 GPUs while C-E2E trains on the others — DDP only needs 2-3 GPUs to scale 80%, so 1-2 free GPUs is fine for inference.
- **Re-check GPU utilization every few minutes.** If you see idle GPUs and pending work, schedule it.

## Constraints

- Total budget: pilot ~$10. Watch wall time. Vast.ai instance is $1.43/hr.
- If a single training step takes >10 minutes per step, abort that experiment and decrease batch size or sequence length.
- If a training run's loss is going UP for >100 steps, abort and try a lower LR.
- If OOM: reduce per_device_batch_size to 2, increase grad_accum_steps to compensate.

## Logging

Maintain these files in `out/`:
- `STATUS.md` — append phase START/END timestamps. First line = `RUNNING` until done, then `DONE`.
- `agent_decisions.md` — record every non-trivial decision (changed hyperparam, killed a job, switched approach). Format: `[ISO timestamp] decision: rationale`.
- After each phase succeeds, `git add out/ && git commit -m "..."`.
  Do NOT `git push` — there are no GitHub credentials on this machine.
  The human will pull manually via `scp` or by `git push` after SSHing in with their PAT.

## When to escalate

If you genuinely don't know what to do (e.g. all training runs diverge, or hardware misbehaves), write the question to `out/AGENT_QUESTION.md`, stop running new commands, and wait for the human.

## Final tip

The human is excited about this pilot. Make their morning easy: a complete phase4_compare.md with clean numbers and 5 sample translations is the goal. Don't fail silently — log everything.

---

When you've read this and `CLAUDE.md`, reply with a one-line plan summary and start executing.

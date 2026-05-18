---
name: cnn-auto-f1-variance-not-mean-lever
description: F1 (compressor uses last-step ckpt) buys variance compression, not peak height — iter-17 Q2b validation
status: closed
tags:
  - cnn-auto-push
outcome: 0.28-nat VMIM bound improvement (iter-5 batch48000 ckpt vs final) → +8.7% per_seed_min and 3.4x tighter std at iter-5 mean (within noise) — iter-17 Q2b confirms F1 buys variance, not peak height, on plain at 60k. Same pattern as cbs=256 (iter-4).
---

Filed 2026-05-18 ~21:52 UTC by Ralph iter-3 after iter-17 (Q2b) landed.

## What iter-17 tested

Q2b from `[[cnn-auto-compressor-last-not-best-ckpt]]`: rerun Stage B (NDE training + posterior sampling, no Phase A) with iter-5's **best-val** compressor checkpoint (`params_nd_compressor_batch48000.pkl`, test loss -12.722) instead of iter-5's **last-step** ckpt (`params_nd_compressor_batch60000.pkl`, test loss -12.440). The 0.282-nat VMIM bound improvement is information that should — by the VMIM theory — translate to a tighter posterior on theta.

Setup: iter-17/compressor/ symlinks the 5 ckpt files from iter-5; `run_arm.py --skip-compressor --total-steps 48000 --gpus 2,2,2 --seeds 41,42,43`. ~7 min wall.

## What landed (numbers)

| metric             | iter-5 (final ckpt) | iter-17 (best-val ckpt) | Δ            |
|--------------------|---------------------|--------------------------|--------------|
| FoM3 mean          | 18 567.67           | 18 456.53                | **−0.60 %**  |
| FoM3 std           | 1 604.25            | 469.76                   | **3.41× tighter** |
| FoM3 per_seed_min  | 16 368.47           | 17 792.96                | **+8.70 %**  |
| FoM3 pooled        | 12 894              | 12 279                   | -4.8%        |

Per-seed FoM3:
- s41: 20 150 → 18 761 (-7%)
- s42: 19 184 → 18 816 (-2%)
- s43: 16 368 → 17 793 (+9%)

The high seed (s41) came down; the low seed (s43) came up. The distribution **collapsed toward the mean** without lifting the mean itself.

## What this means

VMIM is a lower bound on `I(θ; s)`. A 0.28-nat improvement is real information added to `s`. But the way it cashes out in the posterior depends on the geometry:

- If the existing posterior **peak** is already well-constrained (the compressor's nonlinearity is fully exercised at the peak), the extra information goes into the **tails** — narrowing the credible interval far from the mode.
- Tails dominate the determinant when posteriors are not perfectly Gaussian. So tighter tails → smaller std → tighter per-seed scatter → smaller worst-seed variance (`per_seed_min` rises).
- The peak (which dominates `mean-of-seeds` because each seed's posterior peak is what FoM3 covariance is computed around) doesn't move.

**The same pattern showed in iter-4** (`[[cnn-auto-cbs256-stability]]`): compressor batch size 128 → 256 gave −3.5% mean, +2.4% per_seed_min, std 5× tighter. Now F1 (best-val ckpt) does the same. Two independent mechanisms produce the same signature: **mode-preserving variance compression**.

Working hypothesis: both interventions reduce the **optimizer-induced stochasticity of the compressor's representation** (better-converged compressor — larger effective batch, or better-selected step — produces a less-noisy `s`, which gives the NDE flow a sharper conditioning signal). The peak is information-bottlenecked by the compressor's representational capacity at cdim=16, but the tails are bottlenecked by optimizer noise.

## Why this is a useful negative

The audit (Ralph iter-2 A2) was right that F1 was a real artifact (kept iters are 0.27-0.34 nats below their best-val test loss). The audit's *consequence* prediction (mean FoM3 will jump) was wrong. The actual consequence (per-seed scatter compresses) is **still useful** — for the 240k promotion specifically, where the Guard floor at 11 000 (or constitution's headline 18 000) is per-seed. A best-val-tracking compressor pickle would buy:

- Free per-seed-min lift → easier Guard pass at 240k.
- Free std compression → fewer rerolls needed to confirm a result is real, not noise.
- No mean penalty.

So the F1 fix (add best-val tracking to `train_compressor_vmim`) is still worth doing — just deprioritized from "dominant lever, do it now" to "free polish, do it before 240k promotion".

## What does NOT translate

- **iter-15 (resnet50_gn) collapse magnitude**: the audit speculated F1 amplified iter-15's -52.8% delta. The new data says F1 doesn't move mean much, so the -52.8% is mostly Q2 (undertraining). Magnitude un-amplifying is small.
- **Q2 interpretation for iter-16**: iter-16 reads at step 120000 (last-step). If F1 doesn't move plain-arm mean at 60k, it likely doesn't move plain-arm mean at 120k either. So iter-16's mean improvement (if any) is "real Q2 effect" not "Q2 + F1". Cleaner orthogonalization than feared.
- **Cross-arm transfer**: this is a plain-only test. The resnet50_gn arm (iter-15) is so undertrained (-1.3 nats vs plain) that F1 there is mixed with Q2. Q2b on iter-15 batch51000 (the prepared but not-launched iter-19) would still be interesting but is now lower-priority than just running resnet50_gn at 120k+.

## Cross-references

- Parent: `[[cnn-auto-push-18-20-2026]]`
- Closed sibling: `[[cnn-auto-compressor-last-not-best-ckpt]]` (the original F1 finding, now resolved)
- Same pattern: `[[cnn-auto-cbs256-stability]]` (cbs=256 also gives variance compression without mean lift)
- Related: `[[cnn-auto-pooled-vs-mos]]` (per-seed posterior drift — F1 doesn't reduce drift, just tightens within each seed)

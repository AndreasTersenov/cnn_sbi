---
name: 'CNN-VMIM optimization: get the learned compressor to (at least) match L1'
status: active
tags:
    - experiment
    - sbi
    - cnn
    - vmim
    - nde
    - optimization
    - paper
created-at: 2026-06-13T11:45:00.000000000Z
outcome: 'OPEN (opened 2026-06-13). OBJECTIVE: make the CNN-VMIM compressor reach — and ideally exceed — the best analytical result (L1+product FoM3 2875, gate-C clean) on the de-leaked flat-local data, with CLEAN calibration, so paper message M1 ("the CNN does not underperform L1") is defended against the "your CNN is undertrained" referee attack. Current flat-local CNN: auto-only best 2620 / mean 2457 (TIE with L1-auto 2405), +product mean 2191 (BEHIND L1+product 2875; product channels HURT the CNN), seed-fragile (auto 2620/2364/2387) = optimization instability, not data scarcity. DATA IS NOT THE BOTTLENECK (checked): 323640 patch examples / 899 cosmologies / 360 patches per cosmo. LEVERS in priority order: (1) NDE flow on CNN summaries — Andreas suspects jaxili MAF is poor here, RealNVP did better in the 20deg analysis [highest-leverage, cheapest: retrain NDE only]; (2) CNN architecture/capacity (--compressor-arch resnet*, GN variants; watch over-capacity vs 899 cosmos); (3) VMIM companion flow (sbi_lens RealNVP, documented unstable); (4) convergence discipline (training curves, best-val ckpts, no last-step bug). Entry point: HANDOFF_CNN_OPTIMIZATION.md (repo root). SCOPE: CNN side ONLY — L1/BNT/joint-stat work stays on [[flatsky-cross-2026-06]]. Continues the CNN thread from [[definitive-l1-vs-cnn-10deg-2026-06]] and [[cnn-auto-push-18-20-2026]].'
---

## Primary metric
Per-seed-median FoM3 of the CNN-VMIM compressor on de-leaked flat-local data, reported with
σ(Ωm,σ8,w0) alongside (marginals-first). Bar: L1+product 2875 (gate-C clean). Calibration
(TARP+SBC) MANDATORY — uncalibrated FoM3 gains do not count (LANE_A_CONCLUSION.md).

## Done condition
Auto-close when the NDE-flow + architecture sweep is exhausted AND the best CNN is either
(a) >= 2875 calibrated [M1 = "CNN >= L1"] OR (b) plateaus below it across 3 consecutive
variants within +/-5% [M1 = "best-effort CNN ties/trails L1 = genuine practical
sub-optimality, NOT undertraining"]. Either outcome resolves M1 for the paper. Plateau
default N=3, X=5%.

## Guardrails
Vary ONE factor at a time; 3 seeds; rank by FoM3 NOT val-loss (val-loss unreliable across
architectures); best-val checkpoints (never last-step); SAME 9000-obs fiducial population +
SAME TARP/SBC gates as the L1 arms (apples-to-apples); GroupNorm on multi-channel input (BN
collapses); watch train/val gap (over-capacity risk at 899 cosmologies); GPU pool 0/1/2
(never 3), tenant-check before launch; do NOT chase "more sims" (data is ample); do NOT
re-do L1/BNT work (other fiber).

## Loop status (OPENED 2026-06-13 ~11:45 UTC)
Fiber created as the split-off CNN-optimization direction (Andreas's call: keep the L1/BNT/
analytical session separate). No work run yet. Entry point HANDOFF_CNN_OPTIMIZATION.md; first
prompt provided to Andreas. First planned move (cheapest, highest-leverage): on a FIXED set
of CNN-VMIM summaries (one good compressor seed), swap the NDE flow family (jaxili MAF vs
sbi_lens RealNVP vs alternatives), 3 NDE seeds each, FoM3 + GATE C — testing Andreas's lead
hypothesis that the flow, not the compressor, caps the CNN. THEN architecture sweep. Dataset
facts + baselines + file map all in the handoff. GPUs released by the sibling session.

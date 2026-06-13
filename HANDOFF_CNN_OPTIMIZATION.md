# HANDOFF — CNN-optimization session (start here)

**Date:** 2026-06-13. **For:** a NEW, separate Claude Code session dedicated to getting the
best possible CNN-VMIM result. **Scope split:** this session does ONLY the CNN side
(compressor architecture, the NDE flow on CNN summaries, training/convergence). The
analytical-summary-stats work (L1, BNT rescue = paper message M4) stays in the OTHER session
on the `flatsky-cross-2026-06` felt fiber. Do not relitigate L1/BNT here.
**Felt fiber:** `.felt/cnn-optimization-2026-06/cnn-optimization-2026-06.md` (read its top;
prepend a stanza per substantive event; commit the .md by path, NEVER `.felt/index.db`).
**Env:** conda `jaxili` (`/home/tersenov/anaconda3/envs/jaxili/bin/python`).
**GPUs:** pool 0/1/2 (GPU 3 NEVER); fresh `nvidia-smi` tenant check before every launch;
co-residency is allowed when Andreas says so, otherwise be polite. ≤50 CPU workers.

## 1. The objective (one primary metric, one bar)
**Primary metric:** pooled / per-seed-median FoM3 of the CNN-VMIM compressor on the
de-leaked flat-local data, reported with σ(Ωm,σ8,w0) alongside (marginals-first).
**Bar to beat:** L1+product **2875** (gate-C clean) — the current best analytical result.
The CNN must reach at least this, calibrated, to defend paper message M1 ("the CNN does not
underperform L1, given best-effort training"). Secondary: does it EXCEED L1 (as a truly
optimal compressor should)?
**Calibration is mandatory:** any FoM3 gain must pass GATE C (TARP+SBC) or it doesn't count
(see the A1 cautionary tale, LANE_A_CONCLUSION.md — a tighter-but-miscalibrated posterior is
fool's gold).

## 2. Where the CNN stands now (flat-local, the apples-to-apples data)
`results/exploratory/flatsky_cross_2026_06/cnn_phase/best_seed/per_seed.json`:
| arm | per-seed FoM3 | best | mean |
|---|---|---|---|
| CNN auto-only | 2620 / 2364 / 2387 | 2620 | 2457 |
| CNN +conv | 2418 / 1968 / 2491 | 2491 | 2292 |
| CNN +product | 2225 / 2331 / 2017 | 2331 | 2191 |
| CNN +both | 2475 / 2436 / 2205 | 2475 | 2372 |
Baselines: L1-auto 2405, **L1+product 2875**, l1-auto-only tie with CNN auto.
Reading: CNN TIES L1 on autos, is BEHIND on cross (product channels HURT the CNN), best
single arm 2620 < 2875, and is SEED-FRAGILE (big scatter) — a fingerprint of optimization
instability, not data scarcity.

## 3. Data is NOT the bottleneck (checked 2026-06-13)
Training cache: **323,640 patch examples / 899 distinct cosmologies / 360 patches per
cosmology** (180 patches × 2 perms 5–6); val 144,000 / 400 cosmologies; grid total 2500.
324k patch examples is ample for the compressor. So do NOT chase "more sims" — chase
architecture / flow / convergence. (More cosmologies via more perms or more of the 2500 grid
is a lever IF the NDE turns out cosmology-limited, but prove that first.)

## 4. What to optimize (Andreas's hypotheses, in priority order)
1. **The NDE flow on the CNN summaries.** Andreas's lead suspicion: the jaxili MAF is not
   great for CNN-VMIM summaries. In the 20° analysis a RealNVP-type NDE produced better
   results. → Compare NDE families (jaxili MAF vs sbi_lens RealNVP vs others) on the SAME
   fixed CNN summaries, with convergence checks. This is cheap (re-train the NDE only) and
   the highest-leverage first test.
2. **The CNN architecture / capacity.** `--compressor-arch {plain, resnet_small, resnet18,
   resnet34, resnet50, resnet50_gn}` in `npe_cnn_nbody_tomo.py`. On multi-channel input use
   GroupNorm variants (BN ResNets collapse — `project_resnet_bn_contamination`). Try larger /
   deeper; watch for the data-limit (`project_resnet50gn_120k_overfits` — resnet50_gn @120k
   overfit ~70k cosmos in an earlier campaign; here we have 899, so over-capacity is a real
   risk — watch train/val gap).
3. **The VMIM companion flow + objective.** The compressor is VMIM-trained with a companion
   flow (sbi_lens RealNVP, documented unstable — `project_maf_companion_not_bottleneck`
   found a beefier MAF companion WORSE; revisit only with convergence discipline).
4. **Convergence discipline (non-negotiable).** Inspect training curves for BOTH the
   compressor and the NDE; confirm val-loss plateau, no divergence, sane LR schedule;
   use best-val checkpoints (NOT last-step — `cnn-auto-compressor-last-not-best-ckpt` bug
   history). Seed-fragility (above) likely means runs aren't converging consistently.

## 5. Method discipline (the controlled comparison)
The whole point is a FAIR, converged CNN. So: fix the data split + obs set + GATE machinery,
vary ONE factor at a time (flow family, then arch, then companion), 3 seeds, report
per-seed-median FoM3 + marginals + GATE C. Use the SAME 9000-obs fiducial population and the
SAME TARP/SBC gates as the L1 arms so the cross-method comparison stays apples-to-apples.
Beware val-loss-as-FoM3-proxy across architectures (`feedback_val_loss_not_reliable_fom3_proxy`)
— rank by FoM3, not val loss.

## 6. Key files
- Compressor+NPE entrypoints: `npe_cnn_nbody_tomo.py` (production; sbi_lens RealNVP NDE),
  `npe_cnn_jaxili_nbody_tomo.py` (jaxili path; "exists but never used for production" —
  `project_nde_architecture_mismatch`). The flow-family test lives at the seam between these.
- Flat-local CNN driver / results: `results/exploratory/flatsky_cross_2026_06/cnn_phase/`
  (incl. `gate_c/`, `best_seed/per_seed.json`, multiseed dir).
- Flat-sky cross operators (shared): `flatsky_cross.py` (jax backend for the CNN input step).
- Dataset: TFDS `NbodyCosmogridDatasetTomo/...` / cross TFDS
  `nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`.
- Gate machinery: `tarp_stratified_val.py`, `run_tarp_coverage.py`, SBC-from-dumps pattern
  (see `run_joint_gate_c.py` / `run_bnt_gate_c.py` for the adapt-pattern).
- Population sweep (for FoM3): `population_sweep_flatsky.py` (L1 path) — for the CNN, the
  cnn_phase driver is the analog; reuse its sampling.

## 7. Context docs (read these, then stop reading)
- `PAPER_MESSAGES.md` (campaign dir) — message M1 is what this session defends.
- `FLATSKY_CNN_RESULT.md` (root) — pillar-1 CNN record (zero systematic cross gain, etc.).
- `LANE_A_CONCLUSION.md` — the DPI / "FoM3 differences are partly estimator quality" lesson;
  the calibration-is-mandatory cautionary tale.
- Memories: `project_nde_architecture_mismatch`, `project_maf_companion_not_bottleneck`,
  `project_resnet_bn_contamination`, `project_resnet50gn_120k_overfits`,
  `feedback_val_loss_not_reliable_fom3_proxy`, `feedback_benchmark_dont_assume`.

## 8. Done condition (declare in the felt constitution)
Auto-close when: (a) the NDE-flow + architecture sweep is exhausted AND the best CNN is
either ≥ 2875 calibrated (M1 defended as "CNN ≥ L1") OR plateaus below it across N=3
consecutive variants within ±5% (M1 defended as "even best-effort CNN ties/trails L1 — a
genuine practical-suboptimality result, not undertraining"); whichever first. Either outcome
is a publishable resolution of M1.

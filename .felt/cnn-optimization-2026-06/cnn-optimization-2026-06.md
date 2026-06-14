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

## 2026-06-14 ~01:15 UTC — Phase-2 DONE: architecture is a SECONDARY lever (~+6%); resnet18 calibrated win
All 4 archs trained (rc=0) + read out (RealNVP fixed), all fidsumm G1-validated. Full-config 3-seed/9000:
**resnet18 3326 (+6% vs plain 3139, SAME-seed ⇒ clean arch effect), GATE-C CLEAN (HIGH net-bias dim3
+0.004 / dim6 +0.003 ≈ 0)**; plain_attn 3205 (+2%, marginal); resnet_small 3072 (−2%); resnet50_gn
2760 (−12%, OVER-FITS at 899 cosmos, as predicted). NB resnet18 BatchNorm did NOT collapse here ⇒
BN-contamination is harmonic-multi-channel-specific, not auto-only flat-local. **Best CNN now =
resnet18 + RealNVP = 3326 (calibrated) vs L1+product 2875 = +16%.** Campaign reading: the readout NDE
(MAF→RealNVP +36%) was the PRIMARY lever; architecture adds a modest calibrated +6%; plain-conv was
already near-optimal. M1 reversed WITH MARGIN. SEED-ROBUST (confirmed 06-14): resnet18 beats plain at ALL 3 compressor
seeds (s41/42/43 = 3326/3314/3273 vs plain 3139/3092/3232), mean 3304 vs 3154 = +5%, wins every
seed, G1-validated. So the arch gain is real+modest+seed-robust+(s41)calibrated. Summary:
arch_sweep_2026_06_13/SUMMARY_ARCH.md. Scripts: run_compressor_arch_overnight.py + _stage2.py,
build_fiducial_summaries_cnn.py (arch-extended).

## 2026-06-13 ~23:43 UTC — OVERNIGHT Phase-2 LAUNCHED: compressor-architecture sweep (NDE fixed)
Andreas: skip MAF-capacity control (fine with RealNVP); launch Phase 2 = optimize the COMPRESSOR,
do NOT chase summary dimension (fix dim=10). Running: `run_compressor_arch_overnight.py` Stage 1
(GPUs 0&2; GPU1 foreign tenant) — 4 archs (resnet50_gn, resnet_small, plain_attn, resnet18) same
VMIM recipe as baseline (seed 41, 80k, best_val), only --compressor-arch varies. Plain baseline 3139
REUSED (not retrained). Stage 2 (run_compressor_arch_stage2.py, after compressors land): arch-aware
fidsumm build (extended build_fiducial_summaries_cnn.py — plain path re-validated G1 max|Δ|=1.2e-4;
each arch G1-checked) + FIXED RealNVP 4×128 readout → SUMMARY_ARCH.md ranked vs 3139. Claude
monitoring via job notifications overnight; will run Stage 2 + GATE-C any winner. HANDOFF:
arch_sweep_2026_06_13/HANDOFF.md. Expected: arch>3139 ⇒ better compressor (gate it); all ≤3139 ⇒
plain near-optimal at dim10, readout-NDE was the real lever (clean campaign closure). resnet18 (BN)
may collapse (expected, [[project_resnet_bn_contamination]]).

## 2026-06-13 — TIDY-UPS DONE: result fully closed (both probes gated, full-config)
(1) L1 FULL-config (3-seed/9000): MAF **2861** (≈canon 2875, screen-noise resolved), RealNVP **1249**
(craters), MDN **2549** ⇒ L1's best NDE = MAF, confirmed; apples-to-apples holds (CNN-RealNVP 3139 >
L1-MAF 2861). (2) CNN-MAF (2312) GATE C: HIGH net-bias dim3 −0.003 / dim6 +0.006 ≈ 0 ⇒ CALIBRATED,
not over-confident ⇒ MAF UNDER-EXTRACTS from the low-D CNN summary (broad-but-honest); RealNVP (3139,
+0.039) recovers more while staying calibrated. Both probes' baselines now gated. The 2312→3139 gain
= pure estimation EFFICIENCY, no calibration trade. OPEN referee-hole flagged for next: MAF-capacity
control (only default MAF 5×[50,50] tested on CNN; sweep bigger MAFs ⇒ show it's flow FAMILY not
capacity). Then dimensionality generalization (low-D L1-VMIM + RealNVP) + Phase-2 architecture.

## 2026-06-13 — FIGURES: L1-vs-CNN best-NDE contour set (nde_sweep_2026_06_13/figs/)
Four figures (PDF+PNG), all show CNN ≥ L1: (1) corner_rep_patch_both_median (perm36/patch118, both
arms AT their medians — CNN 3292 / L1 2922, the honest single-realization corner); (2)
corner_fiducial_datavector (noise-averaged fiducial vector, conventional deterministic headline —
CNN 3066 / L1 2732, NO OOD inflation); (3) **fom3_distribution_cnn_vs_l1** (the CLAIM-carrier:
per-patch FoM3 violins over 9000 patches — **CNN tighter at 70% of patches**, median 3139 vs 2875
= 1.09×, AND CNN distribution NARROWER ⇒ more realization-consistent); (4) corner_population_stacked
(realization-marginalized, appendix-grade). KEY POINT (Andreas): single contours show only
"comparable, CNN marginally tighter" because a ~9–12% FoM3 edge between two good summaries is
inherently subtle to the eye — the quantitative claim lives in the FoM3 DISTRIBUTION + calibration +
seed-robustness, NOT the eyeballed corner. Standard paper pairing = one clean corner (fiducial or
rep-patch) + the FoM3 distribution. Scripts: corner_l1_vs_cnn_best_nde.py, figs_l1_vs_cnn_rep_and_
stacked.py, makefig_fom3_distribution.py, makefig_fiducial_contour.py. perm16/patch23 (first try)
was median-for-CNN but L1-favorable → per-patch flip; superseded by perm36/patch118 (both-at-median).

## 2026-06-13 — M1 RESOLVED: each probe at its best CALIBRATED NDE ⇒ CNN auto-only (3139) ≥ L1+product (2875)
SEED ROBUSTNESS (workstream B): RealNVP on compressor s41/s42/s43 = **3139 / 3092 / 3232**, every
seed beats L1 2875; MAF baselines 2312/2170/2480 ⇒ +30–43% lift is STRUCTURAL, not a seed-41 fluke.
APPLES-TO-APPLES (workstream A; same NDEs on L1+product 2000-D, screen 2-seed/1000-obs):
jaxili MAF **2778** (≈ canonical 2875, validates setup) | sbilens RealNVP 4×128 **1111** (CRATERS) |
jaxili MDN **2643**. The RealNVP that lifts the CNN CRATERS L1 — diagnosed: 2 RealNVP seeds disagree
1.77 nat val-loss on 2000-D ⇒ pooled covariance inflates (marginals fine, FoM3 tanks) = documented
sbi_lens-RealNVP high-D instability ([[project_nde_architecture_mismatch]]). ⇒ gain is CNN-SPECIFIC
(expressive coupling flow suits low-D 10-D summary, unstable on high-D 2000-D L1); **L1's best NDE
is the MAF (2875), CNN's is the RealNVP (3139).** RESOLUTION: each summary at its best CALIBRATED
readout ⇒ CNN ≥ L1. The "CNN underperforms L1" was an artifact of a COMMON jaxili MAF (poor readout
for the CNN's low-D summary). Andreas (2026-06-13): OK to use different NDEs per probe IF (1) both
calibrated [✓ GATE C], (2) clear qualitative reason [✓ L1 200× higher-D]. **This REVERSES the
paper's current "L1 wins / CNN ties" headline** (CLAUDE.md, M1, [[project_10deg_definitive_cnn_geq_l1]],
[[project_flatsky_cnn_no_cross_gain]] — all used the common MAF). Building the representative-patch
(perm16/patch23) contour overlay (best L1 MAF vs best CNN RealNVP) per Andreas request. TIDYING
before paper claim: (a) full-config L1 arms (confirm 2875); (b) CNN-MAF 2312 is under-efficient-but-
calibrated (broad), not RealNVP fluke; (c) auto-only only (cross story unchanged).

## 2026-06-13 — GATE C VERDICT: the NDE FoM3 gain is CALIBRATED (not over-confidence) ⇒ M1 flips (CNN ≥ L1)
Varied-θ TARP-DRP (cnn_val, 400 disjoint cosmologies; 600 pts; FoM3-tercile-stratified; 3 seeds;
tarp_stratified_val_nde.py + run_tarp_coverage). Net-bias (+conservative / −over-confident),
HIGH tercile = tightest posteriors = where over-confidence shows:
- **A0 sbilens RealNVP 4×128 (3139): HIGH net +0.039 (dim3) / +0.021 (dim6)** — POSITIVE ⇒
  conservative, NOT over-confident. dim3 marginally past L1 ±0.037 band but in the SAFE direction
  (3139 = conservative ⇒ if anything an UNDER-estimate). MID −0.022, LOW +0.028. seed-spread ±0.002.
- **jaxili MDN 10×50 (2885): HIGH net +0.004 (dim3) / −0.026 (dim6)** — PRISTINE.
Both pass "not over-confident" (LANE_A fool's-gold REFUTED). ⇒ CNN with a proper NDE is CALIBRATED
and ≥ L1: RealNVP 3139 (beats L1 2875 +9%) or MDN 2885 (ties). **The 2312→2875 "gap" was the
jaxili MAF under-serving the CNN's 10-D summary — NOT undertraining/worse compressor.** Reframes
prior "CNN underperforms" results (all used the common MAF). M1 referee-defense achieved on the NDE
axis. STILL OPEN before a paper claim: (1) **apples-to-apples — does L1+product ALSO rise under a
better NDE?** (fair = each summary with its best calibrated readout; caveat: CNN-tuned RealNVP 4×128
is for 10-D, L1+product is high-D, so "same NDE" non-trivial — need framing from Andreas); (2)
**compressor-seed robustness** (all on seed-41; ~8% compressor-seed variance historically).
Gate artifacts: gate_c/tarp_summary.json + curves/. Scripts: tarp_stratified_val_nde.py.

## 2026-06-13 — FAN-OUT EARLY SIGNAL: sbi_lens RealNVP NDE ≫ jaxili MAF on SAME summaries (UNCALIBRATED — gate next)
Screen (2 seeds / 1000 obs, frozen seed-41 auto summaries): **A0 sbilens_realnvp 4×128 (production
flow) FoM3 3141** vs jaxili-MAF baseline 2312 (+36%) — BEATS the L1+product bar 2875. A1 6×256 =
2770 (< A0 ⇒ non-monotonic: over-capacity or 2-seed noise). jaxili_maf self-test still running.
**CRITICAL CAVEAT:** summaries are FROZEN ⇒ 2312→3141 is a PURE estimation-path difference (same
I(θ;summary); one NDE mis-estimates). Per [[project_10deg_definitive_cnn_geq_l1]]/LANE_A,
tighter≠better — could be RealNVP over-confidence (fool's gold) OR MAF under-fitting. **GATE C
(TARP+SBC on the ACTUAL sbi_lens RealNVP posteriors — NOT the common-MAF, since the RealNVP IS
what's under test) is DECISIVE and mandatory before any M1 claim.** Launched A0_full (3 seeds /
9000 obs) on GPU 0 to confirm the number is seed-robust. If A0 passes GATE C ⇒ M1 may FLIP to
"CNN ≥ L1, calibrated" AND reinterpret history (jaxili-MAF NDE may have under-served the CNN in
the 10°/definitive comparisons that used it) — strong support that [[project_nde_architecture_mismatch]]
materially mattered. Do NOT headline 3141 until gated.

**SWEEP COMPLETE + HARNESS VALIDATED EXACTLY.** Self-test jaxili_maf_baseline (3 seeds/9000) =
**2312.2597881019083** = B1 bit-for-bit ⇒ new harness reproduces population_sweep exactly.
A0_full (3 seeds/9000) = **3139** (≈ screen 3141, seed+obs robust). Full ladder on SAME frozen
summaries: A0 sbilens-RNVP 4×128 **3139** > jaxili-MDN 10×50 **2885** (≈L1 bar 2875) > A2
8×256 2980 > A1 6×256 2771 > A3 8×512 2690 > jaxili-MAF 2312 > B2 jaxili-RNVP 5×50 **2258**.
KEY REFINEMENTS: (1) NOT just "RealNVP>MAF" — jaxili's OWN RealNVP (5×50) is the WORST (2258);
the win is specifically **sbi_lens RealNVP @ production 4×128** (expressive AffineCoupling + right
capacity), bigger hurts (non-monotonic). (2) NDE alone swings FoM3 2258→3139 (+39%) at ZERO
information change = vivid LANE_A/DPI demonstration (estimation quality, not physics) — publishable
methodological point. DECISIVE: calibrated arm wins. RealNVP-calibrated ⇒ CNN>L1; MDN-calibrated
& RealNVP-over-confident ⇒ CNN≈L1 (ideal M1). Either way the 2312→2875 "gap" was largely the
jaxili-MAF under-serving the CNN summary. NEXT: GATE C (TARP-DRP+SBC) on A0 sbilens-RealNVP AND
jaxili-MDN, using cnn_val.npz (400 disjoint val cosmologies) as the varied-θ set; scope before compute.

## 2026-06-13 — NDE-family harness BUILT + smoke-validated; fan-out ready (awaiting go)
Built `train_nde_from_compressed.py` (one metric loop, reuses preproc + compute_fom3/fom2d/
marginal_stats verbatim) + `run_nde_sweep.py` (GPU-2 fan-out, concurrency cap, ranked SUMMARY.md).
Families: **sbilens_realnvp** (production flow via npe_cnn build_flow/train_flow, wandb-off,
capacity = nvp-layers/hidden) + **jaxili_maf/realnvp/mdn** (NPE model_class, with a
FAMILY-PRESERVING NaN retry — the shared train_with_nan_retry reverts to default MAF on NaN, a
silent-corruption bug for non-MAF arms). All 4 paths smoke-PASS end-to-end (capped train, 100
obs). Bug found+fixed: `jax.nn.silu` = PjitFunction lacking `__name__` in the full import stack
crashes jaxili create_trainer:455; fixed by reusing jaxili's OWN default activation object.
Fan-out matrix: jaxili_maf_baseline (3 seeds/9000 obs = harness self-test → must hit ~2312) +
A0–A3 sbilens RealNVP ladder (4×128 production → 8×512) + B2 jaxili_realnvp + B3 jaxili_mdn,
2-seed/1000-obs screen, GPU-2 concurrency 2, mem 0.18. Awaiting Andreas go (shared GPU 2).

## 2026-06-13 — B1 harness VALIDATED + independent training audit PASS
B1 (jaxili MAF default on frozen seed-41 auto summaries, full settings) → 9000-median **FoM3
2312** vs existing baseline 2325 (−0.6%, within jit/TF32 noise; σ 0.051/0.077/0.245 match) ⇒
HARNESS VALIDATED. **Measured cost split:** NDE train 593–986 s/seed (DOMINATES; 3 seeds ≈40 min);
9000-obs jitted sampling 65 s TOTAL ⇒ n_obs NOT the bottleneck (empirically confirmed); levers =
2-seed screen + GPU-2 packing.
Independent training-health auditor (Opus, own context, artifact-only, no Phase-0 hints):
**OVERALL PASS.** Corroborated Phase 0 + went further: proved best_val ≡ batch38000.pkl (sha) =
test-loss min ⇒ killed the "secret last-step ckpt" failure mode. Caveats adjudicated: (1) LEAKAGE
flag = FALSE ALARM — verified compressor perms 0-4 vs NDE perms 5-6, production log `overlap=[]`;
auditor keyed on TFDS-split-name meta field, missed the perm-subsetting. Residual perm-aug sharing
= known-negligible [[project_tfdata_cross_route_leakage]], and CNN-specific (L1 has no compressor)
so if anything it makes "CNN<L1" CONSERVATIVE. Renamed B1 "oracle"→"harness-validation ref" in
plan. (2) summary eff-rank ~3/10 (corr-cond 301) = REAL → disclose in paper (matches 3-param FoM3
target), not a defect. Report: nde_sweep_2026_06_13/AUDIT_seed41_compressor_and_nde_TRAINING.md.
Process win: independent flag → I verified with the perm-audit line → resolved with evidence
(the right way to handle an auditor concern; don't hand-wave).
Next: build NDE-family harness extension (sbi_lens RealNVP + NSF into the population_sweep
train+sample loop; jaxili MAF/RealNVP/MDN native) → fan out A0–A3/B2/B3, 2-seed screen, GPU-2 packed.

## 2026-06-13 — Phase 0 DONE + baseline CORRECTION (plan signed off)
Plan: `PLAN_CNN_NDE_SWEEP_2026-06-13.md` (repo root). Levers adjudicated by Andreas: pack
variants on GPU 2 ✅; **keep full 323k train (no subsample)** ✅ (avoid a permanent confound,
esp. for larger compressors); screen 2 seeds / finalists 3 ✅; **screen n_obs=1000** (~0.4% SE,
bootstrapped from existing 9000-patch FoM3) ✅; m_samples stays 2000, no training truncation ✅.
GPU posture: **GPU 2 only**, co-resident, conservative mem (Andreas also running jobs there).

**Baseline correction (matters for M1).** `population_sweep_flatsky.py` runs EVERY arm (L1 &
CNN) through a **common jaxili MAF** (5 transforms, hidden [50,50]) at the 9000-obs pooled-3-seed
median — so the primary metric ALREADY fixes the NDE. Corrected apples-to-apples standing:
L1 auto **2405** / +product **2875** (bar) / +both 2910; CNN auto **2325** / +product 2181
(cross HURTS) / +both 2306. ⇒ CNN auto is ~3% *behind* L1 auto (NOT the "tie" the handoff
implied) and −19% vs bar. The handoff's "2620" = best single un-pooled seed (`plot_best_seed.py`),
not the primary metric. Sharpened lever-1: is CNN's 2325 depressed by the *small default MAF*
under-serving its 10-D (corr-cond≈300) summary, and would a stronger NDE lift CNN MORE than L1?
⇒ added MANDATORY apples-to-apples step (§2c): winning NDE must ALSO be run on L1+product.

**Phase 0 checks (frozen seed-41 auto-only summaries, `cnn_none_s41/cache/`): ALL CLEAN.**
SHA256(params) ↔ meta MATCHES; params = `..._best_val.pkl` (best-val, not last-step);
summaries healthy (no NaN/dead dims, 899 cosmos, mild collinearity corr-cond 301); VMIM curve
best-val@step18/40 captured at test-min, mild post-min overfit +0.32 nats (→ Phase-2 lever, not
Phase-1). **Oracle is B1 (jaxili MAF default) reproducing 2325**, NOT A0/2620.
Next: build `train_nde_from_compressed.py` (no GPU) → B1 oracle run on GPU 2 (fresh nvidia-smi
first), confirm 2325, then fan out A0–A3 / B2 / B3.

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

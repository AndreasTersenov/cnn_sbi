---
name: 'Flat-sky cross-maps: de-leaked auto+cross (build, train, contours)'
status: active
tags:
    - experiment
    - sbi
    - cnn
    - l1
    - cross-maps
    - flat-sky
    - paper
created-at: 2026-06-08T16:45:21.09193286Z
outcome: 'RESOLVED 2026-06-10 (calibrated, multi-compressor-seed-robust): on the physically-buildable patch-local cross, L1 gains +20% (product=xi_ij; 2405->2875) while the CNN gets NO SYSTEMATIC gain (product/auto flips sign with the compressor draw 0.94/1.10/0.98, mean 1.00x — zero gain dominated by +-8% seed variance, NOT a systematic loss); every CNN product seed <= 0.85x L1 product (L1 > CNN on the explicit cross channel, draw-robust); auto-only = statistical TIE (CNN seeds 2170-2480 straddle L1 2405). ~92% of the old full-sphere cross gain was leakage. VMIM val loss product~=auto per seed => optimization-limited hypothesis lives at the RECIPE level (untested). OPEN threads: recipe-level test, principled best-seed (val-loss), BNT-cross campaign, scheduler packing (EFFICIENCY_AUDIT_2026-06-10.md). HISTORICAL OBJECTIVE (filed 2026-06-08): Replace the LEAKY full-sphere harmonic cross-maps (every cross-patch is a global functional of the whole sky; auto+cross constraining power partly UNPHYSICAL — see CROSS_MAP_LEAKAGE_FINDING.md) with PATCH-LOCAL flat-sky cross-maps, recompute stats, train L1+CNN, get CALIBRATED cosmological contours. Design+validation DONE (FLATSKY_CROSS_REDESIGN_NOTES.md S1-14): cross map = apodized-circular CONVOLUTION (Zurcher Eq.12 flat-sky analog) AND pointwise PRODUCT (its mean = xi_ij); complementary -> TEST BOTH. NO sim/dataset rebuild (cross computed on-the-fly from auto ch0-3 of TFDS grid_10deg_80px_nonoverlap180; auto-only baseline uses identical autos => clean comparison). CAUTION: do NOT reuse the old --cross-maps route wholesale (it is the bad one we dissected) — reuse only the FFT-product math + apodization window (re-validated), REWRITE the noise/SNR (per-channel, not shared auto-sigma), and compute the FFTs ON-DEVICE/batched (torch/JAX), NOT in a CPU tf.data map (starves the GPU). PRIMARY METRIC: median over typical patches of sigma(w0) + 2D(Om,s8); FoM3 reported NEVER headlined. NEXT: implement on-device augmentation (L1 --cross-op {conv,product,both}; ADD flat-sky to CNN; per-channel/per-scale noise) -> GATE A construction+throughput (bitmatch, xi_ij, benchmark augment ON vs OFF) -> GATE B cosmology-dependence (NEW, decisive) -> train matrix (auto-only / +conv / +product / +both x L1,CNN, 3 seeds) -> GATE C calibration (TARP/SBC/L-C2ST) -> contours vs auto-only AND vs full-sphere. Expect MODEST gains (cross info is large-scale, patch samples it poorly = physically correct). See FLATSKY_CROSS_BUILD_PLAN.md + HANDOFF_FLATSKY_CROSS_2026-06-08.md. Continues [[definitive-l1-vs-cnn-10deg-2026-06]].'
---

## Primary metric
median over typical patches of sigma(w0) + 2D(Om,s8) area. FoM3 reported, NEVER headlined (feedback_fom3_fragile_use_2d_areas).

## Done condition
Each arm cross-gain over auto-only is measured AND calibration-validated (TARP/SBC/L-C2ST); conv-vs-product complementarity decided; honest flat-sky gain compared to the inflated full-sphere number. Stop when contours + comparison are produced and written up.

## Guardrails
patch-local cross ONLY (never full-sphere = leakage); per-channel noise (not shared auto-sigma); never PCA L1; GPU 1 only; example-disjoint compressor/NDE split by perm; calibrate BEFORE contours; SAME auto channels across all arms; one apodized-circular convolution definition; do not relitigate the operator choice (notes S8-12).

## Loop status (bnt-campaign-ready 2026-06-10 ~16:15)
★ BNT CAMPAIGN ORCHESTRATOR READY (run_flatsky_bnt_campaign.py, dry-run validated): P0 sigma
freeze --bnt (skipped if table exists) → P1 L1 both-BNT build (solo) → P2 L1 {none,product} BNT
slices + P3 CNN {none,product}×s{41,42,43} BNT compressors (RECIPE-MATCHED to no-BNT baseline:
plain/80k/val-batches-1 — inflation ratios must compare like-trained arms) → fidsumms (CNN ×6 +
L1 precompute --bnt + per-arm slice) → 8 jit sweeps → BNT_CAMPAIGN_RESULT.md with DERIVED
inflation table (no-BNT refs read from disk: L1 2405/2875, CNN per-seed multiseed medians) +
derived prediction-ladder verdict (L1-auto <0.9 / L1-product > L1-auto / CNN >0.9, contingency
note if CNN inflates). precompute_fiducial_both_datavectors.py parameterized (--bnt/--both-cache/
--sigma/--out). Est. ~5-6 h on 2 GPUs (post-jit). AWAITING Andreas's GO + free GPUs (recipe
check still running).

## Loop status (bnt-wiring-built 2026-06-10 ~15:45)
★ BNT WIRING BUILT + GATE-A VALIDATED (CPU, all 8 op×bnt combos PASS; matrix det=1.0 exactly —
unit-determinant lower-triangular). Single source of truth: flatsky_cross.apply_bnt_{np,torch,jax}
+ bnt= switch on all three build_channels_* (BNT'd autos feed BOTH auto and cross channels).
CNN: --flatsky-bnt (npe_cnn + fidsumm) → RMS estimator + jitted transform, tomo4-validated,
mutually exclusive with legacy --apply-bnt, save-path _bnt suffix, CONDITIONAL flatsky_bnt cache
fingerprint key (old caches stay valid; BNT can't hit a no-BNT cache). L1: --apply-bnt now works
on flat_local (was hard error) → SNR calibration + dataset passes + obs L1; cache dir _bnt;
both-cache space-mixing tripwire. freeze_flatsky_cross_noise.py --bnt → _bnt.npz with bnt=True
key; select_frozen_sigma HARD-ERRORS on table/arm BNT mismatch (fences the May wrong-σ failure
mode). GATE A1b auto-vs-white uses the analytic BNT reference (per-bin sqrt(sum B_ij^2); BNT of
independent whites stays white PER MAP — correlation is between bins). REMAINING before campaign
launch: BNT noise-freeze run (1 GPU job when recipe check frees a slot) → campaign orchestrator →
Andreas's go. Recipe check (160k) still in compressor phase.

## Loop status (recipe-check-launched + BNT-scoped 2026-06-10 ~15:00)
Andreas: START recipe test, SKIP L1 per-seed retrain, BNT = PAPER PILLAR 2. ★ LAUNCHED
run_recipe_160k_check.py (pid 3977580, GPUs 1+2, idle co-tenants): {none,product} × s{42,43} at
160k steps + --compressor-val-batches 16, paired vs the 80k multiseed numbers → multiseed_160k/
RECIPE_160K_CHECK.md (~2 h; monitor armed). Tests the RECIPE-level optimization-limited
hypothesis AND calibrates the BNT contingency ladder. ★ BNT PLAN REWRITTEN with Andreas's
framing (NEXT_THREADS_PLAN §B): prediction ladder = L1-auto inflates / L1+product inflates less /
CNN ~no inflation ⇒ BNT lossless for channel-mixing compressors. Inflation ratio FoM3_BNT/
FoM3_noBNT vs existing arms. Grounding: prior 20° campaign (advanced arch, 120k, 5 seeds)
recovered 0.85× FoM3 (+3.7% std_sum) — near-lossless but needed the BIGGER compressor; plain CNN
may inflate here without falsifying losslessness → contingency ladder (160k recipe → advanced
arch → discuss). Implementation: tomo4_bnt_v1 matrix from bnt_utils (bin-dependent only, carries
to 10°), JAX applier behind --flatsky-bnt in make_flat_cross_transform + np oracle + L1 torch
twin, GATE A extended, L1 noise σ RE-FROZEN through BNT (noise-only realizations must be BNT'd —
correlated post-BNT noise is the point). 4 arms (L1/CNN × auto/auto+product, BNT) + 2 extra CNN
compressor seeds for the headline ratio (multiseed lesson). AWAITING Andreas sign-off on §B
before building.

## Loop status (jit-sampling-adopted 2026-06-10 ~14:00)
★ SAMPLING JIT MEASURED + ADOPTED. bench_sample_jit.py (GPU 2): eager 183 ms/obs → jit 1.05 ms/obs
(174×); vmap unnecessary. Bit-identity fails at TF32 kernel level (max|Δ| 3.4e-3, same keys/u-draws)
⇒ adoption gate = full-arm rerun: validate_jit_sweep.py re-derived none_s42's 9000-obs pooled
median in 49 s (vs ~4100–4800 s eager): FoM3 −0.39%, σ ±0.2% — 10× inside seed scatter
(jit_validation.json). Wired as DEFAULT into population_sweep_flatsky.py (--sample-eager = legacy
bit-exact path). Sweeps now NDE-training-bound ~30 min/arm (was ~100). Keys-not-bits is the new
reproducibility contract for sweeps (flagged to Andreas). Packing benchmarks (NDE-training
co-residency, the post-jit bottleneck) deferred — GPUs 0/1 have ACTIVE foreign tenants right now;
run before the BNT campaign launch. Audit fix batches A+B all landed; scheduler Tier-1 next.

## Loop status (multiseed-verdict 2026-06-10 ~13:30)
★★ MULTI-COMPRESSOR-SEED CHECK DONE (277 min, 4 sweeps n=9000 each). Pooled 3-MAF 9000-obs median
FoM3 — auto: s41 2325 / s42 2170 / s43 2480; product: 2181 / 2393 / 2433 ⇒ product/auto = 0.94 /
**1.10** / 0.98 (mean-of-seeds 1.00×). VERDICT (mixed branch): the strict "CNN gains NOTHING" is
NOT compressor-seed-robust — the cross effect FLIPS SIGN with the draw; correct claim = ZERO
SYSTEMATIC gain, dominated by compressor-seed variance (±~8%). ROBUST facts: every CNN product
seed ≤ 0.85× L1 product 2875 (L1 +20% stands; L1 > CNN on the explicit cross channel across all
draws); CNN auto seeds STRADDLE L1 auto 2405 ⇒ auto-only = statistical tie. VMIM val loss product
≈ auto per seed (Δ≲0.02 nats) ⇒ compressor objective sees no extra MI in the product channel at
this recipe ⇒ optimization-limited hypothesis moves to the RECIPE level (untested), seed-level
rescue falsified. Writeup: FLATSKY_CNN_RESULT.md gains a derived "Robustness — compressor seed"
section (generator-emitted, consolidate_cnn_vs_l1.py); MULTISEED_COMPRESSOR_CHECK.md verdict
hand-corrected (the in-flight driver predated the 5f1afd9 hardcoded-verdict fix and wrote the
WRONG verdict against its own 1.10× table — live demonstration of the bug). Memory
project_flatsky_cnn_no_cross_gain rewritten. NB the A2 "~20% on-device cross cost" note below is
WRONG (run-level medians: product/both at parity or faster than none; 75-vs-93 was ambient load).
NEXT: bench_sample_jit on freed GPUs → packing benchmarks → scheduler plan (EFFICIENCY_AUDIT_
2026-06-10.md); audit fix batches A (5f1afd9) + B (dabda3a) + GPU-policy update (c0ae139) landed.

## Loop status (audit+fixes 2026-06-10 ~12:15)
★ PIPELINE AUDIT DONE (4 parallel subagents, read-only): PUBLISHED NUMBERS STAND — disjointness
CLEAN (metadata perm-filter + runtime audit; artifact row counts verified), RMS-whitening CLEAN
(bit-identical via deterministic recomputation + G1 ≤1.9e-4), aggregation CLEAN (n=9000/9000 all 8
arms, best_val genuine, preprocessing consistent). All findings forward-looking: dead NaN guard in
train_with_nan_retry (hasattr on dict — never fires); make_headline_corner.py conditioned on train
row 0 not the obs (latent, never produced output); HARDCODED VERDICTS in run_multiseed_compressor_
check.py + consolidate_cnn_vs_l1.py + plot_best_seed.py title; --decay-steps silently inert in
jaxili (LR const 1e-4, symmetric); "3 MAF seeds" vary ONLY the data split (flow init fixed at 42);
best_val selection = single random 128-batch (noisy); sweeps are HOST-DISPATCH-BOUND (jaxili
.sample un-jitted, ~600 dispatches/call; dim-3200 only ~25% slower than dim-10) ⇒ jit fix ~10
lines. Full report: cnn_phase/../PIPELINE_AUDIT_2026-06-10.md. BATCH-A FIXES COMMITTED 5f1afd9
(derived verdicts, wrong-obs fix, --compressor-val-batches, channel_scale+effective-policy in
cache meta, truth key). BATCH B (population_sweep_flatsky.py + npe_l1norm imports) DEFERRED until
the multiseed driver exits — pending product sweeps execute those files from disk. bench_sample_
jit.py ready (bit-identity gate + timings, run on GPU 1 when free). NEXT_THREADS_PLAN_2026-06-10.md
drafted (best-seed-by-val-loss, BNT design: noise→BNT→cross-build→whiten, batch-B+jit). MULTISEED
EARLY READ: auto-only compressor seeds 2325/2170/2480 STRADDLE L1 2405 (best +3%, worst −10%);
product-vs-none compressor VMIM val loss ≈ EQUAL per seed (−10.76≈−10.75 s42, −10.80≈−10.82 s43)
⇒ product channel adds no measurable MI at the compressor objective. Product sweeps land ~13:40.

## Loop status (fable5-handoff 2026-06-10)
SESSION HANDOFF to Fable 5 written: HANDOFF_FABLE5_2026-06-10.md (repo root, the new entry point) +
FABLE5_FIRST_PROMPT.md (copy-paste first prompt). ★ LIVE QUESTION (Andreas 2026-06-10): is the CNN
no-cross-gain OPTIMIZATION-LIMITED, not a real method difference? His point: CNN gets bins as channels
⇒ can learn cross-correlations implicitly ⇒ explicit cross redundant for CNN; cross-map trick is for
per-channel methods (L1). NOT a 10-d bottleneck (10-d enough for 6 params) — he reads it as compressor
TRAINING INEFFICIENCY, and notes BEST seed (2620) > L1 auto (2405). Data nuance: CNN auto TIES L1 auto
(not beats), big per-seed scatter (auto 2620/2364/2387, product 2225/2331/2017) ⇒ optimizer-into-
optima not capacity wall; but that scatter is MAF-seed (1 compressor), best-vs-L1 is best-vs-pooled
single-obs (suggestive not clean). RUNNING NOW: run_multiseed_compressor_check.py (compressor seeds
42,43 × {product,none}, GPU 1+2, ~4h) → does a well-trained COMPRESSOR lift product toward/over L1?
→ cnn_phase/multiseed/MULTISEED_COMPRESSOR_CHECK.md. NEXT-SESSION TODO: (1) interpret it + reframe
FLATSKY_CNN_RESULT.md/memory IF supported; (2) principled best-seed (val-loss, not post-hoc FoM3;
L1 per-seed needs retrain — 2000-d reload truncates); (3) BNT for flat-sky cross (scope first); (4)
bug/inefficiency audit (fan out subagents). Commits through cf96f25→(this). Branch pushed.

## Loop status (cnn-best-seed 2026-06-10)
★ CNN BEST-SINGLE-SEED (un-pooled) CHECK DONE 2026-06-10. The no-cross-gain is NOT a pool-haircut
artifact: reloaded the per-seed MAF checkpoints (CNN 10-d reloads bit-exact; L1 2000-d truncates →
can't), sampled each MAF seed separately at the typical obs (perm16/patch23). Best single seed FoM3:
auto 2620 | conv 2491 (0.95×) | product 2331 (0.89×) | both 2475 (0.94×) — every cross arm STILL
≤ auto-only. Auto-only tie holds (CNN best-seed 2620 ≈ L1 pooled 2487; fair pooled-vs-pooled 2325 vs
2405). Even CNN's best-seed product is wider than L1 pooled. (NB: MAF seeds, not compressor seeds —
still 1 compressor seed; the multi-COMPRESSOR-seed check remains the open robustness follow-up.)
Artifacts: cnn_phase/best_seed/{CNN_BEST_SEED.md, per_seed.json, fom3_best_seed.*, corner_best_seed_
product.*, corner_best_seed_vs_l1_<arm>.*}; scripts cnn_per_seed_best.py + plot_best_seed.py. Baked
into FLATSKY_CNN_RESULT.md (§Robustness) + GATE_C_INTERPRETATION.md. Commits d4884b3→cf96f25, pushed.

## Loop status (cnn-overnight 2026-06-09 22:10)
AUTONOMOUS OVERNIGHT PIPELINE LAUNCHED (overnight_cnn_pipeline.sh, detached pid 1518292). Waits for
the population sweep, then runs: headline consolidate → SBC(cnn) → L-C2ST(cnn) → representative
corners(3-seed,4 arms) → final consolidate. Logs+PASS/FAIL in cnn_phase/STATUS_OVERNIGHT.md;
deliverables: FLATSKY_CNN_RESULT.md (root, L1-vs-CNN table), cnn_phase/figs/ (overlays+bars),
cnn_phase/gate_c/{tarp_drp,sbc,lc2st}/.
STATE: matrix DONE (4 arms plain-CNN seed41 80k; `both` NaN'd → fixed w/ --compressor-grad-clip 1.0,
best_val@38k). Fiducial summaries DONE (9000/arm, all 4 G1 PASS max|Δ|≤1.4e-4). GATE C TARP DONE
(product/none/both calibrated-or-conservative; conv-HIGH mildly OVER-CONFIDENT −0.068, least
important arm). SBC(cnn) TESTED: means≈0.5 no bias, std 0.273-0.281 (<0.289 ⇒ mildly CONSERVATIVE/
over-cover, matches TARP), KS flags mild non-uniformity Om/s8 (safe direction). Population sweep
RUNNING (single-wave 4-GPU; ETA ~22:30) → the 9000-obs median FoM3 table.
SINGLE-OBS PREVIEW (perm0/patch90, CNN product 1-seed FoM3 2549 vs L1 2676; overlay = comparable
contours): de-leaked CNN≈L1 (NOT the leaky full-sphere CNN≫L1 ~17k vs 8.5k). Headline holds: the
big CNN cross advantage was mostly leakage. New scripts: npe_cnn_nbody_tomo.py --cnn-map-route
flat_local + --cross-op + --compressor-grad-clip; build_fiducial_summaries_cnn.py flat_local;
gate_a_flat_cross_cnn.py; run_flatsky_cnn_{matrix,fiducial_summaries,gate_c_tarp,gate_c_lc2st,
population_sweep,repr_corners}.py; compute_sbc_from_tarp_dumps_cnn.py; cnn_representative_corners.py;
quick_single_obs_cnn.py; consolidate_cnn_vs_l1.py. NONE committed. NEXT (morning): read
STATUS_OVERNIGHT.md + FLATSKY_CNN_RESULT.md; verify L-C2ST/repr corners PASSed; write memory +
final writeup; git stage by path.

## Loop status (cnn-live)
CNN PHASE STARTED 2026-06-09 (session 3). Decisions LOCKED w/ Andreas: plain CNN (conv 64,128,256/
dense 256/dim 10, NO BatchNorm); 1 compressor/arm seed-41 + 3-MAF-seed pooling downstream (mirrors
Phase D, symmetric w/ L1 — removes NDE confound via the COMMON jaxili MAF); GPU 1+2 granted; per-
channel RMS whitening frozen from a TRAIN-SAMPLE (not fiducial). 4 arms none/conv/product/both.
WIRED into npe_cnn_nbody_tomo.py: --cnn-map-route flat_local + --cross-op {none,conv,product,both}
(reads autos ch 0-3 only; builds patch-local cross ON-DEVICE in JAX via flatsky_cross.build_channels_jax
roll 0.10; per-channel RMS whiten; same transform for train/val/NDE-compress/obs). Shared helpers
compute_flat_cross_channel_rms + make_flat_cross_transform (importable by build_fiducial_summaries_cnn).
★ GATE A PASS (2026-06-09): A1 (gate_a_flat_cross_cnn.py) all 4 ops PASS — jax-vs-numpy oracle exact
(none/product) / 1.7e-7 FFT-roundoff (conv/both), channels 4/10/10/16, raw autos preserved exact,
whitening exact, deterministic (obs↔train identity). Product channel RMS ~115-167× smaller than autos
(whitening MANDATORY; post-whiten std≈1). A2 throughput (real GPU-1 train, measured): steady-state
none ~93 it/s, product ~75 it/s (batch 128) ⇒ on-device cross costs ~20%, does NOT starve GPU (no
L1-style 251/s collapse). At 75 it/s an 80k-step train ≈ 18 min + 1 one-time compress pass; single
TFDS reader/job. AWAITING Andreas sign-off before launching the 4-arm compressor matrix.

## Loop status (live)
★★ CAMPAIGN RESULT DONE 2026-06-09 (population sweep complete, 9000 obs/arm pooled 3-seed median):
~92% OF THE FULL-SPHERE L1 CROSS GAIN WAS LEAKAGE. flat-local L1 FoM3: auto-only 2405 | +conv 2499
(1.04×) | +product 2875 (1.20×) | +both 2910 (1.21×) vs full-sphere auto+cross 8530 (3.88×). Physical
patch-local cross retains +21%; conv(=alm-product/Zürcher flat-sky analog) +4% ⇒ ~99% of THAT op's
gain was leakage; pointwise PRODUCT (=ξ_ij) survives +20% (σOm/σs8 −9%); both≈product (high-dim NDE
fine in pooled median). Leakage lived in w0 (full-sphere σw0 .246→.188 vs patch-local .245→.232; σOm
physical-both .046 MATCHES full-sphere .046). Auto-only 2405≈full-sphere 2200 validates. POPULATION
MEDIAN OVERTURNS single-obs ranking (single-obs conv+32%>product+17% via per-seed-mean+favorable patch;
pooled median product≈both≫conv). Calibrated TARP✓+SBC✓ (L-C2ST N/A high-dim). Writeup
FLATSKY_CROSS_RESULT.md + memory project_flatsky_cross_deleaked_result. Per-arm medians in
results/exploratory/flatsky_cross_2026_06/population_sweep/<arm>/median_summary.json.
NEXT: CNN arms (jax flat cross + per-ch RMS) → de-leaked L1-vs-CNN. HANDOFF written:
HANDOFF_FLATSKY_CNN_2026-06-09.md (repo root) — start the next session there. L1 side COMPLETE +
calibrated + written up (FLATSKY_CROSS_RESULT.md) + showcase/representative-corner plots delivered.
CNN phase notes (in handoff): wire --cnn-map-route flat_local into npe_cnn_nbody_tomo.py (jax flat
cross on-device, per-channel RMS not frozen-σ, resnet50_gn/plain not BN, example-disjoint perm split,
build-both-slice does NOT transfer so each arm = own compressor; L-C2ST WORKS for CNN 10-dim).

## Loop status (gate-c)
★ GATE C CLOSED 2026-06-09 (Andreas: "fine with just tarp and sbc"): TARP-DRP ✓ + SBC ✓ both pass;
L-C2ST N/A (UNDERPOWERED at high-dim L1: self-test ST_H0 ok median-p 0.98 but ST_H1 +0.5σ-w0 not
detected median-p 0.85 ⇒ logreg can't resolve local miscalib conditioning on 800-3200-dim x with
n_cal=2000; prior campaign ran L-C2ST on CNN 10-dim where it works. Self-test gate correctly aborts
— see memory). ⇒ the de-leaked flat-local cross result is CALIBRATION-TRUSTWORTHY.
CONSOLIDATED HEADLINE (calibrated): patch-local cross gains MODEST — conv +32% FoM3 / σ8 0.89,
product +17% / Ωm 0.77, both +13% (not additive past conv; high-dim NDE), w0 ~unhelped — vs LEAKY
full-sphere auto+cross L1 8530 (~3.9×) ⇒ MOST OF THE FULL-SPHERE CROSS GAIN WAS LEAKAGE. conv→σ8,
product→Ωm operator split (matches GATE B). All single-obs/3-seed; population sweep firms the headline.
NEXT: population sweep (9000 fiducial obs/arm; obs datavectors precomputed fiducial_both_datavectors.npz
→ slice per arm + retrain-in-process sample like TARP → median σ(w0)/2D/FoM3) → full-sphere SUMMARY_PHASE_D
comparison → writeup. THEN CNN arms (jax flat cross + per-ch RMS).

## Loop status (archive)
GATE C TARP-DRP = PASS (2026-06-09, run_flatsky_gate_c_tarp.py → tarp_stratified_val retrain+val-
sample, 4 arms × 3 seeds, FoM3-tercile-stratified). dim-3 (Om,s8,w0): ALL arms + ALL terciles incl
HIGH (tightest cross posteriors) hug diagonal, max|dev| 0.037-0.094 (mostly <0.05), calibrated or
mildly CONSERVATIVE (over-cover), NEVER over-confident ⇒ the modest de-leaked cross gains are REAL,
not over-tight. dim-6: mild dev (<0.10) in weak nuisances h0/ns/Ob (matches prior-campaign known
mild nuisance miscalib); science params clean. Figs gate_c/tarp_drp/figures/tarp_{overlay,per_arm}_dim{3,6}.
Per-pair 2D (mean of seeds, single-obs): conv→σ8 (σ 0.89), product→Ωm (σ 0.77); gains modest (2D FoM
1.1-1.35×); both NOT additive past conv; w0 barely helped. compute_l1_2d_areas.py, plot_l1_matrix_corners.py.
NEXT GATE C: SBC (run_sbc_harm_l1_nobnt.py global rank uniformity) + L-C2ST (lc2st_diagnostic.py local
@fiducial). Then population sweep (9000 obs, datavectors precomputed) + full-sphere comparison + writeup.

## Loop status (history)
L1 MATRIX COMPLETE 2026-06-09 (build-both-slice, all 12 OK; 48min both-build pass cached then 19min
slice+train). FoM3 (pre-GATE-C, mean of seeds 41/42/43):
  auto-only(none) 2354 | conv 3112 (+32%) | product 2759 (+17%) | both 2664 (+13%).
HEADLINE (robust, seed-stable ~5%): patch-local cross gains are MODEST (conv +32%, product +17%)
vs the LEAKY full-sphere auto+cross L1 8530 (~3.9×=+290%) ⇒ MOST OF THE FULL-SPHERE CROSS GAIN WAS
LEAKAGE. The de-leaking test confirms the CROSS_MAP_LEAKAGE_FINDING prediction. conv > product
(conv is the strongest single patch-local cross operator; the smooth large-scale morphology helps).
⚠ both(16ch,3200-dim) 2664 < conv(2000-dim) 3112 — IMPOSSIBLE for info content ⇒ NDE ARTIFACT: the
MAF overfits the high-dim 3200 input (both_s41 early-stopped best@epoch9). both arm UNRELIABLE as-is;
complementarity (does conv+product > each alone?) is CONFOUNDED until the high-dim NDE training is
fixed (more patience/epochs, or L1-VMIM compression — NOT PCA). conv is the clean cross read.
NEXT: (a) investigate/fix both high-dim NDE under-training; (b) σ(w0)+2D per arm (not just FoM3);
(c) GATE C calibration TARP/SBC/L-C2ST BEFORE any contour; (d) population sweep (obs datavectors
precomputed: fiducial_both_datavectors.npz 36000×3200); (e) compare vs auto-only AND full-sphere.

## Loop status (history)
L1 MATRIX = BUILD-BOTH-ONCE-SLICE (run_flatsky_l1_matrix.py, 2026-06-09, 3rd relaunch). The 4 arms
read the SAME autos & 'both'(16ch) is the superset ⇒ none/conv/product are exact column-slices of
'both' (validated bit-identical to ~2e-11 GPU-roundoff). Phase 1: build 'both' datavector ONCE solo
(the expensive loader pass, ~45min, loader-bound ~174/s steady — GATE A2's 486/s was a PREFETCH
ARTIFACT, real steady-state is lower). Phase 2: 11 arm×seed jobs --flatsky-both-cache (slice cols +
slice ranges, obs computed per-op single-map, NO loader pass; NDE-light, pack 4/GPU). Eliminates the
naive version's redundant 4× loader passes + disk-I/O thrash (4 concurrent readers → 40/s).
EARLY PARTIAL (pre-GATE-C, separate-build run, now superseded by single-source): auto-only FoM3
mean ~2323 (≈ full-sphere auto-only 2200 ✓); product mean ~2763 (+19% over auto-only) ⇐ vs leaky
full-sphere auto+cross 8530 (~3.9×) → SUGGESTS most full-sphere cross gain was LEAKAGE. NOT trusted
until GATE C. FoM3 OK to headline now (rule retired 2026-06-09) but report σ(w0)/2D alongside.
NEXT after matrix: pool per-arm FoM3 + σ(w0)/2D, GATE C calibration (TARP/SBC/L-C2ST) BEFORE any
contour, single-obs quick-look (labeled), then population sweep + compare vs auto-only AND full-sphere.
LOADER NOTE: filter decodes 1.13M train→keep 323k (perm 5-6) = 3.5× waste; bumping interleave
cycle_length for FINITE passes is a deferred speedup for the population sweep (don't touch mid-run).

## Loop status (history)
STARTED 2026-06-08 (session 2). Plan signed off by Andreas; 5 design decisions LOCKED:
(1) apodization roll = 10%. (2) L1 cross noise = full per-(channel,scale) σ, FROZEN at fiducial:
add ~32–50 indep noise realizations to the SIGNAL, rebuild cross, wavelet, take per-scale std
across realizations (captures n⊛n AND colored κ⊛n); record in meta; NOT per-patch/cosmo. ALL
channels uniform (autos too). Repo's NOISE-based SNR convention (coeff / propagated-noise-σ), NOT
Zürcher filtered-map-std. (3) product = raw κᵢ·κⱼ, NO ×W²; wavelet handles scales+edges; boundary
handling identical across all channels. (4) slice ch 0–3 of 10-ch TFDS (byte-identical autos to
full-sphere campaign → confound-free; verify ch order + post-preprocess identity in GATE A; autos
are SHT-roundtripped lmax1024). (5) reuse auto n_scales.
INJECTION POINT (verified): wl_stats_torch compute_wavelet_transform → get_noise_levels assumes
WHITE noise per-scale; override stats.noise_levels with frozen per-(ch,scale) σ then recompute
snr_coeffs before compute_wavelet_l1_norms.

PROGRESS (session 2, 2026-06-08):
- GPUs 1+2 both granted this session (Andreas override of GPU-1-only); max them out.
- scripts/sbi/flatsky_cross.py written (np/torch/jax conv+product, bit-identical).
- GATE A1 (operator correctness) ALL PASS: bit-match np/torch/jax conv ~2e-6 / product exact;
  ξ_ij recovery reproduces §14 (diag 0.534→1.039, r 0.479>0.311>0.170); unapod conv mean ~6e-10.
- DATA REALITY (verified): ch0-3 are NOISY SHT-roundtripped autos (no augmentation noise needed);
  perms = independent skies (mean-over-200 → noise floor, ratio 1.05) so freezing needs PAIRED
  realizations; existing channel_empirical_global is total-RMS (signal+noise), NOT noise-based.
- scripts/sbi/freeze_flatsky_cross_noise.py written + RUN. Frozen per-(ch,scale) σ saved to
  results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.{npz,json}. R=48, Bessel
  R/(R-1), all 16 channels, faithful sphere-SHT noise (nside512/lmax1024/σ_pix from cache).
- GATE A1b ALL PASS + FINDING: analytic white propagation mis-normalizes even AUTO finest scale
  by ~2× (0.48×, finest starlet ℓ≳1440 > lmax1024 band-limit) and +19% coarsest; conv cross-noise
  per-scale profile departs hugely from white (L1 dist 0.68), product 0.25. ⇒ frozen-empirical
  fixes autos too; auto-only L1 will differ slightly from prior full-sphere campaign normalization.
- scripts/sbi/flatsky_cross_l1.py written (frozen-σ override L1, on-device cross, train/val/obs,
  per-channel SNR-range calibration). GATE A1c ALL PASS: shapes (conv/prod D=2000, both 3200),
  obs↔train bit-match, autos op-independent, conv≠product, SNR not collapsed (median ~1.0 vs old
  bug ~1e-4). FINDING: pointwise PRODUCT is intrinsically heavy-tailed (prod_23 raw max|SNR| 181;
  conv/auto ~25) ⇒ adopted PER-CHANNEL percentile (0.5/99.5) histogram ranges. NOT the old band-aid
  (σ now correct; percentiles only set bin extent, cf. Zürcher fixed [-4,4]). AWAITING Andreas read
  on per-channel-percentile vs fixed-symmetric range before main-script wiring.
- L1 WIRING DONE: --cross-maps-route flat_local + --cross-op {none,conv,product,both} +
  --flatsky-cross-sigma + per-channel frozen percentile SNR ranges (clamp_overflow=True),
  meta provenance block (apod 0.10, frozen σ, ranges, channel source). Smoke run PASSED
  end-to-end (obs L1 2000-dim, calibration, datavector build, route resolution all correct).
- THROUGHPUT FIX: per-channel wavelet loop starved GPU (29% util, 251/s). Batched wavelet over
  channels + channel-chunking (FLATSKY_MAX_WAVELET_MAPS=6144, OOM-proof) → conv 870/s, both 486/s.
- CAMPAIGN MATRIX APPROVED (Andreas): 4 arms (none/conv/product/both) × {L1,CNN} × 3 seeds; L1
  arms FIRST (de-risk); single-obs quick-look (labeled, not a result) before 9000-obs sweep;
  NO contour trusted before GATE C. Scope reminders: final compare vs auto-only AND vs full-sphere
  SUMMARY_PHASE_D (how much gain was leakage); provenance in every meta. (B) fixed-[-4,4] binning
  robustness = BACKLOG.
- ALL CHEAP GATES PASS: A1 operators / A1b noise-σ / A1c L1-module / A2 throughput (cross build
  does NOT starve GPU; rel-to-none = channel-count ratio) / B cosmology-dependence (cross tracks
  σ8 0.96, w0 0.72-0.80, product tracks Ωm 0.78 >> conv 0.43 ⇒ complementary).
NEXT (A2/B checkpoint cleared): build flat_local L1 orchestrator (4 arms × 3 seeds, GPU 1+2,
datavector cached per arm) → launch → GATE C calibration (TARP/SBC/L-C2ST) → single-obs contours
→ population sweep → CNN wiring (jax flat cross + per-ch RMS) → CNN arms. Report at each gate.

## Pointers
FLATSKY_CROSS_BUILD_PLAN.md (steps/gates), FLATSKY_CROSS_REDESIGN_NOTES.md (design+validation S1-14), CROSS_MAP_LEAKAGE_FINDING.md (why), HANDOFF_FLATSKY_CROSS_2026-06-08.md (handoff). Memory: project_cross_map_leakage_fullsphere, feedback_l1_cross_must_use_harmonic_route, feedback_never_pca_l1, feedback_fom3_fragile_use_2d_areas, feedback_gpu1_only, reference_jaxili_checkpoint_reload_truncation.

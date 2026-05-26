---
name: Canonical-anchors refresh — 4 arms × 3 seeds with proper 70/30 split discipline
tags:
    - experiment
    - sbi
    - cnn
    - l1
    - weak-lensing
    - methodology-fix
created-at: 2026-05-24T09:55:35.882936928Z
status: open
outcome: 'Refresh of the 4 headline anchors (CNN auto-only, CNN auto+cross, L1 auto-only, L1 auto+cross) on the canonical setup that all recent runs missed: --compressor-train-split train[:70%], --nde-train-split train[70%:], --require-disjoint-train-examples on. All on nonoverlap48 TFDS. Background: every CNN run in cnn-auto-push and cnn-auto-cross-push used --compressor-train-split=train AND --nde-train-split=train (100% example overlap), leading to over-confident posteriors. L1 anchors were similarly using full train for NDE. Internal comparisons within campaigns (H1 attention haircut identity, H3 cdim=100 cratering) are robust because they share the same contamination pattern; cross-pipeline absolute comparisons need refreshing. Configs match the existing ''best simple plain CNN'' (iter-108-Q6ON-60k recipe) and ''v2_chsigma L1'' anchors. ETA ~3-5h. Decision rule: report 4 anchors, the corrected cross/auto ratios for CNN and L1, the (auto, cross) pool/MoS haircuts. Output dir: scripts/sbi/results/exploratory/canonical_anchors_refresh/. Will write METHODOLOGY.md as the canonical reference once the runs land. Launcher: launch_canonical_all.sh. The fiber [[cnn-h2-data-limit-scoping-2026-05]] depends on these corrected anchors; no other open fibers depend on this.'
---

## Objective

Refresh the four headline anchors used in `[[CNN_CROSS_MAPS_INFORMATION_NOTE]]`
and all downstream cross-pipeline comparisons:

- **CNN auto-only** (current stale anchor: pooled FoM3 ≈ 11,130)
- **CNN auto+cross** (current stale anchor: pooled FoM3 ≈ 23,986)
- **L1 auto-only** (current stale anchor: pooled FoM3 ≈ 9,015)
- **L1 auto+cross** (current stale anchor: pooled FoM3 ≈ 33,820)

Re-run on the **canonical setup**: `nonoverlap48` TFDS, 70/30 compressor/NDE
train split, `--require-disjoint-train-examples` enforced, `--zero-mean-maps`
on. Configs match the simplest-best-performing setup found in prior campaigns
(plain-CNN at the iter-108-Q6ON-60k recipe; L1 at the v2_chsigma recipe).

The point is **not** to test a new hypothesis — it's to get the four
absolute numbers right so that downstream comparisons (CNN-vs-L1, cross/auto
ratios, the H1 / H2 / H3 / global-info hypothesis ranking) can be quoted
with confidence in the paper.

## Background — what we discovered (2026-05-24)

During an apples-to-apples audit triggered by Andreas's question about
patch sizes, two systematic methodological issues surfaced:

1. **TFDS variant mismatch** between auto-only anchors (used standard
   `grid_20deg_160px`, with overlapping patches) and auto+cross anchors
   (used `grid_20deg_160px_nonoverlap48` for L1, or the matching harmonic
   cache for CNN, both with 48 non-overlapping patches per cosmology).
   Same patch size, different patch-sampling strategy. L1 auto-only on
   `nonoverlap48` (11k pool) is *better* than on the standard grid (9k pool)
   — counter to the "more training data is always better" intuition —
   suggesting overlapping patches actively degrade the L1 datavector
   estimator.
2. **Compressor/NDE training overlap** across all recent CNN campaigns.
   Both autoresearch drivers (`autoresearch_cnn-auto-push/run_arm.py:61-63`
   and `autoresearch_cnn-auto-cross-push/run_arm.py:81-83`) hardcode
   `--compressor-train-split=train` and `--nde-train-split=train`. The
   audit flag `--require-disjoint-train-examples` was never enabled. So
   the compressor and NDE saw **the same (cosmology, perm, patch)
   triples** during training — 100% example overlap. Effect: NDE sees
   tighter-than-reality summaries on training data and learns to map them
   to θ → systematically over-confident posteriors.

The earlier campaign `cnn_bnt_losslessness_campaign_indep_split_*` (at
`scripts/sbi/results/final/paper_sbi_consolidation/`) DID use the proper
70/30 split with `--require-disjoint-train-examples`. So the precedent
exists in the project; the cnn-auto-push and cnn-auto-cross-push campaigns
reverted to the simpler (broken) setup, probably because the drivers I
edited hardcoded `train/train`.

Andreas's correct framing: the absolute numbers are over-confident, but
internal-to-campaign comparisons (e.g., H1 attention vs plain CNN both on
the contaminated split) still hold because the contamination is shared.
Cross-pipeline absolute comparisons need refreshing for the paper, but the
science findings are not invalidated.

## Primary metric (declared per CLAUDE.md §"Felt / Ralph operating conventions" #1)

**3-seed pooled FoM3 = 1/√det Cov₃(Ωₘ, σ₈, w₀)** on the fiducial cosmology,
3 seeds (41, 42, 43), 100k posterior samples each.

For each arm, report:
- per-seed FoM3
- mean-of-seeds (MoS) FoM3
- 3-seed **pooled** FoM3 (the primary metric)
- pool/MoS haircut (the seed-to-seed mode-drift diagnostic)
- per-parameter bias-vs-truth in σ-units on (Ωₘ, σ₈, w₀)
- global median |bias|

## Decision rule

**This is a measurement campaign, not a hypothesis test** — there is no
pass/fail threshold. We report the four anchors and the corrected
cross/auto ratios; downstream hypothesis re-evaluation happens in
`[[CNN_CROSS_MAPS_INFORMATION_NOTE]]` based on those numbers.

Sanity checks that would flag a problem and trigger a re-investigation:

- Any arm's pooled FoM3 differs from its stale anchor by > 5×: investigate
  the run, not just the numbers (likely indicates a config error in the
  launcher, not real shift).
- Any arm's |bias|med > 2σ: investigate noise model and zero-mean-maps
  application.
- Any arm has NaN posterior samples: hard failure, investigate before
  trusting results.

## Budget

- **Fixed scope**: 4 arms × 3 seeds = 12 runs. No iteration, no plateau-stop.
- **Wall-clock**: ~5h on GPU 1 sole tenant (parallel-3 within each arm,
  sequential between arms — see launch_canonical_all.sh).

## Loop Status (live)

**Iteration 1 (2026-05-24 09:55–13:57 UTC) PARTIAL FAILURE.**
First canonical-refresh run produced 9 of 12 posteriors. Three distinct
problems uncovered (the post-mortem of which justifies the slower-but-clean
Option A path Andreas approved):

1. **CNN auto-only s41 NDE NaN crash.** Val loss went to NaN at step 7000,
   best-val=inf, posterior is garbage (mean w0=+0.5 vs truth −1.0, +33σ
   bias). s42 and s43 ran cleanly. Seed-specific numerical fluke; just
   re-run s41.
2. **CNN auto+cross arm FAILED in 19s (rc=1×3)**. Harmonic-cache route
   structurally rejects both `train[:70%]` slicing syntax AND
   `--require-disjoint-train-examples` audit flag. The canonical 70/30
   split has only ever been used with the TFDS route (e.g.,
   `cnn_bnt_losslessness_campaign_indep_split_*`); never with harmonic
   cache. Filed as ship-blocker
   `[[code-extend-harmonic-slicing]]`.
3. **L1 launcher missing five SNR-related flags** — both L1 arms used
   the script defaults (`l1_min_snr=-10/+10`) instead of matching
   v2_chsigma (`-13/+13` for auto, `-5/+5` for cross, `percentile=1.0`
   mode for empirical cross-channel SNR adaptation). Both L1 arms need
   re-run with the matching flags.

**Andreas-raised concern (filed as
[[sanity-check-auto-channel-tfds-vs-cache]])**: build_full_sphere_cross_cache.py:283
shows the auto channels in the harmonic cache go through an SHT/iSHT
roundtrip with lmax=1024, so they are bandlimited relative to the TFDS
auto channels. Patch Nyquist is ell~720 so the bandlimit is at
sub-Nyquist scales but the maps are NOT byte-identical between the two
data sources. We need a sanity-check arm to bound whether this matters
for the cross/auto FoM3 comparison.

**COMPUTE COMPLETE 2026-05-26 ~06:30 UTC.** All 12 canonical posteriors landed cleanly. L1 cross verifies the v2_chsigma anchor to 0.5%. See **`HANDOFF_CANONICAL_REFRESH.md`** at repo root for the final numbers + ratios. Remaining work: write-up only.

Final pool FoM3: CNN auto=12,873; CNN cross=12,615; L1 auto=12,004; **L1 cross=34,004** (matches v2_chsigma anchor to 0.5%).

Canonical headline ratios: CNN cross/auto=0.98×; L1 cross/auto=2.83×; CNN/L1 at auto=1.07×; **CNN/L1 at auto+cross=0.37× (L1 dominates by 2.7×)**.

TWO new hard rules saved to project memory: `feedback_never_pca_l1.md` and `feedback_l1_cross_must_use_harmonic_route.md`. Pre-flight `flag_diff.py` now has gotcha tripwires for both.

**Iteration 2 status (2026-05-24 ~15:00 UTC, in flight)**:
- ✅ (a) Pre-flight flag-diff tool at `tools/flag_diff.py` — caught the 5
  missing L1 SNR flags + 13 silent default-fallbacks; 4 anchors checked.
- ✅ (b) L1 launchers updated with `--l1-min-snr -13 --l1-max-snr 13
  --l1-min-snr-cross -5 --l1-max-snr-cross 5 --cross-snr-percentile 1.0`.
- ✅ (c) `[[code-extend-harmonic-slicing]]` shipped: `_parse_harmonic_split_slice`,
  extended `_normalize_harmonic_split` + `_list_harmonic_cache_files` for
  TFDS-style slicing notation, new `audit_harmonic_split_overlap`,
  replaced the rejection at npe_cnn_nbody_tomo.py:3249 with a harmonic-
  aware audit branch.
- ✅ Helper smoke-test passed: 6293 cache files → 70/30 slice yields
  4405 + 1888 with **zero file-level overlap** (file naming
  cosmo_NNNNNN_permK.npz confirms file-level = example-level disjointness).
- 🟡 (d) End-to-end CLI smoke test running (background `b3ls3qls1`) —
  confirms the audit branch actually fires through the CLI under canonical
  flags.
- 🔜 (e) After smoke passes: launch full 5-arm campaign
  (CNN auto-only + CNN auto+cross + L1 auto-only + L1 auto+cross + sanity:
  CNN-on-cache-auto-only-slice). 5 arms × 3 seeds = 15 runs, ~5h.

No autonomous-loop iterations should fire until (e) completes.

No autonomous-loop iterations should fire until (a) clears. Cold-read
agents that see this fiber should resume from step (a) of the canonical
work-plan documented in the "Iteration 2 plan" section below.

## Iteration 2 plan (2026-05-24, post-failure)

Approach: **Option A+** (full code extension + sanity check). Approved by
Andreas. Total ~4-5h.

Sequence:

| step | action | est time | gating |
|:--:|:---|:---|:---|
| 1 | Pre-flight flag-diff script: for each of the 4 anchor types, compare every CLI-relevant key in the new launcher to the corresponding posterior's meta.json from the existing best anchor. Surface ANY mismatch. | ~10 min | none |
| 2 | Fix L1 launchers: add `--l1-min-snr -13 --l1-max-snr 13 --l1-min-snr-cross -5 --l1-max-snr-cross 5 --cross-snr-percentile 1.0` (the five missing v2_chsigma flags). | ~5 min | step 1 verdict |
| 3 | Implement `[[code-extend-harmonic-slicing]]`: parse `train[:70%]` syntax in `_normalize_harmonic_split`, slice file list deterministically in `_list_harmonic_cache_files`, add `audit_harmonic_split_overlap`, remove rejection at npe_cnn_nbody_tomo.py:3249. | ~1h | step 1 |
| 4 | Smoke-test harmonic slicing: `--no-train --no-sample` on cache, verify file counts + zero overlap. | ~10 min | step 3 |
| 5 | Re-launch full canonical refresh (all 4 arms × 3 seeds, parallel-3 within arm) + the `[[sanity-check-auto-channel-tfds-vs-cache]]` arm (cache auto-only slice, 3 seeds). | ~4-5h | steps 1-4 |
| 6 | Analysis script + write up METHODOLOGY.md, refresh CNN_CROSS_MAPS_INFORMATION_NOTE.md §1 and §8c.4. | ~30 min | step 5 |
| 7 | Update CLAUDE.md "Felt / Ralph operating conventions" with convention #8 (mandatory 70/30 split discipline). Close this fiber with exit interview. | ~15 min | step 6 |

The pre-flight flag-diff is a one-time tool that pays for itself on
every future campaign; will be saved at
`scripts/sbi/results/exploratory/canonical_anchors_refresh/tools/flag_diff.py`.

## Lessons documented from iteration 1 (for the exit interview)

- **Don't trust launcher flag completeness without a meta-diff.** I checked
  the major flags but missed `l1_min_snr`/`l1_max_snr`. A pre-flight diff
  against the existing best-anchor meta.json would have caught this.
- **Harmonic cache route ≠ TFDS route.** Disjointness audit and slicing
  notation were silently TFDS-only. Should have checked path-specific
  capabilities before promising uniform discipline.
- **Per-seed NaN failures should fail loudly.** s41's NDE NaN'd, was saved
  as "best_val = inf" and produced a 100k-sample garbage posterior that
  passed shape checks. The analysis script needs a NaN/best-val sanity
  check to flag broken runs before computing pooled FoM3.

## Driver and orchestration (CLAUDE.md convention #7)

- **Driver**: shell-orchestrated. Single master script
  `scripts/sbi/results/exploratory/canonical_anchors_refresh/launch_canonical_all.sh`
  runs all 4 arms sequentially (parallel-3 within each arm). No
  autoresearch driver involved — this is a one-shot refresh, not an
  ongoing campaign.
- **GPU policy**: GPU 1 sole tenant, mem fraction 0.30 per parallel
  worker (3 workers → 0.90 total). Per the H1 exit-interview lesson,
  parallel-N on a single A100 works for small instances and fails for
  GPU-saturating ones — at cdim=10 with 4-10 channel input, instances
  should be small enough.
- **Checkpoint policy**: `--compressor-checkpoint-policy best_val`
  (project default for new campaigns, established post-2026-05-19).

## Canonical setup (the spec)

**Dataset**: `NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`
(48 non-overlapping 20deg × 20deg / 160px patches per cosmology). For the
CNN auto+cross arm: the matching harmonic-cross cache at
`scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid`
(built from the same 48-non-overlap projection convention).

**Splits**:
```
--compressor-train-split train[:70%]    # 211,445 examples
--compressor-val-split   test           # 134,400 examples (cosmos 900-1300)
--nde-train-split        train[70%:]    # 90,619 examples (disjoint triples)
--nde-val-split          test           # CNN; L1-harmonic route uses 'val'
--require-disjoint-train-examples       # hard-fail if any triple overlap
```

The 70/30 numbers exactly match the audited
`cnn_bnt_losslessness_campaign_indep_split_*` precedent
(`split_independence_audit.json` shows 0 example-level overlap, all 899
cosmologies shared by theta — which is fine, only example identity matters).

**Per-arm config table**:

| arm | compressor / datavector | NDE | notes |
|:---|:---|:---|:---|
| CNN auto-only  | plain CNN, conv 64/128/256, dense 256, cdim=10, batch 128, LR 5e-4, 60k steps, `--standardize-summary`, best_val ckpt | RealNVP 8 layers, hidden 256, batch 256, 50k steps | matches iter-108-Q6ON-60k recipe (cross-push winner, unbeaten by complex archs in cnn-auto-push) |
| CNN auto+cross | same compressor as above + `--cnn-map-route harmonic --full-sphere-cross-cache <path> --harmonic-cache-regime nobnt --harmonic-normalize-input-channels` | same NDE | matches iter-108-Q6ON-60k recipe |
| L1 auto-only   | `npe_l1norm_cross_jaxili_nbody_tomo.py` with cross_maps **off**, log1p-zscore (clip 5), SNR [-13,13], 4 channels | npe_epochs 50k, batch 256, LR 1e-4 | matches v2_chsigma config; L1 NDE epoch budget standardized to 50k |
| L1 auto+cross  | same script with cross_maps **on**, SNR auto [-13,13] cross [-5,5], 10 channels, `--cross-noise-model channel_empirical_global` | same NDE | matches v2_chsigma config (post-noise-model-fix) |

**Common to all arms**:
- `--zero-mean-maps` (mandatory project rule)
- `--seed` ∈ {41, 42, 43}
- `--npe-samples 100000`
- `--cuda-visible-devices 1`
- `--no-wandb`
- Posteriors: `posteriors/<arm>_canon_s<seed>.npy`
- Logs: `logs/<arm>_canon_s<seed>.log`

## L1 code change (2026-05-24, this campaign)

`scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py` previously hardcoded
`split="train"` for the NDE training data load. Added `--nde-train-split`
flag (default `"train"` for backward-compat). Preprocessing stats (SNR
calibration on line 791, log1p-zscore standardization) continue to use
the full `"train"` split — Andreas's rule: L1 has no learned compressor,
so the standardization stage doesn't need 70/30 discipline; only the NDE
training data must restrict to the smaller subset to match the CNN's NDE
training data quantity for fair comparison.

## Diagnostic outputs (per run + cross-arm)

In addition to the primary metric, the campaign produces an
interpretability-first plot suite via `plot_canonical_diagnostics.py`
(written 2026-05-24 alongside iter-2). Per Andreas's explicit ask:
"interpret the runs, not just see the contours."

**Per-run diagnostics** (15 runs):
1. **Compressor train/val loss curves** (CNN arms only) — spot overfit,
   NaN crashes (e.g. the iter-1 s41 issue), early stopping.
2. **NDE train/val loss curves** (all arms) — spot val-loss spikes that
   broke iter-1 s41 (Best validation loss: inf).
3. **Per-seed corner plot** (`--figure-out` output, already produced).

**Per-arm aggregate plots** (5 arms × 3 seeds):
4. **3-seed overlay corner plot** with truth markers — visualize
   seed-to-seed mode drift.
5. **Bias-vs-truth bar chart** (3 params × 3 seeds per arm) — calibration
   check, cross-check against the
   [[project_resnet_bn_contamination]] signature (shared bias direction
   across seeds = BN contamination).
6. **Pool/MoS haircut** — should be ~0.69-0.85 for plain-CNN auto-only,
   plain-CNN auto+cross was ~0.98 in iter-108-Q6ON-60k.

**L1-specific interpretability plots** (arms 3, 4):
7. **L1 datavector on the fiducial map** — one panel per channel, x =
   feature bin, y = L1 wavelet ℓ₁ value. Shows what the NDE actually
   conditions on at inference.
8. **L1 datavector dispersion across training cosmologies** — 16-84%
   band per feature, shows the variability the NDE has to learn from.
9. **L1 feature-mask coverage** — which bins got dropped (low variance)
   from `l1_jaxili_feature_mask.npz`.
10. **Per-channel SNR distribution for L1 cross** — verify the [-13, 13]
    auto / [-5, 5] cross ranges cover the informative part (no pile-up
    at boundaries that would indicate truncation of informative tails).

**Cross-arm comparison plots**:
11. **FoM3 bar chart with seed scatter** — 5 arms × 3 seeds, headline result.
12. **CNN-vs-L1 contour overlay at auto-only** (arms 1 & 3).
13. **CNN-vs-L1 contour overlay at auto+cross** (arms 2 & 4).
14. **Sanity check: TFDS-auto (arm 1) vs cache-auto (arm 5)** — resolves
    the harmonic-cache bandlimiting concern raised in
    [[sanity-check-auto-channel-tfds-vs-cache]].

All saved as individual PNGs + a single bound PDF
(`canonical_diagnostics.pdf`) for easy review.

**Data inventory** (existing per-run artifacts that the plotter consumes):
- `train/<arm>_canon_s<seed>/.../loss_compressor_{train,test}.npy` (CNN)
- `train/<arm>_canon_s<seed>/.../loss_{train,val}_cnn.npy` (CNN NDE)
- `train/<arm>_canon_s<seed>/l1_{train,val}.npz` (L1 datavectors)
- `train/<arm>_canon_s<seed>/.../l1_jaxili_standardization.npz`
- `train/<arm>_canon_s<seed>/.../l1_jaxili_feature_mask.npz`
- `posteriors/<arm>_canon_s<seed>.npy` (the posterior samples)
- `posteriors/<arm>_canon_s<seed>.meta.json` (config record)

## Sub-fibers

**None planned.** The 4 arms are tightly coupled (same launcher, same
analysis, same paper-table destination). Per `[[FELT_TUTORIAL]]` §5 step 3,
the constitution body itself carries the sub-fiber detail and a single
fiber suffices. If any one arm crashes or produces an anomaly that warrants
an independent investigation, file a sub-fiber at that point.

## Connections

- **Triggers**: Andreas's pushback during the
  `[[CNN_CROSS_MAPS_INFORMATION_NOTE]]` discussion (2026-05-24) — asking
  whether old runs at 10deg might have contaminated the analysis. Audit
  revealed the more serious split-discipline issue.
- **Precedent for the canonical setup**:
  `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign_indep_split_*`
  (cdim10 advanced_arch64_dense256_nostd run with the exact 70/30
  split discipline; see `CNN_BNT_LOSSLESSNESS_RETRAIN_REPORT.md`).
- **Downstream blockers**: `[[cnn-h2-data-limit-scoping-2026-05]]`
  depends on these corrected anchors; its decision rule references the
  CNN auto-only anchor.
- **Closed campaigns whose verdicts remain valid (relative findings)**:
  `[[cnn-h1-inductive-bias-2026-05]]` (H1 attention falsified),
  `[[cnn-h3-summary-dim-cdim100-test-2026-05]]` (H3 cdim=100 falsified),
  `[[cnn-auto-push-18-20-2026]]` (CNN auto-only ceiling, 4 confirmations).
  All used train/train splits; the relative findings are unchanged but
  the absolute numbers are over-confident by some unknown factor that
  this refresh will bound.

## Post-completion deliverables (when compute lands)

1. **3-seed pooled FoM3 analysis** for each arm (single Python script):
   `analyze_canonical_anchors.py` — writes `canonical_anchors_3seed.json`.
2. **`METHODOLOGY.md`** at repo root: the canonical setup specification.
   Includes the 70/30 split rule, the disjointness audit, the standard
   config for each compressor, and the L1 fair-comparison clause. Future
   runs must follow this.
3. **CLAUDE.md update**: add #8 to "Felt / Ralph operating conventions" —
   "All new compressor+NDE runs must use 70/30 disjoint splits with
   `--require-disjoint-train-examples` enforced; deviation requires an
   explicit ship-blocker fiber documenting why."
4. **`CNN_CROSS_MAPS_INFORMATION_NOTE.md` refresh**: §1 (empirical
   baseline table) and §8c.4 (hypothesis table) get the corrected
   numbers, with the stale numbers preserved with a strikethrough so
   the evolution is visible.
5. **Memory entry**: `project_canonical_setup_split_discipline.md`
   capturing the split-discipline rule + the audit findings.
6. **Constitution outcome update**: this fiber gets closed with the
   4 refreshed anchors and the cross/auto ratios in the outcome string.

## Exit interview (CLAUDE.md convention #5 / felt §exit-interview)

To be filled when campaign closes. Template:

- **What worked.**
- **What was wasted effort.**
- **What we'd do differently.**
- **Convention updates proposed for CLAUDE.md.**

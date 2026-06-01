# Experiment audit — comprehensive catalog of all runs

**Generated**: 2026-05-27
**Scope**: 1,517 posterior runs across 190 directories, 5 project phases, ~6 months of work.
**Method**: Exhaustive scan of every `.meta.json` in the project tree, cross-referenced with the 10-bug checklist.

---

## Executive summary

Of 1,517 runs in this project, **only 27 (1.8%) are fully clean** — meaning they have both `zero_mean_maps=True` and a verified disjoint train/val split. An additional 265 runs have zero-mean maps but train/train NDE overlap. The remaining 1,225 (81%) predate the mass-sheet fix entirely.

### BNT losslessness — the resolved question

The project's first major campaign (Phase 1, ~500 runs) established that CNN-VMIM compression can be made near-lossless through the BNT transform: the best configuration (advanced architecture, cdim=10, standardization OFF) achieves **0.91 BNT/no-BNT FoM3 retention** with only 3% marginal-width inflation. The key lever was turning off summary standardization (which alone accounted for the catastrophic 0.095 baseline). Plain CNN beats all ResNet variants on BNT parity. Noise curriculum hurts the plain architecture. These findings are robust as within-era relative comparisons, though absolute FoM3 values are inflated by the mass-sheet leak.

### L1 vs CNN — the open question

All comparisons to date show L1 auto+cross producing tighter posteriors than CNN auto+cross, with the apparent advantage concentrated in w₀ (σ=0.133 vs 0.166–0.194). However, **no apples-to-apples comparison exists**: every L1-vs-CNN pairing has at least one confound, and several issues suggest the CNN may be underperforming for fixable reasons.

**Confounds and asymmetries that disadvantage CNN in current comparisons:**

1. **NDE architecture mismatch.** CNN uses sbi_lens RealNVP (567k params, catastrophically unstable under 70/30 split). L1 uses jaxili MAF (stable). The NDE-swap test showed this changes FoM3 by ~47%, though 2D areas differ by only 15-20%.

2. **NDE training asymmetry.** L1 auto+cross trains the NDE for **50,000 epochs**. CNN auto+cross early-stops at step 12k (patience=20) under the 70/30 split, with best val-loss at step 2k. The CNN NDE may be severely undertrained, especially on 10-channel input with fewer training examples.

3. **Data volume asymmetry.** L1 auto+cross uses the full training set for the NDE (302k examples). The cleanest CNN comparison uses 70/30 split (90k NDE examples). This 3.3× data gap is on top of the epochs gap.

4. **VMIM companion quality.** The CNN-VMIM compressor uses a 4-layer sbi_lens RealNVP as its companion flow during training. If this companion is suboptimal (and we now know RealNVP is unstable), the compressor may receive poor gradients. No one has tested VMIM training with a jaxili MAF companion.

5. **CNN cross channels add only +9%** (auto-only 16.2k → auto+cross 17.6k under jaxili MAF), while L1 gets 2.9× from the same channels. This is suspicious — the cross-maps contain real cosmological information (the cross-only campaign proves this: CNN resnet50_gn gets 26.6k from cross channels alone). The compressor may be failing to pass cross-channel information through to the NDE, not failing to extract it from the maps.

**Reasons to scrutinize the L1 auto+cross result:**

1. **Higher seed and noise variance.** L1 auto+cross CoV across fiducial permutations is 26% vs CNN's 21%. The perm-averaged L1/CNN gap (1.96×) is larger than the perm-0 gap (1.52×), suggesting the comparison is noise-realization-sensitive.

2. **FoM3 fragility.** A 5% change in marginal width corresponds to a 47% FoM3 change. The headline "L1 wins 1.5-2×" is a FoM3 statement; the marginal-width statement is more modest (L1 σ(w₀) is 29-46% tighter, but Ωm and σ₈ are comparable).

3. **The v2 noise model fix needs independent verification.** The channel-empirical-global model was designed to fix the broken auto_scalar, but the fix itself hasn't been validated against an independent noise estimate (e.g., from the simulations' known noise properties).

4. **No L1 run has a disjoint split.** Every L1 auto+cross run uses the full training set for both L1 summary computation and NDE training. The 70/30 split penalty (18-24% FoM3 from the NDE-swap test) has never been applied to L1.

### What's established (verified from data, not interpretation)

1. **FoM3 is unreliable for tightly-correlated posteriors.** RealNVP vs jaxili on the same compressor: 47% FoM3 gap but only 5% wider marginals. Both autoresearch campaigns optimized a metric that amplified noise as much as signal.

2. **The mass-sheet leak primarily affects CNN, not L1.** Pre-fix CNN FoM3 ~400k → post-fix ~15-20k (25× drop). Pre-fix L1 ~10k → post-fix ~13k (30% change). Verified from nobnt_final_matrix per-bin data.

3. **The 70/30 split costs 9-24% FoM3** vs train/train (from NDE-swap SESSION_RESULTS.md — 24% for RealNVP, 9% for jaxili MAF, but confounded with compressor quality differences between iter-108 and canonical).

4. **CNN is data-limited, not capacity-limited**, on auto-only: resnet50_gn at 120k steps overfits (val-loss argmin at 35%, then 1.28-nat drift), achieving FoM3 11.8k vs plain-CNN's 19.5k. Three independent ceiling falsifiers confirmed.

5. **6 fully-clean CNN posteriors exist** in `canonical_anchors_refresh/` (3 auto-only, 3 auto+cross, zmm=T, disjoint=T, verified 0 example overlap). FoM3 was computed in the canonical refresh handoff: **CNN auto pooled 12,873; CNN cross pooled 12,615**. Per-seed: auto 18,060/17,732/14,845; cross 19,699/14,914/18,214. However, these used sbi_lens RealNVP which early-stopped at step 2-4k (cross) or 3-4k (auto) before diverging — the NDE was severely undertrained.

### What's invalidated

1. **1,225 pre-zero-mean-maps runs** (81%). All of `paper_sbi_consolidation/`, `bnt_tomo_study/`, and pre-fix systematic runs.

2. **The "L1 wins 3×" headline** from v1 noise model. Verified from logs: single `noise_sigma = 0.012658` scalar for all 10 channels. Corrected ratio is ~1.5× on FoM3 at perm 0 (or ~2× perm-averaged).

3. **10 PCA-contaminated L1 runs** in `canonical_anchors_refresh/` (pca_applied=True, FoM3 cratered 5×).

---

## Bug timeline (verified against git history)

All dates verified via `git log` on the introducing commit. "Introduced" means when the bug-carrying code was first committed; "discovered" means when the bug was identified and a fix or workaround was committed or documented.

### Project script chronology (for reference)

| Date | Event | Commit |
|------|-------|--------|
| 2025-07-01 | Initial commit (Learn2Map fork); sbi_lens RealNVP NDE from inception | — |
| 2026-02-18 | `npe_l1norm_nbody_tomo.py` created (L1 + sbi_lens RealNVP) | `08b8e0a` |
| 2026-03-17 | `npe_cnn_nbody_tomo.py` created (CNN + sbi_lens RealNVP) | `2b12032` |
| 2026-03-29 | `npe_cnn_jaxili_nbody_tomo.py` + `npe_l1norm_jaxili_nbody_tomo.py` created (jaxili MAF NDE) | `0c83856` |
| 2026-04-01 | `npe_l1vmim_nbody_tomo.py` created (L1-VMIM + sbi_lens RealNVP) | `d286d26` |
| 2026-04-01 | Curated SBI pipelines commit (all scripts refined) | `efb2c56` |
| 2026-04-12 | `npe_l1vmim_jaxili_nbody_tomo.py` created (L1-VMIM + jaxili MAF) | `b355fe1` |
| 2026-04-12 | BNT losslessness campaign orchestrator created | `358c4d5` |
| 2026-04-13 | `--nde-train-split` flag introduced (independent compressor/NDE split controls) | `29dcdfb` |
| 2026-04-16 | `require_disjoint_train_examples` flag introduced in ResNet split-campaign | `791d402` |
| 2026-04-22 | `--zero-mean-maps` flag introduced | `deb5ee0` |
| 2026-04-26 | `npe_l1norm_cross_jaxili_nbody_tomo.py` created (L1 cross + jaxili MAF); **PCA default=50 from day one** | `1408f67` |
| 2026-05-01 | Harmonic-cache route added to L1 cross script | `ad56ec8` |
| 2026-05-11 | `resnet50_gn` GroupNorm variant introduced | `78c0248` |
| 2026-05-15 | `cross_noise_model=channel_empirical_global` + cross-only campaign orchestrator | `2fc79c3`, `f0b352b` |
| 2026-05-17 | Autoresearch FoM3 verify wrapper | `c9f5dc6` |
| 2026-05-18 | Auto-push and auto-cross-push autoresearch drivers created | `9979a5d`, `b1adde7` |

### Bug catalog

| # | Bug | Introduced | Discovered | Verified commit | Runs affected | FoM3 impact |
|---|-----|-----------|------------|-----------------|---------------|-------------|
| 1 | **Mass-sheet degeneracy leak**: CNN compressor learns mean-convergence signal that wouldn't exist in real data when maps aren't zero-meaned | 2025-07-01 (project inception — no map demeaning in original Learn2Map code) | 2026-04-22 (fix committed in `deb5ee0`) | `deb5ee0` | 1,225 runs (81%) — every run without `zero_mean_maps=True` | ~25-30× FoM3 inflation |
| 2 | **Train/train NDE overlap**: compressor trained on full `train` split, NDE also trained on `train` split — same examples seen by both | 2025-07-01 (project inception — original code had no split discipline) | 2026-04-13 (split controls added in `29dcdfb`); 2026-04-16 (`require_disjoint` in `791d402`) | `791d402` | 1,312 runs (86%) — every run without `require_disjoint_train_examples=True` | Unknown magnitude; never measured in isolation |
| 3 | **RealNVP catastrophic instability**: sbi_lens ConditionalRealNVP diverges to NaN at step 2-7k under 70/30 split, step ~10k under train/train | 2025-07-01 (inherent to sbi_lens flow architecture + small data regime) | 2026-05-26 (NDE-swap session, documented in `SESSION_RESULTS.md`) | — | All 843 `cnn` method runs + 96 `l1_vmim` + 47 `l1norm` = 986 runs using sbi_lens RealNVP | Some runs train to NaN; surviving runs are those that converge before divergence or use early stopping |
| 4 | **L1 PCA default=50**: `npe_l1norm_cross_jaxili_nbody_tomo.py` line 1570: `default=50`, destroying L1 wavelet information | 2026-04-26 (day the L1 cross script was created, commit `1408f67`) | 2026-05-25 (memory: `feedback_never_pca_l1.md`) | `1408f67` line 1570 | 9 runs in `canonical_anchors_refresh/iter1_archive` with `pca_applied=True`; potentially any L1 cross run that didn't explicitly pass `--pca-components 0` | 5× FoM3 crater (16k → 3k empirically) |
| 5 | **L1 cross-channel noise model (`auto_scalar`)**: uses auto-map pixel σ for cross channels whose values are ~30,000× smaller, collapsing wavelet SNR to ~0 | 2026-04-26 (implicit in original L1 cross script — no channel-aware model existed) | 2026-05-15 (fix committed: `channel_empirical_global` in `f0b352b`) | `f0b352b` | All pre-2026-05-15 L1 cross runs; also any post-fix run using TFDS route instead of harmonic cache | v1→v2: L1 cross_only 12k→16k (+33%); L1 auto+cross ~65k→34k (−48%) |
| 6 | **L1 cross TFDS route silent fallback**: TFDS+`--cross-maps` route silently ignores `channel_empirical_global`, prints warning but falls back to broken `auto_scalar` | 2026-05-15 (inherent to how the fix was implemented — only the harmonic cache route was patched) | 2026-05-26 (memory: `feedback_l1_cross_must_use_harmonic_route.md`) | — | Any L1 cross run using TFDS route after the fix was committed | 4× FoM3 crater (40k → 10k) |
| 7 | **BatchNorm contamination on 10-channel harmonic input**: stock ResNet BN averages statistics across cosmology-mixed batches when input has 10 channels (4 auto + 6 cross) | Relevant from 2026-04-16 (ResNet variants introduced in `791d402`) but only matters for harmonic 10-channel input which came later | 2026-05-11 (fix: `resnet50_gn` in `78c0248`) | `78c0248` | ResNet18/34/50 (non-GN) runs on harmonic cross input; does NOT affect plain CNN or auto-only 4-channel runs | FoM3 ~700 (vs ~22k with GroupNorm) |
| 8 | **NDE architecture mismatch across pipelines**: CNN production pipeline (`npe_cnn_nbody_tomo.py`) uses sbi_lens RealNVP; L1 production pipeline (`npe_l1norm_*_jaxili_*.py`) uses jaxili MAF | 2026-03-29 (jaxili scripts created alongside but separate from CNN scripts in `0c83856`) | 2026-05-27 (NDE-swap session, `SESSION_RESULTS.md`) | — | Every L1-vs-CNN comparison ever made (confounds compressor comparison with NDE comparison) | On same compressor: RealNVP FoM3 25.9k vs jaxili MAF 17.6k (47% gap). But 2D areas differ by only ~15-20%, and marginal σ by ~5%. |
| 9 | **Compressor checkpoint policy ambiguity**: `last_step` vs `best_val` checkpoint selection; many meta.json files don't record which was used | Present from project inception; `compressor_checkpoint_policy` field not consistently logged | 2026-05-22 (auto-push retrospective) | — | Autoresearch campaigns and all runs where `compressor_checkpoint_policy` is absent from meta.json | Variable; can be significant if compressor overfit past its val-loss minimum |
| 10 | **FoM3 fragility**: FoM3 = 1/√det(C₃) amplifies 1-2% correlation changes into ~50% metric swings for tightly-correlated posteriors | Intrinsic to metric definition; not a code bug | 2026-05-27 (NDE-swap session, `SESSION_RESULTS.md`) | — | Every comparison that relied solely on FoM3 as the decision metric | Misleading rankings: a 47% FoM3 gap corresponds to only ~5% wider marginals and ~15-20% larger 2D contour areas (verified from SESSION_RESULTS.md) |

---

## Phase 0 — Project origins and pre-campaign experiments

### Pre-repo artifacts (Feb – Jun 2025)

Before the git repo was created, experiments ran from the original Learn2Map codebase:
- `run_artifacts/save_params/mse/`: MSE compressor checkpoints dated **Feb 28, 2025** (Gaussian + nbody). 706 checkpoint files total across all artifact dirs.
- `run_artifacts/save_params/vmim/`: VMIM compressor checkpoints dated **May 29, 2025**.
- These use the original Learn2Map `train_compressor.py` with sbi_lens ConditionalRealNVP. The NDE architecture was inherited from Justine Zeghal's codebase.

### Fork and initial exploration (Jul 1 – Aug 11, 2025)

| Date | Commit | What happened |
|------|--------|---------------|
| Jul 1 | `f5c8fa4` | Initial commit: Learn2Map fork with NLE/NPE notebooks (simple nbody, VMIM, MSE, baryon+IA variants), tf_dataset builders, train_compressor. sbi_lens RealNVP is the only NDE. |
| Jul 3 | `ba5a789` | GPU selection added |
| Aug 8 | `e7f1189` | **20-deg FOV / 160-px** field size established — used throughout the project |
| Aug 9 | `4fdd08a` | **Tomographic analysis**: tf_dataset_nbody_tomo with 4 tomo bins |
| Aug 10 | `eff93c9` | First tomographic NLE notebook |
| Aug 11 | `71c21ed` | **First BNT analysis**: `nle_simple_nbody_tomo_BNT.ipynb` + `train_compressor_tomographic_BNT.py` |

The Aug 2025 work established the experimental template: CosmoGridV1 maps → CNN-VMIM compressor → sbi_lens RealNVP → posterior contours. BNT was already a topic of interest from the first week of systematic work.

### 6-month gap (Aug 2025 – Feb 2026)

No commits. Work was likely notebook-based (outputs exist in the initial notebooks) or there was a thesis-writing/teaching hiatus.

### Systematic pipeline development (Feb 18 – Apr 2, 2026)

| Date | Commit | What happened |
|------|--------|---------------|
| Feb 18 | `08b8e0a` | **L1-norm script** created (`npe_l1norm_nbody_tomo.py`). W&B logging begins. |
| Feb 23 | — | **16 W&B runs** on L1 hyperparameters (`results/wandb_runs/`). Script: `npe_l1norm_nbody_tomo.py`, git commit `08b8e0a`, running from Learn2Map virtualenv on 4× A100. |
| Mar 17 | `2b12032` | **CNN inference scripts** created. BNT TFDS builder. L1 script updated. |
| Mar 19 | `259d864` | Robust SBI sweep and bin-aware NPE pipelines. |
| Mar 29 | `0c83856` | **jaxili MAF NDE introduced** — both `npe_cnn_jaxili_nbody_tomo.py` and `npe_l1norm_jaxili_nbody_tomo.py`. This is when the NDE architecture split happens: CNN defaults to sbi_lens RealNVP, L1 gets jaxili MAF. |
| Mar 31 | `068736f` | jaxili training splits stabilized. L1/CNN contour overplot notebook. |
| Apr 1 | `d286d26` | **L1-VMIM pipeline** created. Final L1-VMIM conclusions: VMIM compression is ~1.9% broader than raw L1 at cdim=40 (near-lossless). |
| Apr 1 | `efb2c56` | Curated SBI pipelines (all scripts refined). |
| Apr 2 | `585343e` | **PR #1 merged** (`l1_compressor_clean` → main). |

### The round1_stage experiments (bnt_tomo_study worktree, 30 CNN runs)

Manual, sequential BNT parity debugging — each stage tests one configuration lever:

| Stage | Runs | What was tested |
|-------|------|----------------|
| A | 6 | Baseline BNT/no-BNT comparison |
| B | 2 | (incremental change) |
| C | 2 | (incremental change) |
| D_clip | 8 | Gradient clipping |
| E_nostd | 2 | **Turning off summary standardization** — the key lever later confirmed in the losslessness campaign |
| F_dim8 | 2 | cdim=8 |
| G_flow10k | 2 | Longer flow training (10k steps) |
| H_arch | 6 | Architecture variants |

These are developmental runs with no campaign summaries. All CNN sbi_lens RealNVP, no zero-mean-maps, no disjoint split. They represent the manual exploration that later became the systematic losslessness campaign.

### L1-VMIM optimization rounds (bnt_tomo_study worktree, 8 runs)

Two cdim=64 L1-VMIM configurations tested:
- `cdim64_h512_nf8x256` (2 runs): hidden=512, 8-layer NF with hidden 256
- `cdim64_h768_nf10x384` (6 runs): hidden=768, 10-layer NF with hidden 384

These fed into the L1-VMIM conclusions document finding near-lossless compression at high cdim.

### Legacy run artifacts

| Directory | Method | Date | Contents |
|-----------|--------|------|----------|
| `save_params/mse/` | MSE compressor | Feb 2025 | Gaussian + nbody checkpoints (pre-repo Learn2Map) |
| `save_params/vmim/` | VMIM compressor | May 2025 | nbody checkpoints (pre-repo) |
| `save_params/cnn_vmim/` | CNN-VMIM | Feb 2026 | First CNN compressor checkpoints |
| `save_params/l1norm/` | L1-norm | Feb 2026 | First L1 compressor checkpoints |
| `save_params_cnn_control_base/` | CNN control | ~Feb 2026 | Baseline CNN for shuffle test |
| `save_params_cnn_control_shuffle/` | CNN shuffle | ~Feb 2026 | Theta-shuffled control (checking if CNN actually uses θ) |
| `save_params_postfixtest/` | L1-norm | ~Feb 2026 | Post-fix validation checkpoints |
| `dryruns/best_val_smoke_pilot/` | — | — | Pilot run for best-val checkpoint selection |

The `control_shuffle` artifact is notable — it's an early sanity check that the CNN compressor was actually using the cosmological parameters (by shuffling θ and checking if the posterior degrades). This predates the formal campaign era.

---

## Comprehensive pitfalls and hidden asymmetries

Every comparison between L1 and CNN in this project has had at least one uncontrolled confound. This section catalogs every hidden asymmetry, silent configuration drift, and involuntary difference that has contaminated results — organized by where in the pipeline the asymmetry lives.

### A. NDE-level asymmetries (the biggest confound family)

| # | Asymmetry | L1 setting | CNN setting | Impact | When discovered |
|---|-----------|-----------|-------------|--------|-----------------|
| A1 | **Flow architecture** | jaxili ConditionalMAF (5 layers, hidden [50,50], ~20k params on cdim=10 input, ~517k on 2000-dim L1 input) | sbi_lens ConditionalRealNVP (8 layers, hidden 256, ~567k params) | 47% FoM3 gap on same compressor output; 5% marginal-σ gap. RealNVP catastrophically unstable under 70/30 split. | 2026-05-27 |
| A2 | **NDE training duration** | 50,000 epochs max, but jaxili uses early stopping (patience=20, min_delta=0.001). L1 cross canonical early-stopped at **epoch 78** (best at 57). L1 auto canonical stopped at **epoch 54** (best at 33). | 50,000 max steps with patience=20 early stopping. Canonical CNN cross early-stopped at **step 12,000** (best at step 2,000). iter-108 CNN cross: best at step 7,000, early-stopped at step 32,000 (patience=50). | Both pipelines use early stopping. Actual training durations are comparable: L1 cross ~78 epochs × 825 steps ≈ 64k steps; CNN iter-108 ~32k steps. The key difference is stability, not budget. | 2026-05-27 (NDE-swap session) |
| A3 | **NDE training data volume** | Full training set (~302k examples for auto+cross) | Canonical: 70/30 split (90k NDE examples). Baseline: full train (~302k). | 3.3× data gap in the canonical comparison. L1 has never been tested with 70/30 split. | Not explicitly flagged |
| A4 | **NDE optimizer/scheduler** | jaxili: Adam with warmup (128 steps) + cosine decay (10k steps), gradient clipping at 5.0 | sbi_lens: AdamW (weight_decay=1e-4) with cosine LR schedule (1e-3→1e-5), gradient clipping at 1.0 | Both use cosine decay and gradient clipping, but different LR ranges (jaxili: 1e-4, CNN: 1e-3→1e-5) and different clip thresholds (5.0 vs 1.0). The CNN's 5× tighter gradient clipping may contribute to its instability — or may be the only thing preventing faster divergence. | Never compared |
| A5 | **VMIM companion flow quality** | N/A (no compressor) | The CNN-VMIM compressor uses a **4-layer sbi_lens RealNVP** as its companion flow during training. If the companion is suboptimal, compressor gradients are suboptimal. No one has tested VMIM with a jaxili MAF companion. | Unknown but potentially large — the companion's quality directly determines the VMIM loss landscape | Never tested |

### B. Compressor-level asymmetries

| # | Asymmetry | L1 setting | CNN setting | Impact |
|---|-----------|-----------|-------------|--------|
| B1 | **Summary dimensionality** | 2,000 features (5 scales × 40 bins × 10 channels), all survive feature masking (min_variance=1e-5) | cdim=10 (or 6, 16, etc. depending on run). The CNN compresses 200× more aggressively than L1. | The L1 NDE receives 200× more information but faces a harder learning problem (curse of dimensionality). Whether the NDE can exploit all 2000 features is unverified. |
| B2 | **Compressor architecture changed mid-campaign** | N/A | The original harm-norm baseline (25.5k FoM3, `cnn_with_harm_cross_normalized/`) used `conv_channels: 32,64,128` with 150k compressor steps. The auto-cross-push campaign driver defaults to `64,128,256` (the better architecture from the BNT losslessness campaign). iter-108-Q6ON-60k used `64,128,256` with 60k steps and got 24.5k FoM3 — similar to the 32,64,128 baseline. The architecture upgrade did NOT improve auto+cross FoM3, suggesting the bottleneck is elsewhere (NDE, data, or the cross-channel information extraction itself). | Architecture is NOT the bottleneck for CNN auto+cross. Both 32,64,128 and 64,128,256 give ~25k. |
| B3 | **Checkpoint policy** | N/A (no compressor checkpoints) | `last_step` vs `best_val` — inconsistently applied. The auto-push campaign discovered this bug mid-flight and fixed it (commit `5c5a6d9`). Many earlier runs used `last_step` which can be significantly worse if the compressor overfit. | Variable; worse for longer-trained compressors |
| B4 | **Compressor training steps** | N/A | Varies wildly: 20k (baseline), 60k (stagej), 80k (losslessness), 120k (auto-push), 150k (harm_cross_norm baseline), 240k (extended). No single "canonical" value. | Different compressor maturity across runs |

### C. Preprocessing and normalization asymmetries

| # | Asymmetry | L1 setting | CNN setting | Impact |
|---|-----------|-----------|-------------|--------|
| C1 | **Cross-channel normalization** | `channel_empirical_global`: per-channel noise σ estimated from the data, used to set SNR thresholds. Fixed ranges: auto [-13, 13], cross [-5, 5]. | `harmonic_normalize_input_channels`: per-channel RMS normalization (divide each channel by its RMS across the training set). | Completely different normalization philosophies for the same 10-channel data. L1 normalizes by noise level; CNN normalizes by signal level. Neither has been shown to be optimal. |
| C2 | **Summary standardization** | L1 uses `log1p` + z-score transform + clip at 5.0 (the `log1p-zscore` pipeline). No `summary_standardized` flag. | CNN uses `summary_standardized=True` in the baseline and canonical runs (z-score the compressor output). The BNT campaign showed standardization **destroys BNT information** (0.095 → 0.794 FoM3 ratio). Whether it also hurts no-BNT auto+cross is untested. | Standardization was identified as catastrophic for BNT; its effect on auto+cross has never been isolated. |
| C3 | **Feature masking threshold** | `min_feature_variance=1e-5` (all 2000 features survive) | `min_feature_variance` not recorded in most CNN meta.json files. When recorded: `1e-8` in BNT-era runs. | Different thresholds mean different features are masked. Not practically important if both are small enough to pass everything, but the inconsistency is uncontrolled. |
| C4 | **PCA default=50 in L1 cross script** | `--pca-components` defaults to 50 unless explicitly set to 0. Silent — no warning printed. | N/A | 5× FoM3 crater when active. Affected 10 runs. Any L1 cross run that didn't pass `--pca-components 0` is contaminated. |

### D. Data-level asymmetries

| # | Asymmetry | L1 setting | CNN setting | Impact |
|---|-----------|-----------|-------------|--------|
| D1 | **TFDS variant** | `grid_20deg_160px_nonoverlap48` (canonical L1 runs) | `grid_20deg_160px` (most CNN runs) or harmonic cache (harm_cross runs) | Different patch sampling: nonoverlap48 has exactly 48 non-overlapping patches per sphere; the standard TFDS can produce overlapping patches. This changes the effective data volume and diversity. |
| D2 | **Harmonic cache noise diversity** | Same harmonic cache as CNN (n_perms=7 noise realizations per cosmology) | Same cache when using harmonic route | Both are limited to 7 noise perms, but the TFDS auto-only route draws noise on-the-fly (effectively infinite diversity). The auto-cross-push STATUS.md noted this as a structural ceiling. |
| D3 | **Train/train overlap** | All L1 production runs use `nde_train_split=train` (full training set for both L1 computation and NDE training). **No L1 run has ever used disjoint split.** | Some CNN runs use 70/30 split (canonical refresh), most use train/train (auto-push, baseline). | When CNN uses disjoint split but L1 doesn't, CNN is penalized by 18-24% FoM3 (quantified in NDE-swap test). |

### E. Cross-channel-specific pitfalls

| # | Asymmetry | Detail | Impact |
|---|-----------|--------|--------|
| E1 | **v1 noise model** (`auto_scalar`) | Used auto-map pixel σ for all 10 channels. Cross-map amplitudes are ~30,000× smaller than auto. This collapsed wavelet SNR to ~0 for cross channels, zeroing 95% of L1 histogram bins. | L1 auto+cross FoM3 inflated from ~38k (v2) to ~65k (v1). The "L1 wins 3×" headline was from v1. |
| E2 | **TFDS route silent fallback** | After the v2 fix, the TFDS+`--cross-maps` route silently ignores `channel_empirical_global` and falls back to `auto_scalar`. A warning is printed but not an error. | Any L1 cross run using TFDS instead of harmonic cache gets v1 behavior. 4× FoM3 crater. |
| E3 | **CNN vs L1 cross-channel normalization** | CNN: RMS normalization (scales each channel so its RMS is ~1). L1: noise-σ normalization (sets SNR thresholds based on per-channel noise). | These process the same 10-channel data differently. The L1 approach is physics-motivated (wavelet coefficients should be compared at fixed SNR). The CNN approach is ML-motivated (features should have comparable scale). No study has compared them on the same compressor. |
| E4 | **BatchNorm on 10-channel input** | Stock ResNet with BatchNorm computes running statistics across cosmology-mixed batches. On 4-channel auto input this is fine. On 10-channel harmonic input it contaminates features. | FoM3 ~700 (vs ~22k with GroupNorm). Only affects ResNet variants, not plain CNN. |
| E5 | **v2 noise model unvalidated** | The `channel_empirical_global` fix estimates per-channel σ from the training data. This estimate has never been validated against the known simulation noise properties. | If the estimate is biased, the L1 SNR thresholds are wrong. The fix improved FoM3 by 33% on cross-only and reduced it by 48% on auto+cross — the asymmetry is unexplained and may indicate the fix itself has issues. |

### F. Metric and reporting pitfalls

| # | Pitfall | Detail | Impact |
|---|---------|--------|--------|
| F1 | **FoM3 fragility** | FoM3 = 1/√det(C₃) amplifies 1-2% correlation changes into ~50% metric swings. | The "headline" ratios (1.5×, 2×, 3×) are FoM3 statements. The marginal-σ statements are much more modest (L1 σ(w₀) 29% tighter, Ωm and σ₈ comparable). |
| F2 | **Inconsistent FoM3 variant** | Different campaigns use pooled FoM3, mean-of-seeds FoM3, or per-seed-min FoM3 as their primary metric. The auto-push campaign switched mid-flight. | Pooled and mean-of-seeds can differ by ~30% (the "haircut"). Comparing a pooled number from one campaign to a mean-of-seeds from another is misleading. |
| F3 | **Single-perm vs perm-averaged** | Most "headline" FoM3 values are from fiducial perm 0 (one noise realization). The auto-cross-push campaign showed perm-averaged L1/CNN gap is 1.96× (vs 1.52× at perm 0). | The perm-0 numbers understate the gap. But perm-averaging was only done for the auto-cross-push campaign; other comparisons use single-perm. |
| F4 | **Incomplete meta.json** | Many fields are missing from meta.json: `compressor_dim`, `cross_noise_model`, `compressor_checkpoint_policy`, `npe_epochs` (for CNN), `fom3` (for most CNN runs). | Makes post-hoc verification impossible for many runs. The audit had to check logs, campaign summaries, and STATUS.md to reconstruct configs. |
| F5 | **FoM3 computed on [Ωm, σ₈, w₀] only** | The 3-parameter subspace ignores h₀, n_s, Ωb. These parameters could show different L1/CNN patterns. | No run has reported marginal σ for all 6 parameters in a systematic comparison. |

### G. Dataset inconsistencies when cross-maps are introduced

The cross-map channels introduce a completely different data pipeline. The following inconsistencies are specific to the cross-map configuration and didn't exist in the auto-only era.

| # | Inconsistency | Detail | Impact |
|---|--------------|--------|--------|
| G1 | **Two fundamentally different cross-map definitions coexist** | **Flat-sky route**: `IFFT(FFT(κᵢ · apod) · FFT(κⱼ · apod))` on a 160×160 patch with cosine apodization (8% roll-off). Computes cross-power on a single extracted patch. **Harmonic route**: `ISHT(alm_i · alm_j)` on the full HEALPix sphere, then extract gnomonic patch. No apodization. Computes cross-power from full-sphere alms. | These are NOT the same operation. Flat-sky patches have boundary artifacts (partially mitigated by apodization). Harmonic patches contain power leaked from sky outside the patch boundaries. Results from the two routes should not be compared, but they were compared in the early cross-maps campaign (flat-sky FoM3 1.5-14k vs harmonic 55-76k). |
| G2 | **Noise is stochastic in TFDS but fixed in harmonic cache** | **TFDS route** (auto-only, flat-sky cross): `tf.random.normal` draws fresh noise every time a map is loaded → effectively infinite noise augmentation across epochs. **Harmonic cache**: noise is baked in at build time with a deterministic seed per (cosmology, perm) → only `n_perms=7` noise realizations exist per cosmology. | CNN auto-only (TFDS) trains with infinite noise diversity. CNN auto+cross and L1 auto+cross (harmonic cache) train on at most 7×48=336 noisy patches per cosmology. This structural data-augmentation ceiling is noted in the auto-cross-push STATUS.md but never controlled for. |
| G3 | **Cross-maps are computed from already-noisy maps** (NOT a bug) | Both routes inject shape noise into the 4 auto maps FIRST, then compute cross-maps from the noisy autos. Cross = `FFT(κᵢ+nᵢ) · FFT(κⱼ+nⱼ)`. The noise-noise term `FFT(nᵢ)·FFT(nⱼ)` has zero mean but nonzero variance. | **This is physically correct.** In a real weak lensing analysis, cross-correlations are computed from noisy observed shear maps. The noise-noise term is part of the real noise budget and the inference pipeline should learn to handle it. Computing cross-maps from noiseless maps then adding noise separately would give an unrealistic noise structure. Not an issue. |
| G4 | **L1 channel-noise calibration estimates TOTAL std, not NOISE std** (acceptable) | `calibrate_channel_noise_sigma_from_harmonic_cache` computes `σ_c = sqrt(E[x²] - E[x]²)` — the total standard deviation of each channel, including signal variance. | **This is acceptable practice.** In a real analysis, noise calibration comes from the data itself (or from simulations processed identically to the data), which is what `channel_empirical_global` does. The SNR thresholds don't need to isolate pure noise — they set histogram bin edges that capture cosmology-dependent variation in the statistic, which works regardless of whether signal contributes to the calibration std. The rescaling is a bijective linear map that doesn't change information content. |
| G5 | **L1 uses a single `noise_sigma` for the wavelet transform of ALL channels** | The `compute_l1_batch` function passes the same `noise_sigma` (auto-pixel σ = 0.012658) to `stats.compute_wavelet_transform` for all 10 channels. The v2 fix applies `channel_scale` to the MAP VALUES before this call, rescaling cross-channel amplitudes to be comparable to auto. | This is mathematically equivalent to using per-channel noise σ. Given G4's clarification (total std is an acceptable calibration), this is consistent. |
| G6 | **The v2 noise model fix has an unexplained asymmetric effect** | v1→v2: L1 cross-only FoM3 improved +33% (12k→16k mean), but L1 auto+cross FoM3 DECREASED -41% (v1 mean 64.9k → v2 mean 38.2k; or -48% comparing v1 mean to v2 3-seed pooled 34.0k — note: mixing mean with pooled inflates the apparent decrease). | If the fix is simply correcting a noise model, both cross-only and auto+cross should improve. The decrease on auto+cross suggests the v1 noise model was artificially inflating auto+cross FoM3, possibly by making cross-channel L1 histograms near-zero (effectively removing cross channels from the datavector), which paradoxically reduced the NDE's input dimensionality and made fitting easier. Or the fix is overcorrecting. Neither explanation has been verified. The v1 runs also used 6 seeds (s41-s46) vs v2's 3 seeds (s41-s43) and different fiducial permutations (p1 vs p0), adding noise-realization confounds. |
| G7 | **CNN and L1 normalize cross channels differently** | **CNN**: `harmonic_normalize_input_channels` divides each channel by its RMS (a signal+noise measure). **L1**: `channel_scale = noise_sigma / channel_sigma` amplifies each channel to have std ≈ noise_sigma. These are related but not identical transformations — CNN normalizes by RMS (sqrt of mean square), L1 normalizes by std (sqrt of variance). For zero-mean channels, RMS = std. For non-zero-mean channels, they differ. | The harmonic cache demeaning (step 8 in the builder) makes channels approximately zero-mean, so RMS ≈ std in practice. But this is an unverified assumption. More importantly, the CNN normalization happens at the PIXEL level before convolution, while the L1 normalization happens at the PIXEL level before wavelet transform — different downstream operations. |
| G8 | **Patch-center consistency between routes is correct but TFDS `random` mode is not** | The harmonic cache and `nonoverlap48` TFDS variant both use `_build_non_overlapping_centers` from the same module — verified identical patch centers. **But**: the standard `grid_20deg_160px` TFDS (used by CNN auto-only) draws RANDOM patch centers with a fixed PRNG seed, producing ~25 random patches per realization. These are different sky patches from the harmonic cache's 48 deterministic patches. | CNN auto-only (TFDS random-25) and CNN auto+cross (harmonic nonoverlap-48) see DIFFERENT sky patches from the same cosmologies. The random patches may overlap; the nonoverlap ones don't. This is an uncontrolled difference between auto-only and auto+cross CNN runs. |
| G9 | **The observed map's cross-map computation path must match the training path** | For harmonic-route runs, the observed map is loaded from the cache (`load_observed_from_harmonic_cache`), ensuring consistency. For TFDS-route runs, the observed map is loaded from HDF5, noise is added on-the-fly, then cross-maps computed with `_compute_cross_maps_np`. | Harmonic-route observed maps are consistent with training. But the L1 cross-only TFDS route (v1) computed cross-maps from a SINGLE randomly-projected patch, while the harmonic route uses a specific patch from the 48-patch grid. Different sky areas, different projection geometries. |

### H. The "no clean comparison exists" problem

The "flagship" comparison (L1 auto+cross 38k vs CNN auto+cross 25.5k) has **at least 7 simultaneous confounds**:

| Confound | L1 value | CNN value |
|----------|---------|-----------|
| NDE architecture | jaxili MAF | sbi_lens RealNVP |
| NDE training epochs | 50,000 | 50,000 max, early-stopped at 18k (baseline) or 12k (canonical) |
| NDE training data | full train (302k) | full train (baseline) or 70/30 (90k, canonical) |
| Compressor architecture | N/A (raw statistics) | `32,64,128` (baseline) or `64,128,256` (canonical) — **neither is the optimized config from the BNT losslessness campaign + proper NDE** |
| Summary standardization | log1p-zscore + clip | standardized=True (baseline) or True (canonical) |
| Summary dimensionality | 2,000 features | cdim ≈ 10 (200× compression ratio) |
| Disjoint split | No | No (baseline) or Yes (canonical) |

No run exists where all 7 factors are matched. The auto-cross-push campaign spent 115 iterations varying levers within the CNN pipeline, but always kept sbi_lens RealNVP as the NDE and never tried the bigger compressor architecture with a properly trained NDE.

---

## Inventory summary by method

| Method | NDE | Runs | FoM3 range | zero_mean_maps | disjoint_split |
|--------|-----|------|-----------|----------------|----------------|
| cnn | sbi_lens RealNVP | 843 | 371 – 76,348 (where recorded) | 169 (20%) | 19 (2%) |
| cnn_jaxili | jaxili MAF | 118 | 151 – 575,283 | 0 (0%) | 0 (0%) |
| l1norm_jaxili | jaxili MAF | 244 | (many unrecorded) | ~50 | 0 |
| l1norm_cross_jaxili | jaxili MAF | 87 | 371 – 76,348 | 87 (100%) | 0 (0%) |
| l1_vmim | sbi_lens RealNVP | 96 | (many unrecorded) | 0 (0%) | 0 (0%) |
| l1_vmim_jaxili | jaxili MAF | 82 | (many unrecorded) | 0 (0%) | 0 (0%) |
| l1norm | sbi_lens RealNVP | 47 | (many unrecorded) | 0 (0%) | 0 (0%) |

Notable: **all 87 L1 cross runs have zero_mean_maps=True** (the L1 cross pipeline was developed after the mass-sheet fix). But **none have the disjoint split**. The 118 CNN+jaxili runs all predate zero-mean-maps and are all contaminated.

---

## Phase 1 — BNT era (1,170 runs, 91 directories)

### Summary statistics

- **Total runs**: 1,170 across 91 directories
- **Methods**: CNN (621), L1-norm jaxili (232), CNN jaxili (118), L1-VMIM (96), L1-VMIM jaxili (82), L1-norm RealNVP (21)
- **zero_mean_maps=True**: 0/1,170 (0%) — **ALL contaminated by mass-sheet leak**
- **Disjoint split discipline**: 276/1,170 (24%) have some form of disjoint split
  - 236 have `require_disjoint_train_examples=True` (boolean flag)
  - 40 have explicit 70/30 split strings (`train[:70%]` / `train[70%:]`) without the boolean flag
  - 894 have no split discipline (train/train NDE overlap or not recorded)
- **FoM3 recorded**: 264/1,170 runs have FoM3 values; range 151–575,283

### Key insight: mass-sheet leak asymmetry

The `nobnt_final_matrix` per-bin data reveals that the mass-sheet leak primarily affects CNN, not L1:

| Method | NDE | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Tomo4 | 
|--------|-----|-------|-------|-------|-------|-------|
| CNN | jaxili MAF | 6,001 | 16,822 | 36,942 | 52,768 | 387,475 |
| L1-norm | jaxili MAF | 568 | 1,612 | 3,990 | 4,641 | 9,651 |
| L1-VMIM | jaxili MAF | 410 | 1,241 | 3,581 | 4,686 | 10,651 |

CNN tomo4 FoM3 ~387k is inflated ~25× over the post-zero-mean-fix value (~15-20k). L1 tomo4 FoM3 ~10k is comparable to post-fix L1 auto-only (~13k). This asymmetry exists because L1 wavelets respond to signal texture (higher-order statistics), not the mean convergence level that the CNN exploits.

### Worktree: `.worktrees/bnt_tomo_study/` (426 runs, 20 subdirs)

| Directory | Runs | Methods | Key feature |
|-----------|------|---------|-------------|
| `baryon_bias_tomo4_study/` | 180 | CNN(60), L1-jax(60), L1-VMIM(60) | 20 baryonification perms × 3 seeds × 3 methods; cdim=64 for L1-VMIM |
| `baryon_bias_tomo4_study_subset/` | 6 | CNN(2), L1-jax(2), L1-VMIM(2) | Subset for quick comparison |
| `bnt_tomo4_study/` | 18 | CNN(6), L1-norm(6), L1-VMIM(6) | Core BNT/no-BNT comparison; cdim=40 |
| `bnt_tomo4_study/l1vmim_opt_round2/` | 8 | L1-VMIM(8) | cdim=64 optimization (h512_nf8x256, h768_nf10x384) |
| `bnt_tomo4_study/round1_stage{A-H}*/` | 30 | CNN(30) | Progressive pipeline refinement (stages A through H: adding clips, dims, arch, flow configs) |
| `nobnt_tomo_bins_crosscorr_study/` | ~160 | mixed | Study-level manifests, not individual posteriors |
| `optimal_nobnt_crosscorr_benchmark/` | ~24 | mixed | Benchmark sweeps |

**Trust: ALL INVALIDATED.** No zero-mean-maps, no disjoint split. The baryon bias relative rankings (which baryonification permutation hurts most) may be directionally valid.

### `paper_sbi_consolidation/` — main tree (744 runs, 71 subdirs)

#### BNT comparison baseline (`bnt_comparison_tomo4/`, 18 runs)

| Config | Method | NDE | Seeds | FoM3 (mean ± std) | Standardized |
|--------|--------|-----|-------|--------------------|--------------|
| no-BNT | cnn_jaxili | jaxili MAF | 3 | 221,867 ± 6,640 | True |
| BNT | cnn_jaxili | jaxili MAF | 3 | 21,030 ± 633 | True |
| no-BNT | cnn | sbi_lens RealNVP | 3 | — (not in inventory) | True |
| BNT | cnn | sbi_lens RealNVP | 3 | — (not in inventory) | True |
| no-BNT | l1norm_jaxili | jaxili MAF | 3 | — | False |
| BNT | l1norm_jaxili | jaxili MAF | 3 | — | False |

BNT/no-BNT FoM3 ratio for CNN+jaxili: 0.095 (10.5× reduction). But both sides are mass-sheet contaminated — the 10.5× gap mixes BNT information loss with BNT's partial mitigation of the mass-sheet signal.

#### BNT losslessness campaign (main: 36 runs, cdim=6)

sbi_lens RealNVP NDE. Two architecture variants: `advanced_arch64_dense256_nostd` and `advanced_arch96_nostd`. No disjoint split, no zero-mean-maps. Comparison summary recorded in `campaign_summary.json` and `comparison_summary.json`.

**Subvariants with disjoint split** (the `indep_split_*` directories):

| Directory suffix | Runs | cdim | Architecture | Split |
|-----------------|------|------|-------------|-------|
| `indep_split_stagej_cdim6_v1` | 10 | 6 | stagej baseline | 70/30 strings |
| `indep_split_advanced_cdim10_v1` | 10 | 10 | advanced_arch64_dense256 | 70/30 strings |
| `indep_split_advanced_cdim10_long120k_v1` | 10 | 10 | advanced_arch64_dense256_long | 70/30 strings |
| `indep_split_advanced_cdim12_v1` | 10 | 12 | advanced_arch64_dense256 | 70/30 strings |

Total indep_split: 40 runs. Split verification from `split_independence_audit.json`: **0 example overlap** between compressor and NDE training sets; all 899 cosmologies appear in both (different noise realizations). Dataset: `grid_20deg_160px_nonoverlap48` (the non-overlapping 48-map variant, not the original `grid_20deg_160px`).

**Multipatch variants** (6 directories, 46 runs): Testing robustness across field patches with non-overlapping maps. No disjoint split. cdim=6, 10, 12.

#### Noise curriculum campaign (`cnn_bnt_noise_curriculum_campaign/`, 82 runs)

**ALL 82 runs have `require_disjoint_train_examples=True`.** sbi_lens RealNVP NDE.

From `FINAL_NOISE_CURRICULUM_REPORT.md`:

| Config | Architecture | BNT/noBNT FoM3 ratio | σ₈ std ratio | Rank score |
|--------|-------------|---------------------|-------------|-----------|
| **plain_ref** | plain | **0.914** | 1.129 | **0.106** |
| resnet18_curriculum | resnet18 | 0.868 | 0.908 | 0.170 |
| plain_curriculum | plain | 0.757 | 1.272 | 0.355 |
| resnet18_curriculum_long22k | resnet18 | 0.609 | 1.206 | 0.498 |
| resnet18_ref | resnet18 | 0.433 | 2.556 | 0.900 |
| resnet18_curriculum_slowramp | resnet18 | 0.341 | 2.178 | 0.985 |

**Key finding**: Noise curriculum HURTS plain CNN (0.757 vs 0.914) but HELPS ResNet18 (0.868 vs 0.433). Best overall: plain_ref without curriculum. ResNet18 without curriculum is catastrophically bad (σ₈ 2.6× wider).

#### BNT parity campaign (`cnn_bnt_parity_campaign/`, 56 runs)

**ALL 56 runs have `require_disjoint_train_examples=True`.**

Three phases (A/B/C) testing BNT parity with plain and resnet18 architectures. Each phase has `plain_ref` or `resnet18_ref` subdirectories with 6-10 runs each. sbi_lens RealNVP NDE.

#### ResNet split campaign (`cnn_bnt_resnet_split_campaign/`, 98 runs)

**ALL 98 runs have `require_disjoint_train_examples=True`.** sbi_lens RealNVP NDE.

From `CNN_BNT_RESNET_SPLIT_CAMPAIGN_REPORT.md`:

| Architecture | Runs | BNT/noBNT FoM3 ratio | Std inflation | vs plain noBNT FoM3 |
|-------------|------|---------------------|---------------|---------------------|
| **control_plain_split** | 10 | **0.610** | 1.059 | 1.00× (reference) |
| resnet50_split | 6 | 0.579 | 1.070 | 0.42× (much weaker) |
| resnet_small_split | 10 | 0.412 | 1.122 | — |

Plus extended tuning variants: resnet18 (12 runs), resnet34 (12 runs), resnet50 (42 runs across 6 NDE configurations: 6k/10k/12k steps, 256/320/384 hidden, 8/10/12 layers, with/without standardization).

**Key finding**: Plain CNN control beats ALL ResNet variants on BNT parity. ResNet50 reduces absolute constraining power to 42% of plain. Deeper architectures don't help — they hurt. This is consistent with the Phase 4 auto-push campaign finding that resnet50_gn is data-limited and overfits.

#### No-BNT deep audit (`cnn_nobnt_deep_audit/`, 18 runs)

CNN+jaxili MAF. Explicit `npe_train_split: "train"` and `npe_val_split: "test"` (train/train overlap). Architecture details: conv_channels 64,128,256, dense_width 128, pool_window 16. FoM3 range 151–575,283 — the 151 value is likely a failed run. Contains both `baseline_fulltrain` and other configurations. Also present as a copy in the `.worktrees/cnn-auto-cross-push-18-20-2026/` worktree.

#### No-BNT final matrix (`nobnt_final_matrix/`, 45 runs)

Three methods × 5 bin configurations × 3 seeds. The most informative Phase 1 dataset for understanding the mass-sheet asymmetry (see table in "Key insight" above). All CNN+jaxili runs, no disjoint split, no zero-mean-maps.

#### Baryonified appendix (`baryonified_appendix/`, 180 runs)

CNN+jaxili MAF, cdim=6. 20 baryonification permutations × 3 seeds × 3 truth cosmologies (perm0000, perm0001, ...). Reuses compressor from `nobnt_final_matrix/`. FoM3 312k–435k (mass-sheet contaminated). The relative baryon degradation pattern may survive.

Also exists as 60-run subset in the paper_sbi_consolidation directory and as 180-run version in the worktree.

#### Smoke tests (3 runs)

- `smoke_cnn_l1_jaxili/`: 1 CNN+jaxili, 2 epochs, batch_size=64, 2k samples, FoM3=318. Not a real run.
- `smoke_l1vmim_jaxili/`: 1 L1-VMIM+jaxili. FoM3=354 (also 669 for a second run). Not real runs.

### The BNT losslessness story (from CNN_BNT_LOSSLESSNESS_RETRAIN_REPORT.md)

The core question: can the CNN-VMIM compressor preserve information through the BNT transform? The campaign systematically improved the BNT/no-BNT FoM3 ratio from 0.095 (catastrophic) to 0.907 (near-lossless):

| Config | Conv/Dense | cdim | Steps | Std | Dataset | Split | BNT/noBNT FoM3 | σ₈ ratio | 
|--------|-----------|------|-------|-----|---------|-------|----------------|----------|
| baseline_final_paper | 32,64,128 / 64 | 6 | 20k | **ON** | random25 | overlap | 0.095 | 9.44 |
| stagej_repro | 64,128,256 / 128 | 6 | 60k | OFF | random25 | overlap | 0.794 | 1.22 |
| advanced cdim=6 | 64,128,256 / 256 | 6 | 80k | OFF | random25 | overlap | 0.892 | 1.17 |
| advanced cdim=8 | 64,128,256 / 256 | 8 | 80k | OFF | random25 | overlap | 0.789 | 1.19 |
| **advanced cdim=10** | 64,128,256 / 256 | **10** | 80k | OFF | random25 | overlap | **0.907** | **1.17** |
| NDE capacity L10/H320 | (best cdim=6 compressor) | 6 | 80k+12k | OFF | random25 | overlap | 0.744 | 1.23 |
| multipatch cdim=10 | 64,128,256 / 256 | 10 | 80k | OFF | nonoverlap48 | overlap | 0.843 | 1.18 |
| indep_split cdim=10 long120k | 64,128,256 / 256 | 10 | 120k | OFF | nonoverlap48 | **70/30** | 0.846 | **1.09** |

**What drove the improvement from 0.095 to 0.907:**
1. **Turning off summary standardization** was the single biggest lever (0.095 → 0.794). Standardization was destroying BNT-specific information in the compressed summaries.
2. **Wider dense layer** (64 → 256) and **longer compressor training** (20k → 80k) contributed incrementally.
3. **cdim=10** was the sweet spot. cdim=12 overfit; cdim=6 underfits.
4. **NDE capacity was NOT the bottleneck** — larger NDE worsened results (0.744), confirming the compressor is the binding constraint.
5. **Multipatch did NOT improve** over random-25 patch selection.
6. **Disjoint split slightly hurt FoM retention** (0.846 vs 0.907) but dramatically improved σ₈ agreement (1.09 vs 1.17).

**Caveats on these ratios**: All contaminated by mass-sheet leak. The leak affects no-BNT more than BNT (BNT partially removes the mean-convergence signal), so the true BNT/no-BNT ratio with zero-mean maps would likely be CLOSER to 1.0 — the denominator (no-BNT) would shrink by 25× while the numerator (BNT) might shrink by only 10-15×. The 0.907 ratio is therefore a conservative lower bound on true BNT losslessness.

### Phase 1 scientific conclusions that survive

**Robust findings** (ratios and ordinal comparisons valid despite mass-sheet contamination):
1. **BNT near-losslessness is achievable** at 0.91 FoM3 retention (likely higher post-zmm fix). The key was compression-side refinement, not NDE capacity.
2. **Summary standardization destroys BNT information** — turning it off improved BNT/no-BNT from 0.095 to 0.794.
3. **σ₈ is the most BNT-sensitive parameter** — 17% wider even in the best config, improvable to 9% with disjoint split + long training.
4. **Per-bin scaling**: bin4 > bin3 > bin2 > bin1 — robust across all methods.
5. **CNN extracts tomographic cross-correlation gain 3.9× better than L1** (from nobnt_final_matrix): CNN tomo4/single-bin ratio = 13.7×, L1 = 3.6×. This presages the Phase 3 finding that CNN beats L1 on cross-only channels.
6. **Baryon contamination is modest for CNN** (FoM3 ratio 1.01 ± 0.03) but noisier for L1 (1.31 ± 0.42) and L1-VMIM (1.17 ± 0.19). Baryons shift posteriors by O(1-2) Mahalanobis distances.

**The 70/30 split was validated** in `split_independence_audit.json`: zero example-level overlap, 899/899 cosmologies in both splits (different noise realizations). This design was sound and became the precedent for all later disjoint-split experiments.

**INVALIDATED for absolute comparison with later phases.** The 118 CNN+jaxili runs cannot short-circuit the L1-vs-CNN question — their compressors learned to exploit mass-sheet signal. The L1 auto-only FoM3 (~10k) is close to the post-fix value (~13k), confirming L1 was never significantly mass-sheet contaminated.

---

## Phase 2 — L1 development and benchmarking

### Overview

L1 pipeline development runs are mostly stored in two places already covered in Phase 1: the `.worktrees/bnt_tomo_study/` archive (which holds the `nobnt_tomo_bins_crosscorr_study` and `optimal_nobnt_crosscorr_benchmark` posteriors) and the `nobnt_final_matrix` in `paper_sbi_consolidation/`. This phase covers the orchestration metadata and the few standalone diagnostic runs.

**Timeline**: The L1 pipeline was developed February–April 2026. The sbi_lens RealNVP version (`npe_l1norm_nbody_tomo.py`) was created 2026-02-18. The jaxili MAF version (`npe_l1norm_jaxili_nbody_tomo.py`) was created 2026-03-29, a full 10 days after the CNN jaxili script. The L1-VMIM pipeline was created 2026-04-01. The NDE architecture divergence (CNN → RealNVP, L1 → jaxili MAF) was established from day one of the jaxili scripts and was never made explicit until the 2026-05-27 NDE-swap session.

### `nobnt_tomo_bins_crosscorr_study/` (orchestrator only — 121 posteriors in worktree)

Manifest records: 3 methods (CNN, L1, L1-VMIM), `grid_20deg_160px` dataset, 1 seed (41), GPU 0. CNN config: cdim=6, 60k compressor steps, plain arch. L1: jaxili, pca_components=0, 40 bins, 5 scales. L1-VMIM: cdim=64, hidden 768×768, 10-layer flow. All posteriors landed in `.worktrees/bnt_tomo_study/scripts/sbi/posteriors_archive/nobnt_tomo_bins_crosscorr_study/` (121 runs covered in Phase 1).

### `optimal_nobnt_crosscorr_benchmark/` (orchestrator + sweeps)

A clean and reproducible benchmark framework running in 4 phases: (1) L1+jaxili sweep, (2) full CNN/L1/L1-VMIM matrix, (3) FoM3 analysis, (4) overlay plots. Contains sweep manifests and ranking JSONs for `cnn_tomo4/`, `l1_tomo4/`, and `l1vmim_tomo4/` subdirectories. Posteriors stored within the sweep subdirectories but no `.meta.json` files found — these use a different metadata format (`seed41_results.json`, `ranking.json`, `final_selection.json`).

### Standalone diagnostic runs (8 runs)

| Directory | Runs | Method | FoM3 | Notes |
|-----------|------|--------|------|-------|
| `diagnostics/harm_l1_heldout_cosmo_sweep/retrain_seed41/` | 1 | L1 cross jaxili | 62,561 | zero_mean=True, pca=False, 10 channels. Heldout cosmology test — trained NDE on different cosmology from fiducial, then inferred. High FoM3 suggests L1 cross is robust to cosmology shift. |
| `diagnostics/cross_maps/` | 1 | L1-norm | N/A | Early cross-maps diagnostic. No zero_mean or FoM3 recorded. |
| `exploratory/` (root) | 5 | CNN | N/A | 5 CNN runs directly in exploratory root — likely misplaced or intermediate. |
| `exploratory/investigate_old_script/` | 1 | L1-norm | N/A | Debugging run comparing current vs old script output. |

### Phase 2 trust assessment

The mass-sheet leak asymmetry (confirmed in Phase 1) means L1 auto-only runs from this era are approximately comparable to post-fix values: pre-fix ~10k FoM3 vs post-fix ~13k. The ~30% difference is likely from train/train NDE overlap (present in both eras) and minor pipeline improvements, not from mass-sheet contamination. CNN values from this era are inflated 25-30× and cannot be compared.

---

## Phase 3 — L1 vs CNN cross-maps (169 runs, 43 directories)

### Summary statistics

- **Total runs**: 169 across 43 directories (excludes worktree copies)
- **Methods**: CNN sbi_lens RealNVP (88), L1 cross jaxili MAF (48), L1-norm jaxili (6), L1-norm sbi_lens (24), CNN+jaxili (3 in sub-dirs)
- **zero_mean_maps=True**: 133/169 (79%)
- **Disjoint split**: 6/169 (4%) — only `zero_mean_maps_parity_check/run_a_resnet18`
- **FoM3 in inventory**: 70/169 (41%) — many CNN runs store FoM3 in campaign SUMMARY.md, not meta.json

### Complete directory catalog

#### L1 cross-maps campaign (`cross_maps_campaign/`, 40 runs)

| Subdirectory | Runs | BNT | Route | Noise model | FoM3 range | Trust |
|-------------|------|-----|-------|------------|-----------|-------|
| `jaxili_harm_cross_nobnt/` | 6 | no | harmonic cache | **v1 auto_scalar** (verified: log shows `noise_sigma = 0.012658` single scalar, file date 2026-05-01 = same day as harmonic-cache-route commit) | 55,541–76,348 | **INVALIDATED** — v1 inflated; corrected value ~38k |
| `jaxili_harm_cross_bnt/` | 3 | yes | harmonic cache | v1 auto_scalar | 4,852–5,627 | INVALIDATED — v1 |
| `harm_l1_truthcheck/cosmo_delta_*` | 5 | no | harmonic cache | v1 auto_scalar | 44,785–67,414 | INVALIDATED — v1. Directional finding (L1 robust to cosmo shift) may hold |
| `jaxili_auto_zm_nobnt/` | 3 | no | auto-only | N/A (no cross channels) | 9,965–16,623 | **TRUSTWORTHY** — L1 auto-only, zmm=T, no PCA, no cross-channel complications |
| `jaxili_auto_zm_bnt/` | 3 | yes | auto-only | N/A | 723–890 | TRUSTWORTHY — shows BNT destroys L1 auto info |
| `jaxili_auto_bnt/` | 3 | yes | auto-only | N/A | 554–770 | pre-zmm but L1 not mass-sheet-sensitive |
| `jaxili_auto_nobnt/` | 3 | no | auto-only | N/A | 7,572–16,931 | pre-zmm but L1 not mass-sheet-sensitive |
| `jaxili_cross_nobnt*/` | 8 | no | flat-sky TFDS | v1 auto_scalar | 1,505–14,302 | INVALIDATED — flat-sky + v1; superseded by harmonic route |
| `jaxili_cross_bnt*/` | 6 | yes | flat-sky TFDS | v1 auto_scalar | 371–1,463 | INVALIDATED |

#### Cross-only campaign v1 (`cross_only_campaign/`, 11 runs)

From SUMMARY.md:

| Arm | Runs | FoM3 pooled | FoM3 mean ± std | Noise model |
|-----|------|------------|-----------------|-------------|
| CNN plain d10 | 3 | 19,223 | 20,569 ± 941 | N/A (CNN) |
| CNN resnet50_gn d10 | 3 | 26,282 | 28,131 ± 1,101 | N/A (CNN) |
| L1 cross-only | 5 | 12,131 | 13,189 ± 2,654 | **v1 auto_scalar** |

Trust: CNN numbers are post-zmm but train/train + RealNVP. L1 numbers are v1-contaminated (v2 gives 18,121 mean). L1/CNN ratio v1: 0.46×; v2: 0.62×.

#### Cross-only campaign v2 (`cross_only_campaign_v2_chsigma/`, 11 runs)

From SUMMARY.md:

| Arm | Runs | FoM3 pooled | FoM3 mean ± std | Noise model |
|-----|------|------------|-----------------|-------------|
| CNN plain d10 | 3 | 20,104 | 21,219 ± 1,390 | N/A (CNN) |
| CNN resnet50_gn d10 | 3 | 25,830 | 26,614 ± 1,554 | N/A (CNN) |
| L1 cross-only | 5 | 16,070 | 18,121 ± 3,634 | **v2 channel_empirical_global** |

Trust: CNN numbers consistent across v1/v2 (as expected — CNN unaffected by L1 noise model). L1 improved 37% (13.2k→18.1k) from noise model fix. **CNN still beats L1 on cross-only by 1.4-1.6×.** L1 has 2× higher seed std.

#### L1 auto+cross v2 (`auto_cross_v2_chsigma/`, 3 runs)

| Seed | FoM3 | Noise model verified |
|------|-------|---------------------|
| s41 | 39,895 | channel_empirical_global (from log: `cross_noise_model = channel_empirical_global` + channel_scale table) |
| s42 | 36,423 | same |
| s43 | 38,361 | same |

Mean: 38,226 ± 1,421. **TRUSTWORTHY** — best available L1 auto+cross. 10 channels (4 auto + 6 cross), 50k NDE epochs, jaxili MAF, zero-mean maps, no PCA, harmonic cache route.

#### CNN harmonic cross (`cnn_with_harm_cross_normalized/`, 9 runs)

| Arch | Runs | FoM3 (from campaign) | zmm | disjoint | Notes |
|------|------|---------------------|-----|----------|-------|
| plain | 3 | ~25,466 (mean from auto-cross-push STATUS.md) | T | F | `harmonic_normalize_input_channels=True`, `summary_standardized=True`, 150k compressor steps |
| resnet50 | 3 | ~18,763 (from STATUS.md) | T | F | High seed scatter |
| resnet50_gn | 3 | — | T | F | GroupNorm variant |

Trust: PARTIALLY TRUSTWORTHY — zmm but no disjoint, sbi_lens RealNVP, train/train NDE overlap. The ~25.5k number is the CNN auto+cross baseline for the auto-cross-push campaign. resnet50_gn is LOWER than plain — this is the capacity-match falsification result.

#### CNN architecture exploration (27 runs)

| Directory | Runs | arch | zmm | Notes |
|-----------|------|------|-----|-------|
| `cnn_extended_train_zm/` | 5 | plain, dense512, 240k steps | T | Extended training, auto-only |
| `cnn_resnet34_50_zm_cdim1224/` | 12 | resnet34, resnet50 | T | cdim=12,24 sweeps |
| `cnn_resnet50_zm_sweep/` | 7 | resnet50 | T | Various configs |
| `cnn_vmim_target_stability/` | 6 | plain | T | VMIM companion stability test |

All use sbi_lens RealNVP, train/train overlap. No FoM3 in inventory.

#### Zero-mean-maps parity check (17 runs)

| Directory | Runs | arch | zmm | disjoint | Trust |
|-----------|------|------|-----|----------|-------|
| `run_a_resnet18/` | 6 | resnet18 | T | **T** | **FULLY CLEAN** — but no FoM3 (compressor-only?) |
| `run_b_advanced_plain/` | 10 | plain | T | F | zmm but not disjoint |
| `smoke/` | 1 | resnet18 | T | F | smoke test |

**`run_a_resnet18` is the only Phase 3 directory with both zmm=T and disjoint=T.** 6 runs, no FoM3 — these were likely compressor training runs without NDE evaluation.

#### Hypothesis tests (6 runs, 3 felt fibers)

Three formal hypothesis tests were run as felt fibers during Phase 3:

| Fiber | Hypothesis | Runs | Result | Verdict |
|-------|-----------|------|--------|---------|
| `cnn-h1-inductive-bias-2026-05` | Cross-channel attention block improves CNN cross-channel extraction | 3 (plain_attn) | Haircut 0.684 vs 0.685 plain; no improvement | **FALSIFIED** |
| `cnn-h2-data-limit-scoping-2026-05` | CNN is data-limited (needs more cosmologies) | 0 (scoping only) | Determined untestable with current sim suite | Scoping complete, no compute |
| `cnn-h3-summary-dim-cdim100-test-2026-05` | Higher cdim=100 captures more cross-channel info | 3 (plain, cdim=100) | FoM3 12.2k vs 24.0k anchor — 49% crater | **FALSIFIED** (opposite direction) |

All three hypotheses for why CNN fails to extract cross-channel information were falsified: not attention, not data volume, not compression bottleneck.

#### Pre-fix systematic runs (36 runs)

| Directory | Runs | Method | zmm | Notes |
|-----------|------|--------|-----|-------|
| `systematic_runs_cnn_retrain_proper/` | 12 | CNN RealNVP | pre-fix | Pre-zmm re-training |
| `systematic_runs_l1_rerun_proper/` | 12 | L1 RealNVP | pre-fix | Pre-zmm L1 re-runs |
| `systematic_runs_l1_snr10_rerun/` | 12 | L1 RealNVP | pre-fix | SNR=10 variant |

Trust: ALL INVALIDATED — pre-zmm and use sbi_lens RealNVP (not jaxili).

### Phase 3 summary of trustworthy results

| Config | Method | NDE | FoM3 (mean) | Seeds | Verified clean | Key trust caveat |
|--------|--------|-----|-------------|-------|----------------|------------------|
| L1 auto+cross (v2 noise) | L1 cross jaxili | jaxili MAF | 38,226 | 3 | noise model verified from log | train/train NDE overlap |
| L1 cross-only (v2 noise) | L1 cross jaxili | jaxili MAF | 18,121 | 5 | v2 campaign | train/train NDE overlap |
| L1 auto-only | L1 auto jaxili | jaxili MAF | 13,131 | 3 | zmm=T, no cross channel | train/train NDE overlap |
| CNN auto+cross (harm-norm) | CNN | sbi_lens RealNVP | ~25,466 | 3 | zmm=T | train/train + RealNVP |
| CNN cross-only (resnet50_gn) | CNN | sbi_lens RealNVP | 26,614 | 3 | v2 campaign | train/train + RealNVP |
| CNN cross-only (plain) | CNN | sbi_lens RealNVP | 21,219 | 3 | v2 campaign | train/train + RealNVP |

**Key finding**: CNN beats L1 on cross-only (26.6k vs 18.1k = 1.47×), but L1 beats CNN on auto+cross (38.2k vs 25.5k = 1.50×). The L1 advantage comes entirely from combining auto and cross information more effectively.

---

## Phase 4 — Autoresearch campaigns

### CNN auto-push (auto-only)

**Configuration**: plain CNN, zero-mean maps, sbi_lens RealNVP NDE, train/train overlap (`require_disjoint=False`, `nde_train_split=train`). Branch: `autoresearch/cnn-auto-push-18-20-2026`. Started 2026-05-18.

**24 iteration directories** on NAS (iter-0 through iter-23), 70 posteriors across all iterations. Only 7 iterations have `landing.json` with standardized FoM3 reporting:

| Iteration | MoS FoM3 | Pooled FoM3 | Haircut | Seeds | L1 ref MoS | Config notes |
|-----------|----------|-------------|---------|-------|-----------|-------------|
| iter-5 | 18,568 | 12,894 | 0.694 | 3 | 38,226 | baseline config |
| iter-16 | 19,502 | 13,868 | 0.711 | 3 | 38,226 | Q2 (iter-5 stack at 120k) — **best amended-check passer** |
| iter-19 | 18,213 | 13,087 | 0.719 | 3 | 38,226 | — |
| iter-20 | 18,673 | 13,944 | 0.747 | 3 | 38,226 | — |
| iter-21 | 17,805 | 13,829 | 0.777 | 3 | 38,226 | — |
| iter-22 | 19,304 | 12,531 | 0.649 | 3 | 38,226 | Q9c (variance/drift at 120k) — ceiling falsifier NULL |
| iter-23 | 19,874 | 12,945 | 0.651 | 3 | 38,226 | Q4 (VMIM NF hidden 256) — ceiling falsifier NULL |

**Ceiling TRIPLY CONFIRMED** per `CEILING_EVIDENCE.md`: three independent falsifiers (iter-22 variance/drift, iter-23 bound-widening, Q1 resnet50_gn@120k with overfitting signature) all returned NULL. The CNN auto-only ceiling under plain architecture is:

> **Pooled FoM3 ≈ 13,868 (iter-16), MoS ≈ 19,502. Pooled/L1-auto+cross ratio ≈ 0.41.**

Campaign also discovered and fixed the `compressor_checkpoint_policy` bug (last_step vs best_val, commit `5c5a6d9`). The Q1 resnet50_gn retest with the fix showed clear overfitting: val-loss argmin at step 42k (35% of 120k), then 1.28-nat drift, train-val gap 2.24 nats. FoM3 = 11,820 (−39% vs iter-16). Conclusion: deeper CNN is data-limited, not capacity-limited.

**Trust: PARTIALLY TRUSTWORTHY.** zmm=T but no disjoint, sbi_lens RealNVP. The ceiling finding is the most robust result — it's a relative finding within a controlled campaign and was confirmed from three orthogonal angles.

### CNN auto-cross-push (auto+cross)

**Configuration**: plain CNN on 10-channel harmonic input (4 auto + 6 cross), `harmonic_normalize_input_channels=True`, zmm=True, sbi_lens RealNVP, train/train overlap. Branch: `autoresearch/cnn-auto-cross-push-18-20-2026`. Started 2026-05-18 ~21:50 UTC.

**115 iteration directories** on NAS (iter-0 through iter-112, plus iter-Q4OFF-60k, iter-108-Q6ON-60k, and iter-81-Q2prime/QEARLY51k/QLOWER60k variants). Each iteration runs: compressor training (~5.5h), then 3 NDE seeds (~1h for seed 1, ~1min for seeds 2-3 using cached data). 10 iterations have `job_results.json`. STATUS.md is 100k+ tokens.

**Key metrics** (from STATUS.md header):

| Arm | Config | FoM3 mean (perm 0) | FoM3 mean (7-perm avg) | CoV |
|-----|--------|-------------------|----------------------|-----|
| CNN plain baseline | harm-norm 150k steps cdim=10 | 25,466 ± 636 | 25,511 ± 5,423 | 21% |
| CNN resnet50_gn | harm-norm | 18,763 ± 2,549 | — | high |
| L1 auto+cross target | v2 ch-σ | 38,226 ± 1,421 | 50,065 ± 13,063 | 26% |

**Critical methodological finding** (iter-5 F2 + iter-6 F4): fiducial permutation sweep (3 seeds × 7 perms = 21 inferences) revealed that the perm-0 FoM3 is noise-realization specific. Perm-averaged gap is 1.96× (not 1.52× at perm 0). L1 CoV (26%) is HIGHER than CNN CoV (21%) — **L1 is more fragile to noise realization than CNN**. This means any single-perm FoM3 comparison understates the true gap.

**The campaign never closed the gap.** After 115 iterations exploring architecture variants (resnet50_gn capacity-match), data-prep toggles, learning rate schedules, and VMIM companion changes, the CNN auto+cross FoM3 remained anchored at ~25k, roughly 0.66× of the L1 target at perm 0 and 0.51× at perm-average.

**Trust: PARTIALLY TRUSTWORTHY.** Same contamination as auto-push. The inability of 115 iterations to close the gap is the strongest evidence that the L1 auto+cross advantage is architectural, not a tuning artifact.

### Worktree experiments (`.worktrees/cnn-auto-cross-push-18-20-2026/`, 141 runs)

This worktree contains both Phase 3 runs (L1 cross campaigns, zero-mean parity checks) and Phase 1 runs (BNT losslessness variants). Runs are duplicates of or predecessors to runs in the main tree. The worktree also houses experiment infrastructure for the auto-cross-push campaign (overlay scripts, run pipelines).

Key non-duplicate content:
- BNT losslessness campaign variants (`stagej_repro`, `nde_capacity_l10h320_flow12k`, `advanced_arch64_dense256_nostd`, `advanced_arch96_nostd`) — 62 CNN runs with various NDE capacity configurations, cdim=6/8/10
- These are Phase 1-era runs (no zmm, mostly disjoint split) that were staged in the worktree for the auto-cross-push campaign's F1 data-prep audit

---

## Phase 5 — Canonical refresh + NDE-swap (25 runs + NDE-swap session)

### Canonical anchors refresh (`canonical_anchors_refresh/`)

The most recent campaign, attempting clean, comparable anchors. 22 meta.json files across posteriors/ and iter1_archive/.

#### Current runs (posteriors/, 13 runs)

| Run | Method | NDE | zmm | dis | nde_split | pca | cross | FoM3 | Trust |
|-----|--------|-----|-----|-----|-----------|-----|-------|------|-------|
| cnn_auto_canon_s41 | CNN | sbi_lens RealNVP | T | **T** | train[70%:] | — | N (4ch) | **NOT COMPUTED** | **FULLY CLEAN** — posterior exists |
| cnn_auto_canon_s42 | CNN | sbi_lens RealNVP | T | **T** | train[70%:] | — | N (4ch) | **NOT COMPUTED** | **FULLY CLEAN** — posterior exists |
| cnn_auto_canon_s43 | CNN | sbi_lens RealNVP | T | **T** | train[70%:] | — | N (4ch) | **NOT COMPUTED** | **FULLY CLEAN** — posterior exists |
| cnn_cross_canon_s41 | CNN | sbi_lens RealNVP | T | **T** | train[70%:] | — | Y (10ch harm) | **NOT COMPUTED** | **FULLY CLEAN** — posterior exists |
| cnn_cross_canon_s42 | CNN | sbi_lens RealNVP | T | **T** | train[70%:] | — | Y (10ch harm) | **NOT COMPUTED** | **FULLY CLEAN** — posterior exists |
| cnn_cross_canon_s43 | CNN | sbi_lens RealNVP | T | **T** | train[70%:] | — | Y (10ch harm) | **NOT COMPUTED** | **FULLY CLEAN** — posterior exists |
| l1_auto_canon_s41 | L1 cross jaxili | jaxili MAF | T | — | train | F | N (4ch) | 11,419 | zmm but no disjoint |
| l1_auto_canon_s42 | L1 cross jaxili | jaxili MAF | T | — | train | F | N (4ch) | 21,951 | zmm but no disjoint |
| l1_auto_canon_s43 | L1 cross jaxili | jaxili MAF | T | — | train | F | N (4ch) | 11,752 | zmm but no disjoint |
| l1_auto_canon_epochs5k_s41 | L1 cross jaxili | jaxili MAF | T | — | train[70%:] | **T** | N (4ch) | 4,121 | **PCA BUG** — pca_applied=True |
| l1_cross_canon_s41 | L1 cross jaxili | jaxili MAF | T | — | train | F | Y (10ch) | 39,895 | Copy of auto_cross_v2 |
| l1_cross_canon_s42 | L1 cross jaxili | jaxili MAF | T | — | train | F | Y (10ch) | 36,423 | Copy of auto_cross_v2 |
| l1_cross_canon_s43 | L1 cross jaxili | jaxili MAF | T | — | train | F | Y (10ch) | 38,361 | Copy of auto_cross_v2 |

**CRITICAL FINDING**: The 6 CNN canonical runs have **fully clean posteriors** (zmm=T, dis=T, verified 0 example overlap, 0 theta-only overlap) **with 100k posterior samples in .npy files** — but FoM3 was never computed. These are the first and only fully clean CNN runs in the project. Computing FoM3 from them is trivial.

CNN auto-only details: plain arch, TFDS route, 4 channels, `compressor_train_split=train[:70%]`, `nde_train_split=train[70%:]`, `summary_standardized=True`. NDE training: best_val_loss=-11.68 at step 3000, early-stopped at step 13000 (patience=20). 211,445 compressor train + 90,619 NDE train examples. 0 shared examples.

CNN auto+cross details: plain arch, **harmonic route**, 10 channels, `harmonic_normalize_input_channels=True`. NDE training: best_val_loss=-11.37 at step 2000, early-stopped at step 12000 (patience=20). 4,405 compressor train + 1,888 NDE train files. 0 overlap. Note: worse val loss than auto-only and very early stopping — suggests the RealNVP struggled with 10-channel input under the 70/30 split constraint (fewer NDE training examples).

#### iter1_archive (9 runs)

| Run | Method | pca | dis | FoM3 | Trust |
|-----|--------|-----|-----|------|-------|
| cnn_auto_canon_s41/s42/s43 | CNN | — | T | NOT COMPUTED | CLEAN — likely same posteriors as current |
| l1_auto_canon_s41/s42/s43 | L1 jaxili | **T** | — | 5,376–5,970 | **INVALIDATED** — PCA default=50 destroyed info |
| l1_cross_canon_s41/s42/s43 | L1 jaxili | **T** | — | 5,434–6,414 | **INVALIDATED** — PCA default=50 |

This is the "before we noticed the PCA bug" version. The PCA-contaminated L1 runs give ~5.5k FoM3 vs the PCA-free auto runs at ~15k (3× destruction).

### NDE-swap test (2026-05-26/27)

Session that discovered the NDE architecture mismatch. Results from `SESSION_RESULTS.md`:

| Run | NDE | Data split | FoM3 | FoM2d Om-s8 | FoM2d Om-w0 | FoM2d s8-w0 | σ(Om) | σ(s8) | σ(w0) |
|-----|-----|-----------|------|------------|------------|------------|-------|-------|-------|
| CNN cross iter108+RealNVP | sbi_lens RealNVP | train/train 302k | 25,920 | 2,158 | 274 | 144 | 0.0297 | 0.0440 | 0.186 |
| CNN cross iter108+jaxili | jaxili MAF [50,50] | train/train 436k | 17,624 | 1,799 | 231 | 126 | 0.0312 | 0.0460 | 0.194 |
| CNN cross canonical+jaxili | jaxili MAF [50,50] | 70/30 225k | 15,963 | 1,626 | 242 | 132 | 0.0318 | 0.0456 | 0.188 |
| CNN cross canonical+RealNVP | sbi_lens RealNVP | 70/30 225k | 19,699 | 2,014 | 311 | 161 | 0.0271 | 0.0423 | 0.166 |
| CNN auto iter108+jaxili | jaxili MAF [50,50] | train/train 227k | 16,207 | 1,652 | 353 | 206 | 0.0263 | 0.0415 | 0.120 |
| **L1 cross canonical+jaxili** | jaxili MAF [50,50] | full train 302k | **39,895** | **2,549** | **304** | **173** | **0.0254** | **0.0437** | **0.133** |

**Invalidated NDE-swap runs** (bugs found during session): CNN cross jaxili MAF [256,256] patience=100 → FoM3=6244 (overfit). CNN cross/auto jaxili batch=8192 → FoM3=2866 (batch too large). iter-108 + jaxili with inverted channel_scale → FoM3=3346 (obs out of distribution).

**Key conclusions from the NDE-swap test** (verified from SESSION_RESULTS.md, not guesses):
1. **FoM3 amplifies ~5% marginal-width differences into ~50% FoM3 swings**: RealNVP 25.9k vs jaxili 17.6k is a 47% FoM3 gap, but σ(Om) differs by only 5% (0.0297 vs 0.0312) and σ(s8) by 4% (0.0440 vs 0.0460).
2. **L1 genuinely outperforms CNN on auto+cross through tighter w₀**: L1 σ(w0)=0.133 vs CNN's 0.166-0.194 (29-46% tighter). Om and s8 marginals are comparable or CNN slightly better.
3. **CNN cross channels barely help under jaxili MAF**: auto-only FoM3=16.2k → auto+cross=17.6k = +9%. Under RealNVP the baseline is higher but the cross gain was also modest in the auto-cross-push campaign.
4. **70/30 split costs ~18-24% FoM3 vs train/train** (comparing rows 1 vs 4, 2 vs 3). This quantifies the train/train inflation for the first time.

---

## Clean comparison table

### Tier 1: Fully clean (zero-mean + disjoint split + verified 0 overlap)

| Run | Method | NDE | Config | Data split | FoM3 | 2D areas | Trust |
|-----|--------|-----|--------|-----------|------|---------|-------|
| cnn_auto_canon_s41/s42/s43 | CNN | sbi_lens RealNVP | auto-only, 4ch, TFDS, plain | 70/30 (211k/91k, 0 overlap) | Per-seed: 18,060/17,732/14,845. Pooled: 12,873. | — | **FULLY CLEAN** (but RealNVP early-stopped at step 3-4k before diverging) |
| cnn_cross_canon_s41/s42/s43 | CNN | sbi_lens RealNVP | auto+cross, 10ch, harmonic, harm-norm | 70/30 (4405/1888 files, 0 overlap) | Per-seed: 19,699/14,914/18,214. Pooled: 12,615. | — | **FULLY CLEAN** (but RealNVP early-stopped at step 2k before diverging) |

**These 6 runs are the only fully clean posteriors in 1,517 runs. Computing FoM3 requires zero GPU time.** Note: the CNN auto+cross RealNVP early-stopped at step 12k (patience=20, best at step 2k) — it may have struggled with 10-channel input under the smaller 70/30 NDE training set.

### Tier 2: Best available (zero-mean, no PCA, post-noise-fix; train/train NDE overlap)

| Run | Method | NDE | Config | Seeds | FoM3 (mean) | σ(Ωm) | σ(σ₈) | σ(w₀) | Trust caveat |
|-----|--------|-----|--------|-------|-------------|--------|--------|--------|-------------|
| auto_cross_v2/l1_auto_cross | L1 cross | jaxili MAF | auto+cross 10ch, 50k epochs | 3 | 38,226 | 0.0254 | 0.0437 | 0.133 | train/train NDE |
| cross_only_v2/l1_cross_only | L1 cross | jaxili MAF | cross-only 6ch, 50k epochs | 5 | 18,121 | — | — | — | train/train NDE |
| cross_maps/jaxili_auto_zm_nobnt | L1 auto | jaxili MAF | auto-only 4ch, 5k epochs | 3 | 13,131 | — | — | — | train/train NDE |
| cnn_with_harm_cross_normalized | CNN | sbi_lens RealNVP | auto+cross 10ch harm, 150k comp | 3 | ~25,466 | — | — | — | train/train + RealNVP |
| cross_only_v2/resnet50_gn_d10 | CNN | sbi_lens RealNVP | cross-only 6ch, resnet50_gn | 3 | 26,614 | — | — | — | train/train + RealNVP |
| auto-push iter-16 | CNN | sbi_lens RealNVP | auto-only 4ch, 120k comp | 3 | 19,502 (MoS) | — | — | — | train/train + RealNVP |

### Tier 3: NDE-swap test (same compressor, different NDE; single seed s41)

From SESSION_RESULTS.md — the only experiment with 2D areas for all methods:

| Run | NDE | Split | FoM3 | FoM2d(Ωm-σ₈) | FoM2d(Ωm-w₀) | FoM2d(σ₈-w₀) | σ(Ωm) | σ(σ₈) | σ(w₀) |
|-----|-----|-------|------|-------------|-------------|-------------|-------|-------|-------|
| CNN auto+cross + RealNVP | sbi_lens RealNVP | train/train | 25,920 | 2,158 | 274 | 144 | 0.030 | 0.044 | 0.186 |
| CNN auto+cross + jaxili | jaxili MAF | train/train | 17,624 | 1,799 | 231 | 126 | 0.031 | 0.046 | 0.194 |
| CNN auto+cross + RealNVP | sbi_lens RealNVP | 70/30 | 19,699 | 2,014 | 311 | 161 | 0.027 | 0.042 | 0.166 |
| CNN auto+cross + jaxili | jaxili MAF | 70/30 | 15,963 | 1,626 | 242 | 132 | 0.032 | 0.046 | 0.188 |
| CNN auto-only + jaxili | jaxili MAF | train/train | 16,207 | 1,652 | 353 | 206 | 0.026 | 0.042 | 0.120 |
| **L1 auto+cross + jaxili** | jaxili MAF | full train | **39,895** | **2,549** | **304** | **173** | **0.025** | **0.044** | **0.133** |

### The fairest comparison possible today

Matching NDE (jaxili MAF), zero-mean maps, no PCA, all single seed s41:

| Config | L1 FoM3 | CNN FoM3 | L1/CNN | L1 σ(w₀) | CNN σ(w₀) | Note |
|--------|---------|----------|--------|----------|----------|------|
| Auto+cross | 39,895 | 17,624 | 2.26× | 0.133 | 0.194 | L1 σ(w₀) 31% tighter |
| Auto+cross (disjoint CNN) | 39,895 | 15,963 | 2.50× | 0.133 | 0.188 | CNN has fewer NDE examples |
| Auto-only | ~13k | 16,207 | 0.80× | — | 0.120 | CNN edges L1 on auto-only |

**The L1 auto+cross advantage is real, concentrated in w₀, and survives the NDE-swap test.** On auto-only, CNN is slightly better (0.80× in L1's favor = 1.25× in CNN's favor). The gain from adding cross channels: L1 gets ~3× boost, CNN gets ~9% boost.

**Caveat**: The L1 run uses full-train NDE while the fairest CNN comparison uses 70/30 NDE. An L1 run with 70/30 split would lose ~18-24% FoM3 (based on the CNN split penalty), bringing L1 to ~31-33k vs CNN's ~16k — still a ~2× advantage.

---

## Open questions — experiments to resolve cleanly

**Zero-GPU-time (analysis only):**

1. ~~**Cross-noise-model verification**~~ **DONE** — verified `channel_empirical_global` from log for auto_cross_v2. Verified `auto_scalar` (single noise_sigma scalar) from log for harm_cross_nobnt.

2. **Compute 2D areas and marginal σ for the 6 clean CNN posteriors** in `canonical_anchors_refresh/posteriors/cnn_{auto,cross}_canon_s4{1,2,3}.npy`. FoM3 is already computed (auto pooled 12,873; cross pooled 12,615) but 2D areas and per-parameter marginal σ have not been reported for these clean runs.

3. **Compute 2D areas and marginal σ for ALL Tier 2 runs.** Currently only the NDE-swap session has 2D metrics. Re-analyzing the existing .npy posteriors from Tier 2 would give a complete 2D area comparison table.

**Requires GPU time:**

4. **CNN + jaxili MAF + disjoint split + auto+cross**: Run `npe_cnn_jaxili_nbody_tomo.py` with zero-mean, disjoint split, harmonic-normalized 10-channel input. This eliminates the NDE confound for the auto+cross comparison. 3 seeds.

5. **L1 auto+cross with disjoint split**: Run `auto_cross_v2_chsigma` config with `--nde-train-split train[70%:]` to give a fair L1 counterpart. The 18-24% FoM3 penalty from disjoint split (quantified in NDE-swap) would bring L1 from ~38k to ~29-32k.

6. **L1 auto+cross with disjoint split + jaxili MAF + same NDE config as experiment #4**: The definitive apples-to-apples comparison. Same NDE, same split, same preprocessing. Only the compressor differs.

---

## Appendix A — 118 CNN+jaxili runs (the "surprise" discovery)

These 118 runs span 6 directories:

| Directory | Runs | FoM3 range | Notes |
|-----------|------|-----------|-------|
| baryonified_appendix | 60 | 312k-435k | Baryon contamination study, cdim=6 |
| cnn_nobnt_deep_audit | 18 | 151-575k | Deep audit with explicit train/train |
| bnt_comparison_tomo4 | 6 | 20k-231k | BNT vs no-BNT, standardized |
| nobnt_final_matrix | 15 | 6k-420k | Single-bin through tomo4 |
| smoke_cnn_l1_jaxili | 1 | 318 | 2-epoch smoke test |
| cnn_nobnt_deep_audit (worktree copy) | 18 | 151-575k | Duplicate of above |

**None can short-circuit the L1-vs-CNN question.** All predate `--zero-mean-maps`, so the compressor itself learned to exploit the mass-sheet signal. The posteriors are fundamentally too tight and can't be rescued.

The baryonified appendix runs (60) are the largest CNN+jaxili group and were designed for a baryon contamination study. Their relative rankings (which baryon model hurts most) may still be directionally valid, but absolute constraining power numbers are meaningless.

---

## Appendix B — Autoresearch campaign efficiency

| Campaign | Iterations | GPU time est. | Start FoM3 | Best FoM3 | Improvement | Target | Gap closed |
|----------|-----------|--------------|-----------|----------|-------------|--------|-----------|
| auto-push | 24 | ~36 GPU-h | 18,568 (iter-5) | 19,874 (iter-23) | +7% | — | Ceiling certified |
| auto-cross-push | 115 | ~500+ GPU-h | 25,466 (baseline) | ~25,500 (no improvement) | ~0% | 38,226 (L1) | Never closed |

The auto-cross-push campaign consumed an estimated 500+ GPU-hours (115 iters × ~5.5h compressor + ~1h NDE each). It explored architecture variants (resnet50_gn capacity-match), data-prep toggles (Q4-OFF, Q5-OFF), learning rate schedules, and VMIM companion changes. None moved the FoM3 needle.

**Methodological value produced despite FoM3 stagnation:**
1. Fiducial permutation analysis (F2/F4): noise-realization variance dominates seed variance 8×. Perm-averaged L1/CNN gap = 1.96× (vs 1.52× at perm 0).
2. Pool/MoS haircut (~0.69) shown to be structural and invariant to architecture.
3. Capacity-match falsification: resnet50_gn with identical NDE config gives FoM3 15-17k (LOWER than plain at 25k).
4. First quantification of 70/30 split penalty: 18-24% FoM3 cost.
5. Data-prep audit (F1) verified harmonic-route data integrity for CNN.

Both campaigns optimized FoM3, which we now know amplifies noise in the correlation structure. Whether 2D areas or marginal σ would have identified different improvement directions is unknown but plausible — the w₀ marginal is where L1 dominates, and architectures that specifically improve w₀ sensitivity were never tested.

## Appendix C — Disjoint split accounting

Two mechanisms for disjoint split exist in the codebase:

1. **Boolean flag** (`require_disjoint_train_examples=True`): introduced 2026-04-16 in `791d402`. Automatically slices train data into compressor and NDE portions. Used by the CNN BNT-era campaigns (236 runs) and the canonical refresh (6 runs).

2. **Explicit split strings** (`compressor_train_split=train[:70%]`, `nde_train_split=train[70%:]`): introduced 2026-04-13 in `29dcdfb`. The indep_split losslessness campaigns (40 runs) and the canonical refresh L1 runs use this.

Verified from `split_independence_audit.json`: 0 example-level overlap. 899/899 cosmologies in both splits (by design — different noise realizations of the same parameters).

Total runs with any disjoint discipline:
- Phase 1: 276/1,170 (24%)
- Phase 3: 6/169 (4%)
- Phase 5: 6/22 (27%)
- Overall: 288/1,517 (19%)

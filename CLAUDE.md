# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Simulation-based inference (SBI) for weak gravitational lensing cosmology. It learns low-dimensional summaries from convergence maps (CNN-VMIM compressor or wavelet L1/L1-VMIM statistics) and feeds them to a conditional RealNVP flow (or `jaxili` NPE) to infer `theta = [Omega_m, sigma_8, w0, h0, n_s, Omega_b]`.

**Current focus (2026-06):** the **flat-sky patch-local** wavelet-L1 vs CNN-VMIM comparison on 10°/80px tomographic maps (with patch-local cross-maps), written up as an A&A paper at `~/papers/L1_vs_CNN_Tomographic_SBI/` (its `REVIEW.md` = manuscript state). The message spine is `scripts/sbi/results/exploratory/flatsky_cross_2026_06/PAPER_MESSAGES.md`.

**Paper framing (decided 2026-06-19, supersedes the earlier "showcase the journey" plan):** present the work as **robust results**, not a failures/pitfalls/journey narrative — now that M1-v2 lands as expected, carry *much less* "things that failed" and instead show that everything was investigated and optimised thoroughly and the results hold. KEEP the methodological findings (leakage → patch-local; NDE estimator-effect / near-sufficiency; BNT frame-artifact) but frame them as robust results, not traps. Plan: `~/papers/L1_vs_CNN_Tomographic_SBI/HANDOFF_PAPER_M1v2_REFRAME_2026-06-19.md`.

**Current science — the headline (2026-06-15; supersedes everything older):**
- **Auto-only: tie.** Patch-local cross-maps use two operators: the *convolution* (≈0 gain — it is a lag-space re-encoding of two-point info, CLT-compressed to a few modes at 10°, seed-fragile) and the *pointwise product* κ_i·κ_j (= ξ_ij; +20% for L1, since its one-point moments are genuine non-Gaussian joint moments ⟨κ_iⁿκ_jⁿ⟩). The conv≈0 / product+20% ranking is a **population-median** statement (it flips on single-obs).
- **L1-vs-CNN (M1):** with best-effort NDE on **both** sides (CNN = ResNet18 + sbi_lens RealNVP; L1+product = VMIM-compress → the *same* RealNVP), the optimal CNN leads L1+product by **~5–9% FoM3 (σ(w0) matched), calibrated — but the gap is an estimator effect (RealNVP vs MAF: +30% on the identical 10-D summary), not a representation gap ⇒ L1+product is near-sufficient.** The earlier full-sphere harmonic "CNN wins ~2–3×" was ≈92% **leakage**; the common-MAF "CNN does not outperform L1" under-served the CNN.
- **BNT (M3):** per-channel L1 **collapses** under BNT (~0.15–0.26×); the channel-mixing CNN is **near-lossless** (~0.93–0.96×). The inflation is a **frame artifact**, not lost information — one fixed rotation of the nulled maps recovers the full no-BNT FoM (1.06/1.01). Mechanism = *no-deep-direction* frame + *mix-then-marginalize* irreversibility (closure criterion P7c; proofs P1–P7c in `BNT_THEORY_DEEP_DIVE.md`). The old "BNT correlates the shape noise / lowers per-map S/N" story is **falsified** as the mechanism. NB whitening is diagnostic (it un-does the per-slice cuts), not a practical analysis frame; the practical BNT-lossless route is a joint compressor.
- **Joint-ℓ1 refinement (2026-06-22; refines M1 + M3, does not overturn either):** the *joint* wavelet ℓ1 (across-channel coefficient histogram, the complete cross-correlation statistic of which the products κ_i·κ_j are only the 2nd-moment slice) → VMIM 10-D → the same RealNVP is the **cleanest** M1 statement: a **3-compressor deep ensemble** (seeds 41/42/43, the principled fix for amortized-SBI over-confidence — *not* finer binning) gives a **calibrated TIE** with the CNN (FoM3 **3371 ≈ CNN 3326**, clean-PASS both bases), versus the analytical ℓ1+product's pass-*with-caveat* 3270≈3293. Under BNT it sharpens M3: joint-ℓ1 retains **0.72** of its no-BNT FoM (calibrated 2424) vs products **0.26** vs CNN **0.96** — i.e. the joint statistic captures ~3× more BNT-surviving cross-correlation than products. Completeness/calibration trade-off: products 3045 → joint-ℓ1 3371/3754 → full-4D 4501 (FAIL) → pair-2D 4864 (FAIL, count-hists over-fit). See `analytical_nde_match/RESULT_JOINTL1_ENSEMBLE.md` and `JOINT_L1_DEFINITION_AND_THEORY.md`.

**Read first** for current state: `PAPER_MESSAGES.md` (spine), `FLATSKY_{CROSS,CNN,BNT}_RESULT.md`, `analytical_nde_match/RESULT_ANALYTICAL_NDE_MATCH.md` (M1, ℓ1+product), `analytical_nde_match/RESULT_JOINTL1_ENSEMBLE.md` + `JOINT_L1_DEFINITION_AND_THEORY.md` (M1/M3 joint-ℓ1 refinement, the cleanest current statement), `BNT_THEORY_DEEP_DIVE.md` (BNT proofs/interpretation). Parked (not in paper): M4 (BNT cut-space rescue, ~1.07× at realistic cuts), M5 (joint one-point stats — "broadly comparable", GATE-C caveated), and the 2D1D Haar wavelet (a tested, understood negative).

**Superseded / historical — do NOT cite as current science:** the full-sphere harmonic "CNN wins ~2–3×" (≈92% leakage); the common-MAF "CNN does not outperform L1"; the 2026-05-15 noise-model framing; `HANDOFF.md`, `HARMONIC_L1_VS_CNN_*`, `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`, `CLAUDE_CODE_HANDOFF.md`.

## Environment and external dependencies

- **Conda env:** `jaxili` (always prefix commands with `conda run -n jaxili python ...`). The virtualenv directory `learn2map/` in the repo is legacy and gitignored — do not install into it.
- **Local PyTorch extension:** `wl_stats_torch` at `/home/tersenov/software/wl_stats_torch` — hard-coded in the L1 pipeline (`_WL_STATS_PATH`). L1 scripts compute summaries in PyTorch on GPU while the flow runs in JAX; both devices are used in the same process.
- **Datasets:** CosmoGridV1 at `/home/tersenov/CosmoGridV1/` (`CosmoGridV1_metainfo.h5`, `stage3_forecast/fiducial/...`). Hard-coded as defaults in scripts.
- **Pip install:** `pip install -e .` registers `learn2map` and pulls `jax-cosmo`, `numpyro`, `lenstools`, and `sbi_lens` (from git).
- **TFDS builders** live in `scripts/sbi/tf_dataset_nbody_tomo*.py` and register via `import tf_dataset_nbody_tomo as _tomo_builder` — TFDS names used by the pipeline are `NbodyCosmogridDatasetTomo/grid` (10 deg / 80 px) and `NbodyCosmogridDatasetTomo/grid_20deg_160px[_nonoverlap48]` (20 deg / 160 px).

## GPU allocation (project rule — updated 2026-06-10)

**Allowed GPU pool: 0, 1, 2 — GPU 3 never.** (Supersedes the 2026-05-19 "GPU 1 only" rule.) Other users share GPUs 0–2: **check `nvidia-smi` for foreign tenants before every launch**, pick devices that won't trample them, and surface contention to Andreas instead of squeezing in beside a busy tenant. Memory caps: up to ~100% when sole tenant on 1–2 GPUs, ~75%/GPU when occupying all three; when packing N of our jobs on one GPU, set per-job `XLA_PYTHON_CLIENT_MEM_FRACTION ≈ 0.9/N`. **CPU budget: ≤50 workers total** across our processes (~120 cores on the host). Orchestration should be written to **maximize GPU utilization** — pack multiple dispatch-bound jobs per GPU rather than defaulting to 1 job/GPU (see `scripts/sbi/results/exploratory/flatsky_cross_2026_06/PIPELINE_AUDIT_2026-06-10.md` §d).

## Code layout (what is actually live)

- `scripts/sbi/` — **all live code** for the current scientific work. This is the only tree to edit.
- `learn2map2/datasets/` — older TFDS dataset builders; largely superseded by `scripts/sbi/tf_dataset_*` but still imported in places.
- `learn2map/` — a Python virtualenv, **not source code**. Never edit.
- `notebooks/sbi/` — exploratory/publication notebooks; mostly read-only for inference runs.
- `skills/sbi/SKILL.md` — the project's own SBI operating protocol (workflow, logging, anti-clutter policy). Treat it as authoritative procedure.
- `.worktrees/` — pre-existing git worktrees for parallel experiments; do not delete.

## Pipeline architecture (mental model)

Always reason in three layers and keep them separated — do not blame "the pipeline" without naming a layer:

1. **Summary extraction**
   - `npe_cnn_nbody_tomo.py` / `npe_cnn_jaxili_nbody_tomo.py` — CNN-VMIM compressor; arch family via `--compressor-arch {plain,resnet_small,resnet18,resnet34,resnet50,resnet50_gn}`. On 10-channel harmonic-cross input, **always use `resnet50_gn`** (or `plain`) — stock BatchNorm ResNets collapse FoM3 to ~700 because BN running stats average across cosmology-mixed batches; GroupNorm restores ~22 k. Memory: `project_resnet_bn_contamination.md`.
   - `npe_l1norm_nbody_tomo.py` / `npe_l1norm_jaxili_nbody_tomo.py` — wavelet L1 datavector on per-bin auto maps (PyTorch `WLStatistics`).
   - `npe_l1norm_cross_jaxili_nbody_tomo.py` — wavelet L1 on the 4 auto + 6 cross map channels (the harmonic-cross arm of the current investigation). The canonical FoM3 formula `FoM3 = 1/√det(C_3)` lives here at `:1737-1759`. **For any cross-channel use, pass `--cross-noise-model channel_empirical_global`** — the default `auto_scalar` is the broken v1 model and is preserved only for reproducing pre-2026-05-15 results.
   - `npe_l1vmim_nbody_tomo.py` / `npe_l1vmim_jaxili_nbody_tomo.py` — L1 feeds a VMIM MLP compressor.
2. **Preprocessing:** log1p / z-score / clipping; PCA only when explicitly asked (L1-VMIM is preferred for compression).
3. **Density estimator `p(theta | summary)`:** in-repo conditional RealNVP (via `sbi_lens.normflow`) or `jaxili` NPE (scripts with `_jaxili_` in the name).

Shared pipeline invariants:
- Parameter order is fixed `[Omega_m, sigma_8, w0, h0, n_s, Omega_b]`, with `h0 = H0/100` applied in preprocessing (`theta[3] /= 100`).
- In the CNN BNT path, **shape noise is injected before BNT** (`apply_bnt_*` in `scripts/sbi/bnt_utils.py`). `--apply-bnt` requires full tomography (`--nbins 4 --tomo-bin-indices 1,2,3,4`); `validate_bnt_configuration` enforces this.
- Paired BNT/no-BNT training (`--compressor-paired-bnt-nobnt-consistency`) returns dict features `{maps_nobnt, maps_bnt}`; `compress_dataset(..., paired_map_view=...)` must select one view, otherwise it crashes with `KeyError: 'maps'`. This is a known fix point; preserve the `paired_map_view` plumbing when editing.

## Campaign orchestration

Multi-seed / multi-GPU experiments are driven by `scripts/sbi/run_*.py` scripts, which shell out to the per-run entrypoints above. Key ones:

- `run_cnn_bnt_losslessness_campaign.py`, `run_cnn_noise_curriculum_campaign.py` — BNT recovery campaigns.
- `run_cnn_l1_systematic_sweep.py`, `run_cnn_tomo4_opt_sweep.py`, `run_l1_jaxili_tomo4_opt_sweep.py`, `run_l1vmim_tomo4_opt_sweep.py` — systematic sweeps.
- `run_bnt_tomo4_study.py`, `run_baryon_bias_tomo4_study.py`, `run_nobnt_tomo_bins_crosscorr_study.py`, `run_optimal_nobnt_crosscorr_benchmark.py` — focused studies.
- `run_sbc_cnn_nobnt.py`, `run_sbc_harm_l1_nobnt.py` — Simulation-Based Calibration runners. They consume a trained checkpoint and produce ranks in `scripts/sbi/results/diagnostics/sbc_*/n<N>_m<M>_seed<...>/`. Note: by default these dump ranks, not full posteriors — if you need posterior samples (e.g. for TARP), add the dump flag or instrument the loop.
- `run_tarp_dumps_campaign.py` — multi-arm TARP posterior-sample dump orchestrator (arm-agnostic, accepts new arms via the same `ARMS` dict pattern as `run_cross_only_campaign.py`).
- `run_cross_only_campaign.py` — **the cross-only campaign orchestrator (2026-05). Phase 1 builds L1 + CNN Stage A (shared compressor with `--exit-after-compress`); Phase 2 runs CNN Stage B NDE per seed against the cached compressed dataset (~100 s/job).** Used for both v1 (`cross_only_campaign/`) and v2 (`cross_only_campaign_v2_chsigma/`); switch via `--cross-noise-model`.
- `compare_cross_only.py` — per-campaign analysis: discovers all (arm, seed, dim), pools posteriors, computes FoM3, writes `SUMMARY.md` + GetDist filled-contour corner figures.
- `compare_probes_configs.py` — cross-campaign 3 probes × 3 input configs overlay: takes auto-only / cross-only / auto+cross posteriors from their respective campaign dirs and emits one figure per probe + a unified FoM3 markdown table.
- `diagnose_cross_only_{inputs,tighter_snr,channel_aware_noise,signal_check}.py` — 4 one-pass diagnostic scripts that document the L1 cross-channel noise-model bug investigation (the smoking-gun plots).
- `build_full_sphere_cross_cache.py`, `diagnose_cross_maps.py`, `diagnose_full_sphere_cross_maps.py` — utilities supporting the harmonic-cross channels (cache builder + sanity checks).

Orchestrators accept `--gpus 0,1,2,3` and an optional `--xla-mem-fraction-by-gpu 0:0.75,1:0.30,...` per-GPU memory cap map. Results go under `scripts/sbi/results/{final,exploratory,dryruns,diagnostics}/` — follow that taxonomy (see `skills/sbi/SKILL.md`).

For example run commands (single-run L1/CNN with typical flags, sweep invocations), see `scripts/sbi/l1_jaxili_run_commands.txt` and `scripts/sbi/l1_vmim_run_commands.txt`.

## Common commands

```bash
# Compile check (fast sanity pass before launching GPU work):
conda run -n jaxili python -m py_compile \
  scripts/sbi/npe_cnn_nbody_tomo.py \
  scripts/sbi/run_cnn_noise_curriculum_campaign.py

# Typical CNN+NPE single run (4 tomo bins, 20 deg / 160 px, no BNT):
conda run -n jaxili python scripts/sbi/npe_cnn_nbody_tomo.py \
  --cuda-visible-devices 0 --map-kind nbody \
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px \
  --field-size 20 --field-npix 160 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --seed 42 --plot

# Same, but apply BNT after shape noise:
... --apply-bnt   # only valid with full tomo4

# Reuse a trained flow checkpoint, sample only:
... --no-train --npe-samples 100000
```

There is no test suite and no lint configuration — integrity is verified by `py_compile`, W&B-logged training runs, and the per-campaign comparison reports.

## Working-tree and git discipline

The tree is chronically dirty (notebooks with outputs, caches, `__pycache__`, campaign results under `scripts/sbi/results/`). Rules from `CLAUDE_CODE_HANDOFF.md`:

1. **Never `git add .` or `git add -A`.** Stage files explicitly by path.
2. Do not delete or rewrite pre-existing dirty files unless the user explicitly asks.
3. Do not run destructive cleanup (`git reset --hard`, mass deletes, `git clean`).
4. Do not commit generated artifacts (results, caches, `.pkl` checkpoints, `*.pyc`) unless asked.
5. Active development branch is `analytical-nde-match-2026-06` (the M1/joint-ℓ1 work); the "main" PR target is `main`. Earlier branches: `cnn-nde-optimization-2026-06` (CNN-opt sweep), `l1-cross-maps` (the harmonic-cross arm), `bnt-parity-techniques` (the BNT campaign).

## Project protocol (from `skills/sbi/SKILL.md`)

When running SBI experiments, follow the workflow in `skills/sbi/SKILL.md`: lock an objective with a decision metric, freeze a config fingerprint, vary one factor at a time, validate cache/checkpoint/preprocess compatibility before drawing conclusions, and report results in the mandated `Objective / Configuration fingerprint / Quantitative outcomes / Robustness / Scientific conclusion / Minimal next action` structure.

Claim acceptance requires apples-to-apples comparison, reproducibility from saved artifacts, stability across seeds, and having ruled out cache/preprocess/compressor mismatches. Keep per-run artifacts minimal: posterior `.npy`, `.meta.json`, metrics CSV/JSON (including FoM), corner plot PDF, and a manifest. W&B logging is expected on non-dry runs.

## Felt / Ralph operating conventions

Adopted 2026-05-22 after the cnn-auto-push-18-20-2026 retrospective. Every long-running campaign fiber (constitution) must follow these or explicitly justify deviation in the constitution body:

1. **Declare ONE primary metric in the constitution.** Pick `pooled_fom3` OR `mean_of_seeds_fom3` OR `per_seed_min_fom3` — not "headline 25k" with a different number used by the keep-rule. Every iteration's keep/discard decision uses this metric. STATUS.md headline numbers must match it. Mixing metrics is the failure mode that made the auto-only-vs-cross-push overlay land on the wrong cross-arm baseline.

2. **Declare a budget AND a plateau-stop in the constitution's "Done condition" stanza.** Format: "auto-close when N consecutive iters land within ±X% of current best on the primary metric, OR when iteration count reaches M, whichever is first." Ralph survey reads this at the start of each iteration and exits without launching work when the trigger fires. Default for hyperparameter sweeps: N=3, X=5%, M=30.

3. **`ship-blocker` tag is reserved for fibers that pause hyperparameter iteration.** When a fiber tagged `ship-blocker` is open, Ralph must either ship its fix in the current iteration or explicitly demote-with-rationale before launching new training. The `[[cnn-auto-compressor-last-not-best-ckpt]]` bug sat unfixed across the entire cnn-auto-push campaign because it lacked this tag.

4. **Constitution must include a "Loop Status (live)" stanza near the top** when in a wait-for-Andreas or wait-for-compute state. List the 2–3 concrete things that unblock work (e.g. "(a) Andreas appends CEILING CONFIRMED; (b) Andreas requests a new branch; (c) Andreas answers methodology fiber"). Cold-read Ralph iterations that find none of the conditions exit with `kill $PPID` and no commits. This is what stops the polish-make-work pattern that ate iters 17–20 of the auto-only campaign.

5. **Self-review every 5 iterations.** A loop-self-review iteration produces a `loop_review.md` append: marginal-info-gained / current-best-delta / wall-time-used / "should this loop continue?" verdict. If the verdict is "no" two reviews in a row, auto-close. Goes in `<run-dir>/loop_review.md`, not STATUS.md (STATUS.md is for substantive findings).

6. **Compress STATUS.md proactively.** Use `scripts/sbi/results/exploratory/tools/compact_status.py` (see "Other tooling" below) to collapse the calibration-ledger / lesson-tracking sections into a digest when STATUS.md exceeds ~30 KB. Keep the last 10 substantive events + current best + open ship-blockers + next 3 planned moves at the top; archive the rest.

7. **Constitutions must declare which autoresearch driver they use and pin its checkpoint policy.** The drivers (`autoresearch_cnn-auto-push/run_arm.py`, `autoresearch_cnn-auto-cross-push/run_arm.py`) accept `--compressor-checkpoint-policy {best_val,last_step}`. New campaigns default to `best_val`; campaigns continuing a historical baseline pin to `last_step` and say so explicitly.

## Other reference docs in the tree

- **Current-state docs** (the read-first list is in "What this repo is" above): `scripts/sbi/results/exploratory/flatsky_cross_2026_06/PAPER_MESSAGES.md` (message spine), repo-root `FLATSKY_{CROSS,CNN,BNT}_RESULT.md`, `…/analytical_nde_match/RESULT_ANALYTICAL_NDE_MATCH.md` (M1), `…/BNT_THEORY_DEEP_DIVE.md` (BNT proofs/interpretation); the manuscript at `~/papers/L1_vs_CNN_Tomographic_SBI/` (`REVIEW.md`).
- **HISTORICAL / superseded** (background only; their numbers are no longer the science answer): `HANDOFF.md`, `HARMONIC_L1_VS_CNN_INVESTIGATION_BRIEF.md`, `HARMONIC_L1_VS_CNN_SESSION2_HANDOFF.md`, `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`, `CLAUDE_CODE_HANDOFF.md`.
- `Harmonic_cross_maps.md`, `Flat-Sky_Tomographic_Cross_Maps.md` — definitions and conventions for the cross-map channels.
- `SBI_L1_CNN_PIPELINE_DETAILED.md` — step-by-step audit trail of all four pipelines.
- `L1_CONTOUR_INVESTIGATION_LOG.md`, `L1_FIXES_VALIDATION_REPORT.md`, `L1_VMIM_FINAL_CONCLUSIONS.md` — L1 diagnosis and fix history.
- `BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md` — conceptual note on BNT inflation mechanisms.
- `scripts/sbi/results/final/paper_sbi_consolidation/FINAL_SCIENTIFIC_REPORT.md` — the top-of-campaign synthesis cited by the knowledge base.

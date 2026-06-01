# INVENTORY.md — repo audit, 2026-05-17 (read-only)

Audit pass over `cnn_sbi` on branch `l1-cross-maps` (HEAD `b66b744`). Goal:
build a single map of every experiment, script, notebook, and artifact so the
upcoming reorganization has a starting picture. Nothing was modified.

Where I couldn't confirm something from code, git, or docs alone, I say so
under **Unclear** rather than guess.

---

## Experiments run

Grouped by branch / scientific direction; each entry cites its primary code
and result locations. Numeric headlines are quoted from the matching report
under `scripts/sbi/results/` or the root `*.md` docs.

### 1. L1 stabilisation and "banana" reproducibility (Feb–Mar 2026)

- **Question.** Could we reproduce the historical tight `(Ω_m, σ_8)` L1
  contour from the legacy pipeline? Why did the rerun produce a near-circular
  contour instead?
- **Files.** `scripts/sbi/npe_l1norm_nbody_tomo.py` (fix site), older
  `scripts/sbi/results/exploratory/investigate_old_script/`,
  `investigate_cnn_dim12/`, plus the diagnostics dir.
- **Docs.** `L1_CONTOUR_INVESTIGATION_LOG.md`, `L1_FIXES_VALIDATION_REPORT.md`.
- **Status.** Complete. Conclusion (`PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` §1):
  the legacy artifact's banana was non-reproducible — different preprocessing
  history. Fixes (SNR calibration policy, coarse-mean toggle, cache metadata
  validation) shipped; no banana, but L1 pipeline is now stable. **Negative**
  reproducibility outcome, **positive** code-hygiene outcome.

### 2. L1-VMIM compressor sweep (Mar–Apr 2026)

- **Question.** Can a VMIM-trained MLP compressor on L1 features approach the
  no-compression L1 baseline?
- **Files.** `scripts/sbi/npe_l1vmim_nbody_tomo.py`,
  `scripts/sbi/npe_l1vmim_jaxili_nbody_tomo.py`,
  `scripts/sbi/run_l1_vmim_systematic_sweep.py`,
  `scripts/sbi/run_l1vmim_tomo4_opt_sweep.py`. Many sweep dirs under
  `scripts/sbi/results/exploratory/l1_vmim_runs*` (per `results/INDEX.txt`).
- **Docs.** `L1_VMIM_FINAL_CONCLUSIONS.md`.
- **Status.** Complete. **Positive within calibration constraints**: best run
  hits `std_ratio=1.019` vs no-compression L1, Mahalanobis 1.303
  (`l1_vmim_tomo4_20deg160_seed202_flowonly.npy`). Tighter-than-baseline
  posteriors were possible but biased, so a calibration-constrained selection
  rule was needed.

### 3. No-BNT / BNT / baryonified consolidation matrix (Apr 2026)

- **Question.** How do L1, L1-VMIM, and CNN compare across no-BNT, BNT, and
  baryon-contaminated regimes in the tomo4 setup? What is the headline FoM3
  table for the paper?
- **Files.** Driver scripts `run_cnn_l1_systematic_sweep.py`,
  `run_cnn_tomo4_opt_sweep.py`, `run_l1_jaxili_tomo4_opt_sweep.py`,
  `run_l1vmim_tomo4_opt_sweep.py`, `run_bnt_tomo4_study.py`,
  `run_baryon_bias_tomo4_study.py`, `run_nobnt_tomo_bins_crosscorr_study.py`,
  `run_optimal_nobnt_crosscorr_benchmark.py`. Results under
  `scripts/sbi/results/final/paper_sbi_consolidation/{nobnt_final_matrix,
  bnt_comparison_tomo4, baryonified_appendix}/`.
- **Docs.** `scripts/sbi/results/final/paper_sbi_consolidation/FINAL_SCIENTIFIC_REPORT.md`,
  `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` §3, `BNT_TOMO4_*.md`,
  `OPTIMAL_NOBNT_CROSSCORR_*.md`, `TOMO_BIN_CROSSCORR_*.md`,
  `BARYON_BIAS_TOMO4_*.md`.
- **Status.** **Largely superseded** (see Experiments §6, §7, §8). QC passed
  (55/55 no-BNT, 22/22 BNT, 180/180 baryon) but every CNN row in the matrix
  is now flagged as inflated by the mass-sheet-degeneracy leak (KB §13).
  Numbers should not be quoted as-is.

### 4. CNN BNT-losslessness retraining (Apr 2026)

- **Question.** Can compressor capacity / training choices close the BNT vs
  no-BNT FoM gap?
- **Files.** `run_cnn_bnt_losslessness_campaign.py` + result tree
  `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign*/`.
- **Docs.** `CNN_BNT_LOSSLESSNESS_RETRAIN_REPORT.md`, KB §4.
- **Status.** Complete. **Positive** — advanced cdim=10 hit `BNT/noBNT
  FoM=0.91, inflation=1.03`. Compressor capacity is the dominant lever;
  NDE-side capacity alone does not help. Caveat (KB §13.4): absolute FoM
  values pre-`--zero-mean-maps` are inflated, ratios mostly survive.

### 5. Multipatch + independent-split CNN-BNT (Apr 2026)

- **Question.** Does decoupling the compressor and NDE training sets, or
  using non-overlapping patches, improve BNT parity?
- **Files.** `run_cnn_bnt_losslessness_campaign.py` (multipatch /
  indep_split variants). Result dirs `..._multipatch_*`, `..._indep_split_*`
  under `paper_sbi_consolidation/`.
- **Docs.** KB §5.
- **Status.** Complete. **Negative** — multipatch and indep-split helped
  certain width diagnostics but did not beat the random25 cdim10 reference
  on global parity. Best `indep_split_advanced_cdim10_long120k` hit FoM
  ratio 0.846 vs 0.907 reference.

### 6. CNN noise-curriculum + parity-techniques (Apr 2026)

- **Question.** Do noise curricula, paired BNT/no-BNT consistency loss, or
  domain-adversarial heads stabilise BNT/no-BNT parity?
- **Files.** `run_cnn_noise_curriculum_campaign.py`, plus parity flags inside
  `npe_cnn_nbody_tomo.py` (`--compressor-paired-bnt-nobnt-consistency`,
  adversarial head). Results under
  `paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/` and
  `cnn_bnt_parity_campaign/`.
- **Docs.** `FINAL_NOISE_CURRICULUM_REPORT.md`, KB §6, §9; phase A/B/C reports
  inside the parity campaign tree; `CLAUDE_CODE_HANDOFF.md`.
- **Status.** Complete. **Inconclusive / largely negative.** Curriculum helped
  ResNet18 (`0.43 → 0.87` FoM ratio) but hurt plain. Paired-consistency and
  consistency+adversarial pilots improved some cells but **did not survive
  5-seed confirmation** (Phase C). No invariance trick generalised stably.

### 7. ResNet variants (Apr 2026)

- **Question.** Does a deeper compressor (ResNet18/34/50) beat plain CNN on
  BNT parity?
- **Files.** ResNet code paths inside `npe_cnn_nbody_tomo.py`
  (`--compressor-arch {plain, resnet_small, resnet18, resnet34, resnet50,
  resnet50_gn}`). Result dirs `cnn_bnt_resnet_split_campaign/`,
  `cnn_resnet34_50_zm_cdim1224/` (62 GB), `cnn_resnet50_zm_sweep/` (33 GB).
- **Docs.** `CNN_BNT_RESNET_SPLIT_CAMPAIGN_REPORT.md`,
  `EXTENDED_RESNET_COMPARISON_REPORT.md`, KB §7; memory
  `project_resnet_bn_contamination.md`.
- **Status.** Complete with one strong actionable. **Negative** for
  beating plain on BNT parity. **Critical positive side-finding (May 2026):**
  stock-BN ResNet50 on 10-channel harmonic input collapses FoM3 to ~700
  because BN running stats average across cosmology-mixed batches;
  `resnet50_gn` (GroupNorm) recovers to ~22k. Default for multi-channel
  harmonic now: `resnet50_gn`.

### 8. Zero-mean-maps / mass-sheet-degeneracy correction (2026-04-21)

- **Question.** Is the CNN-VMIM compressor leaking the per-channel spatial
  mean (an unphysical, mass-sheet-degeneracy-bound signal)?
- **Files.** `--zero-mean-maps` flag in `npe_cnn_nbody_tomo.py`; paired-BNT
  branch logic.
  Results: `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/` and
  follow-up `cnn_extended_train_zm/`, `cnn_resnet50_zm_sweep/`,
  `cnn_resnet34_50_zm_cdim1224/`, `cnn_vmim_target_stability/`.
- **Docs.** `zero_mean_maps_parity_check/SUMMARY.md`, KB §13; memory
  `project_cnn_vmim_mass_sheet_leak.md`.
- **Status.** Complete. **Very significant negative finding** — every
  pre-2026-04-21 CNN posterior overstates constraining power by ~2× in
  marginals, ~25–32× in FoM3. KB §3 and §4 numbers retracted in strong form.
  `--zero-mean-maps` is now the default for any reportable CNN run. Note:
  flag default in CLI is still OFF for backwards-compat (must opt in).

### 9. Tomographic cross-maps L1 (flat-sky, Apr 2026)

- **Question.** Does adding flat-sky FFT cross-bin κ_i × κ_j channels to the
  L1 datavector improve FoM3?
- **Files.** `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py` (with
  flat-sky FFT cross route, now disabled for production), older flat-sky arm
  in `scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_cross_*`.
- **Docs.** `Flat-Sky_Tomographic_Cross_Maps.md`, `Harmonic_cross_maps.md`,
  KB §14 (intro).
- **Status.** Complete and **superseded** (see Experiment 10). Reading at the
  time: BNT pct=1 cross channels gave +46% FoM3 vs auto-only; no-BNT cross
  channels gave **−12%** (i.e. seemingly hurt). The "−12%" was an artefact of
  the flat-sky FFT-on-patches construction discarding large-scale cross
  power.

### 10. Full-sphere harmonic cross-maps (May 2026, branch `l1-cross-maps`)

- **Question.** Are tomographic cross-maps informative when built à la Zürcher
  in `a_ℓm` space (full sphere) and then patched, vs flat-sky on patches?
- **Files.** `scripts/sbi/build_full_sphere_cross_cache.py`,
  `scripts/sbi/diagnose_full_sphere_cross_maps.py`,
  `scripts/sbi/diagnose_cross_maps.py`. Cache:
  `scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid/`
  (**623 GB**, 18 186 npz files, both regimes).
  L1 entry point: `--full-sphere-cross-cache` flag in
  `npe_l1norm_cross_jaxili_nbody_tomo.py`.
- **Docs.** `cross_maps_campaign/cross_summary/harmonic_results.md`,
  KB §14, `HARMONIC_L1_VS_CNN_INVESTIGATION_BRIEF.md`.
- **Status.** Complete on infrastructure; scientific picture **half settled**.
  Initial 3-seed pool: harm_cross_bnt FoM3 = 5 161 (+554% vs auto-only),
  harm_cross_nobnt FoM3 = 59 243 (+351%). The no-BNT headline was
  subsequently **flagged as artefact** by SBC + held-out cosmology checks
  (σ_8 sensitivity inverted near fiducial; SBC h0 and Ω_b χ² >> 50). The
  harm_cross_bnt 5 161 number is the defensible headline.

### 11. TARP joint-coverage refinement (May 2026)

- **Question.** Are the L1 harm-cross posterior contours genuinely tighter,
  or are they just over-confident?
- **Files.** `scripts/sbi/run_tarp_dumps_campaign.py`,
  `scripts/sbi/run_tarp_coverage.py`. Results
  `scripts/sbi/results/diagnostics/tarp_harm_cross/{curves,dumps,figures,logs}/`.
- **Docs.** `HARMONIC_L1_VS_CNN_INVESTIGATION_NOTES.md` §2; memory
  `project_harmonic_cross_overturns_flatsky.md`.
- **Status.** Complete for **v1** (auto-σ noise model). 17 (arm, seed) cells,
  N=200, M=2000, 200-bootstrap. 3-D ECP for L1 harm-cross is only 1–2 pp
  under-covered at α=0.9/0.95. **Inconclusive on v2** (channel-aware-σ)
  — not yet run. See Open threads.

### 12. SBC (simulation-based calibration) for CNN and L1 (May 2026)

- **Question.** Are the harm-cross posteriors marginally calibrated?
- **Files.** `run_sbc_cnn_nobnt.py`, `run_sbc_cnn_harm_cross_nobnt.py`,
  `run_sbc_harm_l1_nobnt.py`. Results
  `scripts/sbi/results/diagnostics/sbc_{cnn_nobnt, harm_l1_nobnt}/`.
- **Status.** Complete for L1 harm_cross v1 (N=1000, M=2000): σ_8 z=−1.78,
  Ω_b χ²=117 — marginally miscalibrated. Joint TARP coverage is much better
  (Experiment 11). v2 SBC not yet run.

### 13. Cross-only test: 6 cross channels alone (May 2026)

- **Question.** Of the auto+cross gain, what comes from cross channels alone
  (i.e. is the L1 advantage intrinsic to the cross channels, or combinatorial
  with the autos)?
- **Files.** `scripts/sbi/run_cross_only_campaign.py` (2-phase: L1 +
  shared-compressor CNN Stage A in parallel, then CNN Stage B per seed).
  `scripts/sbi/npe_cnn_nbody_tomo.py:--channel-mode {auto_cross,cross_only}`
  and `--exit-after-compress`.
  Results: `scripts/sbi/results/exploratory/cross_only_campaign/` (v1, 27 GB)
  and `cross_only_campaign_v2_chsigma/` (v2, 27 GB, **headline**).
- **Docs.** `cross_only_campaign{,_v2_chsigma}/SUMMARY.md`,
  `probes_configs_comparison/fom3_probes_configs.md`, `HANDOFF.md`,
  memory `project_cross_only_l1_loses.md` (v1) +
  `project_l1_noise_model_correction.md` (v2).
- **Status.** Complete v2. **Cross-only: CNN beats L1** (resnet50_gn d10 FoM3
  25 830 vs L1 16 070, pooled). Auto+cross L1 advantage drops from ~3× to
  ~1.5× under v2. Negative for the "L1 dominates cross channels" hypothesis.

### 14. L1 cross-channel noise-model bug + fix (2026-05-15)

- **Question.** Why does v2 differ from v1?
- **Files.** `npe_l1norm_cross_jaxili_nbody_tomo.py`:
  `calibrate_channel_noise_sigma_from_harmonic_cache` (~line 430), `channel_scale`
  threading, `--cross-noise-model channel_empirical_global` flag.
  Diagnostic scripts `scripts/sbi/diagnose_cross_only_{inputs,tighter_snr,
  channel_aware_noise,signal_check}.py`.
- **Docs.** Memory `project_l1_noise_model_correction.md`, `HANDOFF.md` §3.
- **Status.** Complete. The bug: auto pixel-σ was used as the SNR
  denominator for all channels, including the ~10⁴–10⁵× smaller cross
  channels. With channel-aware σ the L1 auto+cross FoM3 fell from ~65k to
  34k (−48%); cross-only rose 12k → 16k (+33%). **Reframes the
  "L1 wins 3× on auto+cross" headline as v1-noise-model artefact.**

### 15. 3 probes × 3 inputs overview (May 2026, current)

- **Files.** `scripts/sbi/compare_probes_configs.py`,
  `scripts/sbi/compare_cross_only.py`. Output
  `scripts/sbi/results/exploratory/probes_configs_comparison/`.
- **Status.** Live. Produces the 3×3 FoM3 table cited in HANDOFF §2 and
  in `probes_configs_comparison/fom3_probes_configs.md`.

### Other experiments referenced but largely historical / superseded

- `cnn_lossiness_check/` (144K) — small probe. Status unclear.
- `cnn_vmim_target_stability/` (1.3 GB) — VMIM target dim sweep. Complete,
  superseded.
- `cnn_extended_train_zm/` (807 MB) — extended CNN training in demeaned
  pipeline. Complete.
- `harmonic_vs_cnn_investigation/` (48K) — investigation outputs leading to
  the σ_8 inversion diagnosis. Complete.
- `systematic_runs_cnn_retrain_proper/`, `systematic_runs_l1_rerun_proper/`,
  `systematic_runs_l1_snr10_rerun/` — older systematic sweeps (218 / 210 /
  204 MB). Complete, partially superseded by experiments above.

---

## Code structure

### Core pipeline modules (should survive cleanup)

All under `scripts/sbi/`. These are the entrypoints invoked by orchestrator
`run_*.py` scripts (verified by `grep -E "npe_l1norm_cross_jaxili|npe_cnn_nbody_tomo|..." run_*.py`):

| File | Lines | Role |
|---|---:|---|
| `npe_cnn_nbody_tomo.py` | 3 939 | Per-run CNN-VMIM compressor + NPE entrypoint. Holds `--apply-bnt`, `--zero-mean-maps`, `--channel-mode`, `--exit-after-compress`, all compressor archs incl. `resnet50_gn`. |
| `npe_l1norm_cross_jaxili_nbody_tomo.py` | 2 875 | L1 cross-channel runner (harmonic + flat-sky cross routes). Hosts `--cross-noise-model channel_empirical_global` fix. Canonical FoM3 definition lives at lines 1737–1759 per HANDOFF. |
| `npe_l1vmim_jaxili_nbody_tomo.py` | 2 949 | L1-VMIM compressor + jaxili NPE. |
| `npe_l1vmim_nbody_tomo.py` | 2 589 | L1-VMIM with the in-repo RealNVP NDE (legacy NDE variant). |
| `npe_l1norm_nbody_tomo.py` | 1 979 | L1 (auto-only) + in-repo RealNVP. |
| `npe_l1norm_jaxili_nbody_tomo.py` | 1 667 | L1 (auto-only) + jaxili NPE. |
| `npe_cnn_jaxili_nbody_tomo.py` | 1 453 | CNN summary + jaxili NPE (eval/no-train path used by several orchestrators). |
| `bnt_utils.py` | thin | BNT matrix + `validate_bnt_configuration`. |
| `build_full_sphere_cross_cache.py` | 505 | Cache builder for harmonic cross-maps (HEALPix → SHT → `a_ℓm` cross → ISHT → 48 gnomonic patches). |
| `tf_dataset_nbody_tomo.py` | live | TFDS builder used by every modern script (`NbodyCosmogridDatasetTomo/grid`, `..._20deg_160px`, `..._nonoverlap48`). |
| `tf_dataset_nbody_tomo_BNT.py` | live | TFDS builder for BNT-applied tomo dataset (rare path). |

Orchestrator drivers (15 `run_*.py` files; each shells out to one or more of
the entrypoints above):

```
run_cross_only_campaign.py           run_cnn_nobnt_deep_audit.py
run_baryon_bias_tomo4_study.py       run_cnn_noise_curriculum_campaign.py
run_bnt_tomo4_study.py               run_cnn_noiseless_vs_noisy.py
run_cnn_bnt_losslessness_campaign.py run_cnn_tomo4_opt_sweep.py
run_cnn_l1_systematic_sweep.py       run_l1_jaxili_tomo4_opt_sweep.py
run_l1_vmim_systematic_sweep.py      run_l1vmim_tomo4_opt_sweep.py
run_nobnt_tomo_bins_crosscorr_study.py
run_optimal_nobnt_crosscorr_benchmark.py
run_sbc_cnn_nobnt.py        run_sbc_cnn_harm_cross_nobnt.py
run_sbc_harm_l1_nobnt.py    run_tarp_coverage.py  run_tarp_dumps_campaign.py
```

Analysis / comparison scripts:

```
compare_cross_only.py        compare_probes_configs.py     compare_cnn_bnt_noiseless_followup.py
analyze_baryon_bias_tomo4.py analyze_cnn_nobnt_deep_audit.py  analyze_nobnt_tomo_bins_fom.py
audit_cnn_nobnt_data_pipeline.py
plot_baryon_bias_overlays.py plot_bnt_nobnt_overlays.py    plot_tomo_bin_method_overlays.py
upload_bnt_tomo4_to_wandb.py
profile_harmonic_iter.py
```

Diagnostic scripts (cross-only noise-model investigation, smoking-gun
artefacts referenced by `memory/project_l1_noise_model_correction.md`):

```
diagnose_cross_only_inputs.py
diagnose_cross_only_tighter_snr.py
diagnose_cross_only_channel_aware_noise.py
diagnose_cross_only_signal_check.py
diagnose_cross_maps.py
diagnose_full_sphere_cross_maps.py
```

### Experimental / scratch code (probably archive)

- `scripts/sbi/nle_simple_nbody_NLE.py` (496 lines), `nle_simple_nbody_NPE.py`
  (508 lines): notebook-derived NLE / NPE drivers. **Not referenced by any
  `run_*.py`.** Predate the unified `npe_*_nbody_tomo.py` pipeline.
- `scripts/sbi/train_compressor.py` (508 lines): standalone Haiku CNN-VMIM
  trainer that imports `tf_dataset_nbody` (the non-tomo legacy builder).
  Not referenced by any `run_*.py`. Pre-`tomo` era.
- `scripts/sbi/train_compressor_tomographic.py`,
  `train_compressor_tomographic_BNT.py`,
  `train_compressor_tomographic_12features.py` (3 × ~500 lines): older
  compressor-only trainers. Not invoked by any orchestrator; the modern
  pipeline trains compressor and flow in one process via `npe_cnn_nbody_tomo.py`.
- `scripts/sbi/tf_dataset.py`, `tf_dataset_fiducial.py`, `tf_dataset_nbody.py`:
  pre-tomo TFDS builders. Only `train_compressor.py` still imports
  `tf_dataset_nbody`. The current pipeline uses `tf_dataset_nbody_tomo[_BNT].py`.
- `notebooks/sbi/nle_vmim_nbody_script.py`: scratch script in the notebooks
  dir (different starts with `\`, looks notebook-export).
- `notebooks/tf_dataset_nbody.py`: copy of the legacy builder colocated with
  the top-level notebooks.

### Dead code / broken refs

- `learn2map/`: a Python virtualenv directory, not source. Both CLAUDE.md and
  README flag it explicitly as "do not edit". `setup.py` is configured as
  `packages=["learn2map"]` — `pip install -e .` is therefore registering an
  empty package (the venv directory has no `__init__.py` matching the source
  spec). Unclear whether `pip install -e .` actually works today; commands in
  README assume it does. **Worth verifying before any clean install.**
- `learn2map2/datasets/`: three older TFDS builders (`tf_dataset.py`,
  `tf_dataset_fiducial.py`, `tf_dataset_nbody.py`). `grep -rn "learn2map2"
  scripts/ notebooks/` returns zero hits. Fully unused.
- `scripts/sbi/posterior_cnn_tomo.npy` (2.4 MB): a stray posterior in the
  scripts dir from Feb 2026. Predates the `results/` reorg. Likely
  abandoned, but I did not verify whether anything reads it.

### Duplication

- **FoM3 formula** is implemented in `npe_l1norm_cross_jaxili_nbody_tomo.py`
  (canonical, lines 1737–1759 per HANDOFF), and re-implemented in
  `compare_cross_only.py`, `compare_probes_configs.py`, several
  `analyze_*.py` and the notebooks. CLAUDE.md / HANDOFF say the duplicates
  are correct — but it is duplication.
- The four `train_compressor*.py` files share a common skeleton (load TFDS,
  build Haiku CNN, fit, save params) with bin/12-feature/BNT variations. If
  any survive cleanup, they should consolidate to one file with flags. None
  appears live, so retiring them may be cheaper.
- `tf_dataset*.py` exists in three locations: `scripts/sbi/`, `learn2map2/datasets/`,
  `notebooks/`. Only the `scripts/sbi/tf_dataset_nbody_tomo*` are live.
- The 16 notebooks under `notebooks/sbi/` share large chunks of fixture
  setup, CNN compressor instantiation, and HMC sampling (see Notebooks
  section below).

---

## Data and artifacts

### Input data

- **CosmoGridV1** at `/home/tersenov/CosmoGridV1/`. Drives every grid-arm
  run. Defaults hard-coded as `CosmoGridV1_metainfo.h5`,
  `stage3_forecast/fiducial/...` in scripts.
- **`wl_stats_torch`** at `/home/tersenov/software/wl_stats_torch`.
  `sys.path` is injected at the top of every L1 runner. PyTorch GPU
  computation runs alongside the JAX flow in the same process.
- **TFDS datasets** registered by `scripts/sbi/tf_dataset_nbody_tomo*.py`.
  Active names in production: `NbodyCosmogridDatasetTomo/grid`,
  `NbodyCosmogridDatasetTomo/grid_20deg_160px`, and
  `NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`. Builds go to
  `~/tensorflow_datasets/` by default.

### Generated artifacts (totals from `du -sh`)

- `scripts/sbi/results/exploratory/` — **970 GB**, dominated by:
  - `cross_maps_campaign/` 755 GB (mostly the **623 GB harmonic cache**
    `full_sphere_cache_grid/`).
  - `cnn_resnet34_50_zm_cdim1224/` 62 GB
  - `cnn_with_harm_cross_normalized/` 42 GB
  - `cnn_resnet50_zm_sweep/` 33 GB
  - `cross_only_campaign_v2_chsigma/` 27 GB **(headline v2 result)**
  - `cross_only_campaign/` 27 GB (v1 — kept for provenance, HANDOFF §3.3)
  - `auto_cross_v2_chsigma/` 20 GB
- `scripts/sbi/results/final/` — **59 GB** under `paper_sbi_consolidation/`.
  Largest subdirs: `cnn_bnt_resnet_split_campaign` 12 GB,
  `cnn_bnt_noise_curriculum_campaign` 9.9 GB, `cnn_bnt_parity_campaign`
  7.1 GB, `baryonified_appendix` 6.8 GB, `nobnt_final_matrix` 6.1 GB.
- `scripts/sbi/results/diagnostics/` — 1.4 GB (SBC, TARP, l1 datavectors,
  cross-map sanity).
- `scripts/sbi/results/run_artifacts/save_params/` — 971 MB (flow / CNN
  pickle checkpoints from older runs).
- `scripts/sbi/results/wandb_runs/` — 6.4 MB (older).
- `artifacts/` — 1.3 GB total (notebook posteriors, maps, checkpoints,
  legacy wandb).
- 831 `posterior*.npy` files under `scripts/sbi/results/`,
  818 `.meta.json` and 321 `.fom.json` siblings.

### Still valid? regeneration cost

| Artifact class | Valid as-is | Regen cost if lost |
|---|---|---|
| Pre-2026-04-21 CNN posteriors | **Invalid** as scientific claims (mass-sheet leak, KB §13). | Cheap to regenerate under `--zero-mean-maps` but the demeaned baseline is already in `zero_mean_maps_parity_check/`. |
| v1 L1 cross posteriors (`cross_only_campaign/`, `auto_cross_v2_chsigma/`'s v1 sibling) | Valid for provenance only (broken noise model). | Cheap to regenerate, but kept on disk per HANDOFF §3.3 for the v1↔v2 comparison narrative. |
| v2 L1 cross posteriors (`cross_only_campaign_v2_chsigma/`, `auto_cross_v2_chsigma/`) | **Headline result.** Valid. | ~hours per seed on GPU. Real loss risk. |
| Harmonic cross cache `full_sphere_cache_grid/` (623 GB) | Valid (manifest sha256 `0a68ea89...`). | 56 min wall on 50 CPU workers; rebuild script committed. **Regenerable but disk-bound.** |
| Trained CNN compressors in `_shared_compressor/<arm>/dim_*/save_params/` | Valid (HANDOFF §3.7: Stage A artefacts that Stage B re-uses). | Hours per arm. Real loss risk if deleted mid-campaign. |
| `paper_sbi_consolidation/cnn_bnt_*` reports | Valid as historical record; FoM3 absolute numbers retracted per KB §13 / §14. | Regeneration of underlying CNN posteriors is now meaningful only with `--zero-mean-maps`, which changes the numbers. |
| SBC and TARP outputs in `results/diagnostics/` | v1 valid for v1 narrative. v2 not yet produced. | TARP rerun on v2: ~3 h on 3 GPUs per HANDOFF. |

---

## Notebooks

`notebooks/sbi/` contains 16 `.ipynb` plus one `.py`. None has had its code
formally promoted to a module — but the modern pipeline in `scripts/sbi/npe_*`
replaces what most of these did. mtimes give the best ordering signal.

| Notebook | mtime | cells (code/md/outputs) | Purpose / status |
|---|---|---|---|
| `comparison_summary.ipynb` | 2024-09 | 11/3/5 | Earliest project notebook (prior/proposal check). **Stale.** |
| `nle_vmim_gaussian.ipynb` | 2024-09 | 23/3/17 | Original Justinezgh-style VMIM Gaussian example. **Stale.** |
| `nle_vmim_nbody_baryon_ia.ipynb` | 2024-09 | 26/3/17 | VMIM N-body with baryon+IA. **Stale.** |
| `nle_mse_gaussian.ipynb` | 2025-02 | 34/3/23 | MSE compressor on Gaussian sims. **Stale.** |
| `nle_mse_nbody.ipynb`, `nle_mse_nbody_baryon_ia.ipynb`, `nle_vmim_nbody.ipynb` | 2025-05 | 23–27/3 each | MSE / VMIM single-bin pre-tomo notebooks. **Stale** (superseded by `npe_cnn_nbody_tomo.py`). |
| `nle_simple_nbody.ipynb` | 2025-05 | 31/35/24 | The notebook that `nle_simple_nbody_NPE.py` was derived from. **Stale.** |
| `nle_simple_nbody_20deg.ipynb` | 2026-02 | 35/33/25 | 20° / 160 px N-body single-bin. **Stale** but the closest single-bin reference for the FOV switch. |
| `nle_simple_nbody_NPE.ipynb` | 2026-04 | 22/27/18 | NPE refactor experiments. **Stale / exploratory**. |
| `nle_simple_nbody_tomo.ipynb`, `..._tomo_BNT.ipynb`, `..._tomo_12features.ipynb` | 2026-04 | 45–56/42–49/34–42 | Tomographic CNN exploration with BNT / 12-feature variants. **Exploratory**, parallel to the modern `npe_cnn_nbody_tomo.py`. |
| `overplot_l1_tomo_bin3_vs_cnn_tomo.ipynb` | 2026-04 | 7/1/0 | Contour overplot scratch. Clean (no outputs). |
| `overplot_l1_vmim_vs_l1_nopca.ipynb` | 2026-03 | 8/4/0 | L1-VMIM vs L1-no-PCA overplot scratch. Clean. |
| `l1_vs_CNN.ipynb` | 2026-03 | 19/4/11 | "Systematic runs 24" — manual L1 vs CNN comparison. **Untracked** in git (per `git status`). Has outputs. |
| `nle_vmim_nbody_script.py` | (script) | n/a | Plain-Python export of `nle_vmim_nbody*.ipynb`. **Stale.** |

Top-level `notebooks/`:

- `check_datasets.ipynb` (2024-09-ish, 3 978 KB with outputs) — dataset
  sanity check. **Stale.**
- `check_ps_gaussians_vs_nbody.ipynb` (5 059 KB with outputs) — power-spectrum
  comparison of Gaussian vs N-body sims. **Stale.**
- `tf_dataset_nbody.py` — a copy of the legacy TFDS builder. **Stale.**

Migration status: the *functionality* (CNN compressor training, BNT, L1
featurisation, NPE sampling) has been moved into `scripts/sbi/npe_*`. The
notebooks remain as **frozen exploratory snapshots with their outputs
committed**; per HANDOFF/CLAUDE.md notebooks are "perpetually dirty" and
should not be committed. Most of these notebooks would not run today
without code edits (TFDS names changed, dataset variants added, paths
moved).

---

## Open threads

Things where something was started and not finished, identified from
`HANDOFF.md`, `git status`, and result trees:

1. **TARP joint-coverage on the v2 (channel-aware-σ) L1 arms** —
   `HANDOFF.md` §4.3 and §5.2. v1 TARP exists; v2 does not. Runner is
   arm-agnostic, ~3 h on 3 GPUs.
2. **CNN resnet50_gn auto+cross seed-scatter rerun** — `HANDOFF.md` §4.1.
   Existing seeds disagree on posterior means; pooled FoM3 11k vs
   mean-of-seeds 19k. Open question whether to retrain.
3. **v1 vs v2 L1 cross-only contour overlay figure** — `HANDOFF.md` §5.1.
   Not produced.
4. **Two superseded memory entries** —
   `project_harmonic_cross_overturns_flatsky.md` and
   `project_cross_only_l1_loses.md` carry the v1 (broken σ) numbers and
   need a "Superseded by [[project_l1_noise_model_correction]]" header.
   Flagged in `MEMORY.md` index but bodies not rewritten.
5. **Paper / report write-up** — `HANDOFF.md` §5.5. Not started.
   `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` is the closest synthesis, but it
   predates the noise-model fix (§14.7 only) and §3, §10 still need a
   round of editing for the v2 narrative.
6. **σ_8 contour shape on v2 L1 cross-only** — `HANDOFF.md` §4.2. Visible
   degeneracy-axis mismatch vs CNN even after the v2 fix. TARP is the
   gating diagnostic.
7. **Auto-only CNN resnet50_gn cell** — `HANDOFF.md` §4.4. The 3×3 table
   currently uses stock-BN resnet50 in that cell. Asymmetric labelling
   "(stock BN)" — open whether to rerun.
8. **dim=20 expansion** — `HANDOFF.md` §4.5. Tier-2 plan never executed;
   author assumed dim=10 is decisive, but not verified.
9. **Many half-named subdirs in worktree** (per `git status` untracked):
   `scripts/sbi/baryon_bias_tomo4_study_dryrun_smoke/`,
   `..._subset/`, `nobnt_tomo_bins_crosscorr_study_dryrun/`,
   `..._l1_jaxili_bestcfg/`, `..._l1_jaxili_nopca/`,
   `l1_jaxili_tomo4_opt_sweep/`. These exist alongside the canonical study
   directories. Unclear whether they were left as scratch from earlier
   sessions or are intentional duplicates. Worth disambiguating before
   reorganisation.
10. **Legacy "bnt_tomo_study" worktree** at `.worktrees/bnt_tomo_study/` —
    standalone working tree on the `bnt_tomo_study` branch (HEAD
    `0184d77`). Per CLAUDE.md "do not delete". Its result tree under
    `scripts/sbi/baryon_bias_tomo4_study/` (in the main repo) actually
    records `repo_root = .worktrees/bnt_tomo_study` in its `manifest.json`
    — i.e. results were produced from the worktree and the artefacts now
    live in two places.
11. **`scripts/sbi/bnt_tomo4_study/posterior_summary.json.local_backup`**
    (per `git status`) — a local-only backup of a JSON the user modified.
    Unclear what diverged; safe to leave for now.

---

## Risks (what would be painful to lose)

1. **Headline v2 results** at
   `scripts/sbi/results/exploratory/cross_only_campaign_v2_chsigma/` and
   `auto_cross_v2_chsigma/` (~47 GB combined). These are the current
   scientific answer. Per-seed CNN compressor and NDE training is hours of
   GPU each; full v2 campaign was the latest multi-day push.
2. **The 623 GB harmonic cross-map cache**
   (`scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid/`).
   Regenerable in ~1 h on 50 CPU workers from
   `build_full_sphere_cross_cache.py`, but it's the sole source of truth
   for both v2 L1 and v2 CNN cross runs — the manifest sha256 is stamped
   into every consumer (`l1_cache_meta.npz`, CNN run logs). Deleting
   without a backup means everything downstream needs to be re-cached
   *and* re-validated against the new sha.
3. **Stage A "shared compressor" checkpoints** under
   `<campaign>/_shared_compressor/<arm>/dim_<N>/save_params/`. The Phase 2
   CNN NDE-only path resolves these via
   `_find_latest_compressor_checkpoint`. HANDOFF §7 explicitly warns
   against deletion. `--skip-existing` is **not** safe with partial
   checkpoints — manual cleanup of `_shared_compressor/<arm>/dim_<N>/save_params/`
   before relaunching is required for a corrupted Stage A.
4. **The 12 GB ResNet split campaign** and **9.9 GB noise-curriculum
   campaign** under `paper_sbi_consolidation/`. The numbers in
   `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` §6 / §7 / §9 cite these
   directly; while the FoM3 absolute values are partially retracted
   (KB §13), the *ratios* (BNT/no-BNT, parity tradeoffs) are how the
   "no invariance trick generalised" claim is anchored.
5. **TARP v1 outputs at `results/diagnostics/tarp_harm_cross/`** —
   17 (arm, seed) cells × N=200 × M=2000 + 200-bootstrap. The
   `project_harmonic_cross_overturns_flatsky.md` memory and KB §14's
   coverage claim depend on it. Regenerating these is non-trivial.
6. **Reports and `*.md` synthesis documents** in repo root
   (`HANDOFF.md`, `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`,
   `HARMONIC_L1_VS_CNN_INVESTIGATION_*`, `BNT_*`, `L1_*`). These are the
   only narrative that maps result paths to scientific claims. Losing
   them would force re-deriving the v1↔v2 distinction from git history
   alone, which is feasible but expensive.
7. **`/tmp/cross_only*_loop_check.py`** (HANDOFF §6) — ephemeral
   autonomous-loop monitors. Marked as throwaway by HANDOFF §9. Not a
   risk; mentioned only because they're referenced in the handoff.
8. **`setup.py` shipping a `packages=["learn2map"]` claim against a
   directory that's a venv.** Low risk but anyone running `pip install -e .`
   in a fresh checkout may get a confusing failure mode. Unverified.

---

## Things I could not confirm

- Whether the older `bnt-parity-techniques`-era CNN posteriors (e.g.
  inside `paper_sbi_consolidation/cnn_bnt_parity_campaign/` confirmation
  arms) have been re-validated under `--zero-mean-maps`. KB §13.5 lists
  "re-launch the strongest parity sweeps under `--zero-mean-maps`" as a
  minimal next action but I did not see a result directory matching
  that. Likely **not done yet**.
- Whether the untracked `*.local_backup` and `_smoke`, `_dryrun`,
  `_bestcfg`, `_nopca` directories under `scripts/sbi/` are duplicates of
  the canonical study dirs or refer to different config families. The
  duplication is real (e.g. `nobnt_tomo_bins_crosscorr_study/` vs
  `nobnt_tomo_bins_crosscorr_study_l1_jaxili_bestcfg/`) and worth a
  human disambiguation before reorganisation.
- Whether `pip install -e .` currently works given the `learn2map`
  package directive against a venv directory. Did not test.
- Whether any of `notebooks/sbi/*.ipynb` notebooks still run end-to-end
  on the current TFDS naming and dataset variants. Almost certainly not
  for the pre-tomo notebooks, but unverified for the 2026-04 tomo ones.

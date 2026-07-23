# TITAN TRANSFER MANIFEST — what to save before the data becomes unreachable

Written 2026-07-22 (laptop session), for Andreas's titan→new-cluster transfer.
Context: titan is broken; data is readable but not usable in place, and not
everything can be moved. This ranks what the *paper* (L1-vs-CNN revision) and
its queued follow-ups still need, most-important first. Paths are relative to
the titan repo root `/mnt/home/tersenov/software/cnn_sbi/` unless noted.
Everything already in git (branch `collect-useful-uncommitted-2026-07-20`) is
safe and NOT listed.

**Rule of thumb: Tier 0 is megabytes — take all of it blindly. Tier 1 is the
science payload (GB-scale). Tier 2 is insurance. Tier 3 only if room.**

---

## Tier 0 — irreplaceable small files (MB total; take ALL, blanket sweep)

One rsync that grabs every document, script, and metadata file in the results
tree while excluding the heavy binaries:

    rsync -av --prune-empty-dirs \
      --include='*/' \
      --include='*.md' --include='*.json' --include='*.csv' \
      --include='*.txt' --include='*.py' --include='*.sh' --include='*.log' \
      --exclude='*' \
      scripts/sbi/results/  <DEST>/results_docs/

Known git-orphans this must catch (verified missing from the clone):

- `scripts/sbi/results/exploratory/flatsky_cross_2026_06/analytical_nde_match/RESULT_TWOPT_SPLIT_ENSEMBLE.md`
  — the de-inflated ΔNG numbers (124/260) are quoted from it second-hand;
  the primary doc exists only on titan.
- `RESULT_OPERATOR_GATE.md` (same campaign tree) — records the *content* of
  the "PASS with caveat" gate verdict for the conv/both operator-probe arms.
  The paper's `fig:ablation` caption says "calibrated with caveat" on its
  authority; rewording that caption (and adding the App-A sentence a referee
  will ask for) is blocked until this file is read.
- `…/flatsky_cross_2026_06/conv_map_secure/` — the adversarial-verification
  outputs behind `CONV_MAP_SECURE_RESULT.md`: `secure_conv_mean.py`,
  `slope_seed_test.py`, `quantify_t4.py`, **`t4_results.npz`** (small npz —
  include it explicitly, the rsync above skips npz).
- every `median_summary.json`, `verdict.json`, `tarp_summary.json`,
  `sbc_summary.json`, `per_seed.json` across all campaigns (tiny, priceless
  provenance).

## Tier 1 — the science payload (blocks queued paper items)

### 1a. Per-seed compressor caches + checkpoints (G-03 error-bar sweep)
The sweep (`PLAN_FOM_ERRORBARS_SWEEP.md`) needs, under
`…/flatsky_cross_2026_06/analytical_nde_match/` (compressed 10-D summary
caches are ~10–20 MB each; checkpoints ~50 MB):

- `l1none_vmim_s41/`, `ens_nobnt_auto_s42/`, `ens_nobnt_auto_s43/`
- `l1product_vmim_s41/`, `l1product_vmim_s42/`, `l1product_vmim_s43/`
- the joint-ℓ₁ per-seed arms (compressor seeds 41/42/43, noBNT **and** BNT)
  used by `RESULT_JOINTL1_SEEDCHECK.md` / `RESULT_JOINTL1_ENSEMBLE.md` /
  `RESULT_JOINTL1_BNT_ENSEMBLE.md` — whatever directories hold their
  compressed caches and gate dumps.
- the **canonical CNN** (resnet18 + RealNVP recipe): compressor
  checkpoint(s) and compressed caches, any seed that exists under the final
  recipe (see `HANDOFF_CANONICAL_REFRESH.md`,
  `cnn_phase/ESTIMATOR_OPTIMIZATION_RECORD.md` for which run is canonical).
  If canonical s42/s43 compressors were ever trained, they are here — search
  before assuming they need retraining.
- NDE / flow checkpoints for the above if present (small; cheap to retrain
  from caches, but free to carry).

### 1b. The fiducial observation caches (both frames) — the evaluation bedrock
`results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg/`
— `nobnt/obs/` and the BNT counterpart (200 files × 180 patches = 36 000
patch sets; order a few–tens of GB per frame). Every re-evaluation, the
median block bootstrap, the per-seed sweep, and the proposed transition-band
experiment run on these. **If only one big thing fits, it is this.**

### 1c. twopt-split arms (pooled-gate prerequisite)
`…/analytical_nde_match/twopt_split/<arm>/` for `auto_cov`, `conv_cov`,
`product_cov` (+ any `*_cov` siblings): compressed caches, per-seed dirs, and
`gate/` dumps. Needed to run the pooled TARP/SBC gate — the precondition for
ever quoting ΔNG(conv)=124 / ΔNG(product)=260 in the paper.

### 1d. Final paper-figure inputs (Phase-D figure polish without recompute)
Wherever the 2026-06-27 figure session wrote the publication set (a
`paper_plots/` or similar flat dir at repo root or under
`flatsky_cross_2026_06`): the **generation scripts and their input arrays**
(.npz/.json), not just the PDFs — posterior samples per arm at the mean
observation (corner + BNT contour figures), per-obs marginal/FoM arrays (the
violin figure), TARP/SBC curves with bootstrap bands (appendix figures), and
the example patch maps behind the data figure. Without these, any figure
restyling means re-running inference.

## Tier 2 — insurance (take if the transfer allows)

- GATE-C TARP-DRP dumps + SBC rank files for the final arms (noBNT and BNT):
  enable pooled re-gates and appendix-figure regeneration.
  (`…/gate_c/tarp_drp/dumps/`, `…/gate_c/sbc/` under the final campaigns.)
- `population_sweep_full/` per-observation FoM/σ arrays for the final arms —
  the block bootstrap runs on these directly, no inference needed.
- Frozen noise σ tables and any RNG/seed state files of the final recipe
  (`freeze_flatsky_cross_noise.py` outputs) — recipe exactness if anything
  is ever rebuilt.

## Tier 3 — only if there is room

- TFDS/TFRecord **training** caches for the final 10° campaign (~tens of GB):
  needed only if CNN compressor seeds 42/43 must be retrained on the new
  cluster. Fallback without them: rebuild from public cosmoGRID + the frozen
  noise recipe (possible, but compute + bit-exactness risk).
- The `/nas` 20° archive (Appendix-D era): the paper no longer uses it.

## Search while you still have access (unknown paths — descriptions)

- **Noiseless-maps BNT run** (paper §6 sentence, triage flag 7): a run where
  noise-free maps went through the BNT and no-BNT pipelines and the contours
  essentially coincide, for the summaries of the final campaign. NOT
  `final/paper_sbi_consolidation/cnn_noiseless_vs_noisy/` (that is the old
  tomo4 CNN-only study, already in git). Search:
  `find scripts/sbi/results -iname '*noiseless*' -o -iname '*nonoise*' -o -iname '*noise0*'`
  and grep run logs for `--no-noise`/`noiseless` flags around the June BNT
  campaigns. If found, take the whole directory.
- **Canonical-recipe CNN seed-42/43 compressors** (see 1a) — search
  checkpoint dirs for the canonical tag before concluding they don't exist.
- Do NOT hunt for the optuna study on titan — it ran on a different cluster
  (confirmed absent from all titan material).

## Explicitly NOT needed (don't spend transfer budget)

- Old-era campaign bulk under `results/final/paper_sbi_consolidation/`
  (tomo4, noise-curriculum, resnet-split raw outputs) — their reports are in
  git; the runs are superseded.
- `zero_mean_maps_parity_check/` overlays, superseded 20° flat-sky caches.
- Raw cosmoGRID full-sky maps — public, re-downloadable.
- Anything already on the `collect-useful-uncommitted-2026-07-20` branch.

## Post-transfer verification (5 minutes)

- [ ] `RESULT_TWOPT_SPLIT_ENSEMBLE.md` present and readable.
- [ ] `RESULT_OPERATOR_GATE.md` present and readable (unblocks the
      `fig:ablation` caveat caption).
- [ ] `conv_map_secure/t4_results.npz` present.
- [ ] `l1product_vmim_s41..43`, `l1none_vmim_s41` + `ens_nobnt_auto_s42/43`
      caches load.
- [ ] joint-ℓ₁ seed arms (noBNT + BNT) present.
- [ ] canonical CNN checkpoint + cache present.
- [ ] fiducial obs caches: 200 files in `nobnt/obs/` (and BNT frame).
- [ ] twopt_split `*_cov` arms with `gate/` dumps.
- [ ] figure input arrays + scripts for the publication figure set.
- [ ] docs sweep: `find results_docs -name '*.md' | wc -l` looks like
      hundreds, and spot-check one `median_summary.json`.

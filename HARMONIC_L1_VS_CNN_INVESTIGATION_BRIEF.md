# Investigation brief: harmonic-L1 surpasses CNN-VMIM in no-BNT — why?

**Author context:** drafted 2026-05-04 to hand off to a follow-up LLM agent.
The reader is expected to be familiar with the codebase via `CLAUDE.md`,
`PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` (especially §13 mass-sheet patch and
§14 harmonic cross-maps), and `skills/sbi/SKILL.md` (campaign protocol).
Nothing in this brief presupposes prior conversation history with you.

---

## 1. The unexpected finding

After building cross-maps in harmonic space on the full sphere (Zürcher 2022
construction, `a_ℓm^(i) · a_ℓm^(j)`, then ISHT, then gnomonic-project to
the same 48 patches the L1 pipeline already uses), the wavelet-L1 statistic
on the 4-auto + 6-cross channels delivers, pooled over seeds 41/42/43:

| arm | Ω_m mean ± std | σ_8 mean ± std | w_0 mean ± std | FoM3 |
|---|---|---|---|---:|
| **harm cross L1, no-BNT**     | 0.256 ± 0.025 | 0.844 ± 0.041 | -1.042 ± 0.129 | **56602** |
| best CNN-VMIM (demeaned), no-BNT | 0.274 ± 0.034 | 0.814 ± 0.048 | -1.125 ± 0.165 | 12421 |
| best CNN-VMIM (demeaned), BNT    | 0.268 ± 0.031 | 0.822 ± 0.044 | -1.094 ± 0.170 | **12754** |
| harm cross L1, BNT             | 0.300 ± 0.042 | 0.808 ± 0.071 | -1.164 ± 0.214 |  4549 |

(Truth = `(0.26, 0.84, -1.0)`. CNN reference is `run_b_advanced_plain` =
`advanced_arch64_dense256_nostd_long`, plain CNN, cdim=10, 120k compressor
steps, cf. PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md §13 and the parity-check
protocol summary at
`scripts/sbi/results/exploratory/zero_mean_maps_parity_check/SUMMARY.md`.)

So in **no-BNT**, harmonic-cross L1 has FoM3 ≈ 4.6× the best CNN; in **BNT**,
the best CNN still leads ≈ 2.8×. This is unexpected because the CNN-VMIM
target should be the asymptotically optimal compressor — and in particular,
a CNN that sees all four tomographic bins as channels should *automatically*
pick up cross-bin information without anyone constructing explicit cross
maps. There is no theoretical reason a handcrafted L1 datavector should
beat a properly-trained CNN-VMIM on the same data.

---

## 2. Hypotheses and proposed tests

### Hypothesis A — the CNN compressor is under-trained / lossy

If true, the CNN posterior is over-broad relative to the true Bayesian
posterior because the compressor throws away information the L1 statistic
happens to retain. The 4.6× FoM3 gap would be a measurement of *how lossy*
the current best CNN is, not a fundamental property of the data.

**Tests for hypothesis A** (do these first; they are cheap-to-medium cost):

1. **CNN training-curve audit.** Read `compressor/{nobnt,bnt}/cnn_vmim/nbody/loss_train_cnn.npy` and
   `loss_val_cnn.npy` under `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/run_b_advanced_plain/`.
   Plot train+val loss and confirm the val loss has plateaued. If it is
   still declining at step 120k, longer training is the obvious first move.
2. **Wider compressor + longer training.** Re-launch the same arch with
   `compressor_steps=240000` (2× the current best) and `compressor_dense_width=512`
   (2× wider). If FoM3 improves monotonically with more training/width,
   we are in the under-training regime.
3. **Feed cross-maps to the CNN.** This is the user's explicit suggestion.
   Concatenate the same 6 harmonic cross-channels onto the 4 auto channels
   and feed all 10 to the CNN as input channels. If the CNN can extract
   cross-information from raw maps in principle, supplying the cross maps
   as explicit channels should at least match the implicit-extraction
   path; if it instead jumps to harmonic-L1 levels, that confirms the CNN
   was failing to extract this information by itself even though the
   information was present in the auto channels.
4. **Larger compressor capacity.** Try resnet34/resnet50 with cdim ∈ {12,
   16, 24}. The earlier resnet50 sweep
   (`scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_resnet_split_campaign/resnet_extended_tuning_v2/`)
   was pre-`--zero-mean-maps` and is now suspect; a fresh resnet50
   demeaned sweep is overdue regardless.
5. **VMIM target stability check.** The VMIM mutual-information bound
   trained jointly with the compressor can saturate well below the actual
   MI if the auxiliary network is under-capacity. Increase the VMIM-MLP
   hidden width and verify the compressor extrema move.

### Hypothesis B — the harmonic-L1 posterior is wrong

If true, the harmonic-L1 chain is overconfident — it reports a much tighter
contour than is actually justified by the data, and the "win" disappears
under coverage testing. Possible mechanisms:

- **Train/val leakage in the harmonic cache.** The cache builder ran on
  CosmoGridV1 cosmologies 1–899 (train) / 900–1299 (val) / fiducial (obs),
  matching the flat-sky TFDS split. Any leak of fiducial maps into the
  train shard would shrink the apparent posterior at fiducial.
- **Information leak in the patch demeaning.** Each patch is demeaned
  per-channel after the gnomonic projection; for cross-channels which are
  products of two κ fields, the per-patch mean of `κ_i · κ_j` carries a
  cosmology-dependent signal that *would* be available to a real survey
  only via a separately measured 2-point auto-correlation. If we remove it
  (we do — `--zero-mean-maps`-equivalent) we are not over-using
  information. Worth re-checking that this is the case in the code, not
  just in the comment.
- **Apodization / patch-boundary effects.** Patches are 20° at NSIDE=512;
  cross-products of the alm field have power at all scales, so the
  patch-boundary discontinuity in the cross channels may introduce
  power-spectrum artifacts that the L1 statistic happens to encode in a
  cosmology-discriminating way that does not actually correspond to true
  signal.
- **Flow over-fit / collapse.** The conditional RealNVP flow trained on
  a 10-channel × 200-bin = 2000-d L1 datavector may be overfitting to
  spurious structure unique to the harmonic cache.

**Tests for hypothesis B** (do these *before* publishing the harmonic-L1
result):

1. **Coverage / SBC test (most important).** For each of N ∈ {200, 1000}
   simulated observations drawn from random `cosmo_grid_*` cosmologies
   (i.e. NOT fiducial), sample the trained NPE posterior and compute the
   rank statistic of the true parameters. A correct posterior gives uniform
   ranks; an overconfident one gives a U-shape. Run both for harmonic-L1
   no-BNT and best-CNN no-BNT; the L1 win is real only if its coverage is
   at least as good as the CNN's.
2. **Second-cosmology truth check.** Re-load the trained NPE flow and
   evaluate it on observed maps drawn from a `cosmo_delta_*_p` /
   `cosmo_delta_*_m` corner (e.g. `cosmo_delta_s8_p`, `cosmo_delta_Om_m`)
   instead of fiducial. The posterior peak should track the input
   cosmology shift. If the harmonic-L1 chain stays glued to fiducial, that
   is direct evidence of over-confidence or pipeline bias; if it tracks
   the shift, that is direct evidence of real information.
3. **Cache-integrity audit.** Re-run
   `scripts/sbi/diagnose_full_sphere_cross_maps.py` on the production
   cache and confirm the cosmology-dependent shifts in the L1
   datavectors look as expected (the diagnose script was sized for the
   smoke cache; rerun it on the grid cache).
4. **Held-out cosmology sweep.** Build a tiny secondary cache with 5
   random cosmologies *removed* from train but *included* in val, retrain
   one seed, and check whether the posterior over the held-out
   cosmologies' truths is well-calibrated (this is the most thorough
   form of test 1).
5. **L1 datavector sanity at fiducial.** For one fiducial realization in
   the cache, plot the 10-channel × 200-bin L1 datavector and overlay it
   on the L1 datavectors from a few `cosmo_delta_*_p/m` realizations.
   The cosmology-induced shifts should be smooth and in the right
   direction (e.g. higher σ_8 → more high-SNR pixels → datavector tilts
   to the right). The figure should already be produced by
   `diagnose_full_sphere_cross_maps.py` — verify it makes physical sense.

### Hypothesis C — both, partially

The most likely real outcome is a mixture: the CNN is under-trained AND
the L1 result is somewhat over-confident, but the L1 still genuinely
contains information the CNN is missing in no-BNT. The coverage tests
above directly quantify this.

### Other mechanisms worth considering

- **L1 datavector dimensionality.** With 10 channels × 5 scales × 40 SNR
  bins = 2000-d raw summary, the flow is consuming a fairly high-dim
  input. If a few of those dimensions concentrate cosmology information
  that the CNN-VMIM bottleneck (cdim=10) cannot represent, that is itself
  an information argument for L1 — but it should be testable by raising
  CNN cdim to 20 or 50 (combined with longer training).
- **Wavelet-L1 implicit prior.** L1 is not a sufficient statistic; it
  encodes *specific* non-Gaussian features (peaks/voids). If the
  cosmology likelihood happens to be very concentrated in those features
  for this fiducial truth, L1 can transiently look "better" than a
  fully-general CNN even if the CNN is asymptotically better. A multi-
  cosmology calibration would expose this (test B.2 above).

---

## 3. Suggested next-step plan (ordered)

These are roughly ordered by cost. Do A.1–B.5 before writing anything in
the paper that uses the no-BNT harmonic-L1 number.

| # | Test | Hyp | Cost | Where it goes |
|---|---|---|---|---|
| 1 | CNN train-curve audit (visual) | A | 30 min | `scripts/sbi/results/exploratory/cnn_lossiness_check/loss_curves.{png,md}` |
| 2 | Cache-integrity diagnostics on grid cache | B | 1 h GPU | `scripts/sbi/results/diagnostics/full_sphere_cross_maps/grid/` |
| 3 | Datavector cosmology-shift sanity figure | B | 1 h GPU | same as #2 |
| 4 | Harmonic-L1 second-cosmology truth check (5 cosmos) | B | 4 h GPU | `scripts/sbi/results/exploratory/cross_maps_campaign/harm_l1_truthcheck/` |
| 5 | CNN longer training + wider dense (240k, dense 512) | A | 1.5 d GPU | `scripts/sbi/results/exploratory/cnn_extended_train_zm/` |
| 6 | CNN with explicit harmonic cross-channels (10-ch input) | A | 1.5 d GPU | `scripts/sbi/results/exploratory/cnn_with_harm_cross/` |
| 7 | SBC coverage test, harmonic-L1 no-BNT, N=1000 obs | B | 4 h GPU | `scripts/sbi/results/diagnostics/sbc_harm_l1_nobnt/` |
| 8 | SBC coverage test, best CNN no-BNT, N=1000 obs | A/B | 4 h GPU | `scripts/sbi/results/diagnostics/sbc_cnn_nobnt/` |
| 9 | resnet50 demeaned sweep with cdim ∈ {10, 20, 50} | A | 3 d GPU | `scripts/sbi/results/exploratory/cnn_resnet50_zm_sweep/` |

---

## 4. File / run / cache locations

### Harmonic L1 campaign (the new winning arm)

- Cache (read-only):
  - `scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid/{bnt,nobnt}/{train,val,obs}/*.npz` (623 GB)
  - manifest: `.../full_sphere_cache_grid/manifest.json` (sha256 `0a68ea89669da18f...`)
  - obs split for `cosmo_fiducial` is a symlink into `full_sphere_cache_fiducial/`.
- Cache builder source: `scripts/sbi/build_full_sphere_cross_cache.py`
- L1 + NPE entrypoint: `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py` with
  `--full-sphere-cross-cache <CACHE_PATH>`.
- Single-arm runner: `scripts/sbi/results/exploratory/cross_maps_campaign/run_harmonic_arm.sh <gpu> <bnt|nobnt> <seed>`
- 6-arm campaign launcher: `scripts/sbi/results/exploratory/cross_maps_campaign/run_harmonic_campaign.sh`
- Posteriors:
  `scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_harm_cross_{bnt,nobnt}/posteriors/l1cross_tomo4_20deg160mp_harm_{bnt,nobnt}_p1_s{41,42,43}.npy`
- Per-seed L1 caches (cheap to delete and rebuild):
  `scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_harm_cross_{bnt,nobnt}/l1_cache_seed{41,42,43}/`
- Diagnostics script (currently smoke-sized; needs a re-run on grid):
  `scripts/sbi/diagnose_full_sphere_cross_maps.py`

### Best CNN-VMIM reference

- Source / runner: `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/run_zero_mean_parity.py`
  (config block `advanced_arch64_dense256_nostd_long`, plain CNN, cdim=10,
  120k compressor steps, 10k flow steps, `--zero-mean-maps`).
- Underlying entrypoint: `scripts/sbi/npe_cnn_nbody_tomo.py`
  (or `npe_cnn_jaxili_nbody_tomo.py` if jaxili NPE is preferred).
- Posteriors: `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/run_b_advanced_plain/posteriors/cnn_tomo4_20deg160_{bnt,nobnt}_advanced_arch64_dense256_nostd_long_zm_s{41..45}.npy`
- Compressor checkpoints + train/val loss arrays:
  `.../run_b_advanced_plain/compressor/{bnt,nobnt}/cnn_vmim/nbody/`
- Pre-existing comparison: `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/SUMMARY.md` and the metrics CSV/JSON beside it.

### Comparison artifacts (already produced)

- 3-arm overlays: `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/overlay_harm_vs_flat_vs_auto_{bnt,nobnt}.pdf`
- 2-arm L1-vs-CNN overlays: `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/overlay_harm_cross_vs_cnn_{bnt,nobnt}.pdf`
- Regime contrast: `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/overlay_harm_cross_bnt_vs_nobnt.pdf`
- Pooled summary: `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/summary.{md,json}`
- Headline writeup: `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/harmonic_results.md`
- Source scripts: `scripts/sbi/results/exploratory/cross_maps_campaign/{summarize_cross,overlay_harm_vs_flat_vs_auto,overlay_harm_cross_vs_cnn}.py`

### Knowledge base entries

- §13 — mass-sheet-degeneracy correction (`--zero-mean-maps`).
- §14 — harmonic cross-maps result (this campaign). Both in `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`.

---

## 5. Important things NOT to do

1. Do not claim the harmonic-L1 no-BNT FoM3 = 56602 number in the paper
   until at least the second-cosmology and SBC coverage tests pass.
2. Do not delete the per-seed L1 caches (`jaxili_harm_cross_*/l1_cache_seed*`)
   without first verifying the cache `.npz` files in `full_sphere_cache_grid/`
   are intact — those are the recoverable artifact; the per-seed L1 caches
   are derived but ~7 GB each and slow to rebuild.
3. Do not rebuild `full_sphere_cache_grid` from scratch unless you have
   strong evidence the existing cache is wrong; it cost 56 min wall on
   50 CPU workers and 623 GB of disk.
4. Do not touch the flat-sky cross-maps pipeline expecting it to also
   improve — the conclusion is that the FFT-on-patches construction is
   irrecoverably lossy, not that the flat-sky implementation has a bug
   to fix.
5. Do not amend or force-push the existing commits on `l1-cross-maps`;
   add new commits. The current head as of this brief is `f512bde`.

---

## 6. Suggested first action for the agent picking this up

Start with task #1 (CNN training-curve audit). It is 30 minutes of work
and immediately rules in or out the simplest version of hypothesis A.
If the CNN val loss is still declining at 120k steps, that alone makes
hypothesis A the leading candidate and reorders the rest of the plan.

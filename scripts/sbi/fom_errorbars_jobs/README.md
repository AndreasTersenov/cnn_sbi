# SLURM jobs — FoM3 error-bar sweep (Jean-Zay, 2026-07-28)

The jobs exactly as run to re-measure all eight Table-1 rows after the Titan RAID0
failure. Results, definitions and provenance:
`AndreasTersenov/L1_vs_CNN_Tomographic_SBI` -> `RESULT_FOM_ERRORBARS.md`.

| job | what it does |
|---|---|
| `build_joint_arm.slurm <nobnt\|bnt>` | raw joint-l1 datavector (`--stat jointl1 --k 10 --adaptive-ranges`), 3000-D |
| `vmim_joint.slurm <nobnt\|bnt>` | 3 VMIM compressor seeds on that raw cache |
| `sw_joint_chain.slurm <nobnt\|bnt>` | 3 single sweeps + the 3-pooled ensemble sweep, chained |
| `cnn_fidsum.slurm [arm...]` | CNN observed summaries, each with its G1 gate (max\|delta\| <= 9e-4) |
| `sw_cnn.slurm <label> <preproc> <clip> <arm...>` | CNN population sweep |
| `extra_seeds.slurm <jointl1_bnt\|auto_bnt>` | seeds 44/45/46 for the six-seed check |

Environment notes that cost real time:

* `build_flatsky_joint_arm.py` computes in torch but loads through tf.data and never
  sets TF memory growth. Without `TF_FORCE_GPU_ALLOW_GROWTH=true` TensorFlow takes the
  whole A100 and torch OOMs with <1 GB of its own allocations.
* Patch centres must always come from `centers_PAPER_180.npy`. Regenerating them is not
  reproducible: `_build_non_overlapping_centers` sorts with an unstable `np.argsort` and
  HEALPix rings are tie blocks, so numpy 1.24.4 and 1.26.4 give tilings sharing only
  2 of 180 centres.
* Compressed-arm sweeps use `--preproc-transform none --clip-value 0
  --min-feature-variance 1e-12` for BOTH l1 and CNN arms. Using `zscore` on the CNN arms
  shifts them by ~3%.

## build_fiducial_summaries_cnn_flatlocal.py

The original `scripts/sbi/build_fiducial_summaries_cnn.py` was destroyed locally by the
RAID0 failure (12 695 B of NULs) and was reconstructed on Jean-Zay before we realised the
authentic file was still on GitHub. The original is left untouched; this flat_local-only
variant is what actually produced the 2026-07-28 CNN summaries, kept here for exact
reproducibility. Differences: it reads the frozen per-channel RMS from
`cache/cnn_cache_meta.npz['info_channel_scale']` instead of recomputing it, requires
`max|delta| <= 9e-4` rather than `allclose(rtol=1e-3, atol=1e-3)`, and verifies the
compressor checkpoint SHA-256 by default. Cross-checked against the original: functionally
equivalent on the flat_local route, slightly stricter.

---

## Complete job inventory

Every SLURM job run during the 2026-07-28 recovery, so the numbers can be traced end to
end. Submit order was: `setup/` -> compressors -> sweeps -> tables.

### Production — these produced the quoted numbers

| job | arm(s) |
|---|---|
| `cnn_nobnt.slurm`, `cnn_bnt.slurm` | 3 ResNet-18 VMIM compressors per frame (seeds 41/42/43) |
| `vmim_none_{nobnt,bnt}.slurm` | l1 auto-only compressors |
| `vmim_product_{nobnt,bnt}.slurm` | l1 +product compressors |
| `vmim_joint.slurm` | joint l1 compressors |
| `build_joint_arm.slurm` | raw joint-l1 datavector (3000-D) |
| `train_l1both_build.slurm` | raw l1 datavector caches |
| `train_l1product_s41.slurm` | the validation gate: l1+product no-BNT s41 vs the published 3045 |
| `cnn_fidsum.slurm` | CNN observed summaries + G1 gate |
| `sw_auto_*`, `sw_product_*`, `sw_joint_chain.slurm`, `sw_cnn.slurm` | the 9000-mock population sweeps |
| `extra_seeds.slurm` | seeds 44/45/46 for the six-seed check (RESULT §7) |

### setup/ — environment and one-off reconstruction

The obs cache, the no-BNT frozen sigma table and the patch geometry all had to be rebuilt
before any arm could run. `{find,fit,measure,prove,confirm}_centers.slurm` are the
investigation that recovered and then verified the 180 patch centres two independent
ways; its output is pinned in `inputs/centers_PAPER_180.npy` and must not be regenerated.

### superseded/ — kept for the record, DO NOT USE

`joint l1` was initially and wrongly mapped onto the ordinary l1 driver's
`--cross-op both` (conv+product channels). It reproduced faithfully — and is a different
statistic. no-BNT looked plausible (3130 vs 3371) and hid the error; BNT exposed it
(703.9 vs 2424). The real statistic is the 2-D pairwise l1-weighted histogram in
`flatsky_joint_stats.py` (`stat="jointl1"`), run via `build_joint_arm.slurm`.
Everything in this directory belongs to that dead end, plus a timing probe and two
retries. The sweeps it produced (`sweeps/joint_l1_nobnt_*`, `sweeps/joint_bnt_*`,
`vmim/both_*`) are discarded and must not be quoted.

## Patched pipeline scripts

Three scripts in `scripts/sbi/` had paths hardcoded to the dead Titan machine and were
made env-overridable so they run anywhere. All three changes are additive — nothing was
removed but the hardcoded paths:

| script | change |
|---|---|
| `build_full_sphere_cross_cache.py` | `CG_PARENT` / `CG_GRID_DIRNAME` env; new `--centers-npy` to load pinned centres instead of regenerating them |
| `freeze_flatsky_cross_noise.py` | `FLATSKY_FID_CACHE` / `FLATSKY_FID_H5` / `FLATSKY_OUT_DIR` env |
| `analytical_nde_match/ensemble_eval.py` | env-overridable paths; persists per-observation mean/cov/arm_mean/arm_cov and adds `--save-samples` |
| `build_flatsky_joint_arm.py` | `JOINT_*` env overrides (see the previous commit) |

`RECOVERY_STATE.md` is the working state document from the recovery session: verified
constants, what was destroyed vs rebuilt, and the gotchas that cost real time.

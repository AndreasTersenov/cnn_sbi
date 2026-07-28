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

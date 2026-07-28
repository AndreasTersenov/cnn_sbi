# Reproducibility inputs — FoM3 error-bar sweep (2026-07-28)

## centers_PAPER_180.npy  — DO NOT REGENERATE

The 180 non-overlapping 10-degree patch centres the paper's TFDS was built with.
`_build_non_overlapping_centers` selects them with `np.argsort`, which is quicksort and
therefore UNSTABLE, and every HEALPix ring is a block of ties. numpy 1.24.4 and 1.26.4
given identical parameters produce tilings that share only **2 of the 180 centres**.
The geometry is therefore not reproducible from the parameters alone; it must be pinned.

Always pass `--centers-npy centers_PAPER_180.npy`. These centres were confirmed 22/22 by
bit-exact reprojection against the cross TFDS, and the generating parameters are recorded
in `scripts/sbi/run_10deg_build.sh` (n-centers 180, center-nside 64, min-separation-deg
14.2, max-abs-lat 75). `centers_PAPER_180.json` is the same data in text form;
`measured_centers.json` is the independent measurement used to confirm them.

## env_freeze.txt / constraints.txt

The Python environment the 2026-07-28 numbers were produced in (python 3.11 venv on
Jean-Zay). Three pins are load-bearing:

* `tensorflow-metadata==1.16.1` with `protobuf==5.29.3` — newer tensorflow-metadata ships
  protobuf gencode 6.31.1 and breaks tfds under TF 2.18.
* `wl_stats_torch` from git (commit 77eafc84), NOT PyPI — the published v0.1.0 predates
  `subtract_coarse_mean` and the wavelet transform silently takes a different code path.
* `sbi_lens` needs a 2-line patch: `tfp.experimental.substrates.jax` no longer exists;
  use `from tensorflow_probability.substrates import jax as tfp`.

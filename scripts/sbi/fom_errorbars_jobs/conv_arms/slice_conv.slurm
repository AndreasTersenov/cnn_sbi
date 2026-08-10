#!/bin/bash
#SBATCH --job-name=slice_conv
#SBATCH --account=nzu@cpu --partition=cpu_p1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/slice_conv_%j.out
#
# Column-slice the l1+conv arm (10 ch = 4 autos + 6 conv) out of the existing 16-channel
# 'both' cache, for BOTH frames. No TFDS pass, no GPU -- the conv channels are already in
# 'both', which the driver builds as autos(4) -> conv(6) -> product(6).
#
# Same mechanism slice_raw.slurm used to make the product and none caches; the column
# selection comes from flatsky_cross_l1.op_feature_columns so it cannot drift from the
# channel order build_channels_* actually emits.
set -uo pipefail
module purge; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp

$N/envs/jaxili/bin/python -u - <<'PY'
import numpy as np, os, sys, shutil
sys.path.insert(0, "/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts")
import flatsky_cross_l1 as fxl
S = "/lustre/fsn1/projects/rech/nzu/ulx34io/recovery"
FEAT = 5 * 40
cols = fxl.op_feature_columns("conv", 4, FEAT)
assert len(cols) == 2000, len(cols)
print(f"conv columns: {len(cols)}  ({cols[0]}..{cols[-1]})", flush=True)

for frame, both_dir, out_dir, fid_in, fid_out in (
    ("nobnt", f"{S}/l1_datavector_cache/flat_local_both",
              f"{S}/l1_datavector_cache/flat_local_conv",
              f"{S}/fid_raw_both.npz", f"{S}/fid_raw_conv.npz"),
    ("bnt",   f"{S}/l1_datavector_cache_bnt/flat_local_both_bnt",
              f"{S}/l1_datavector_cache_bnt/flat_local_conv_bnt",
              f"{S}/fid_raw_both_bnt.npz", f"{S}/fid_raw_conv_bnt.npz")):
    print(f"\n=== {frame} ===", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    for split in ("l1_train", "l1_val"):
        d = np.load(f"{both_dir}/{split}.npz")
        x = d["x"][:, cols].astype(np.float32)
        np.savez(f"{out_dir}/{split}.npz", theta=d["theta"], x=x)
        print(f"  {split}: {d['x'].shape} -> {x.shape}", flush=True)
    r = f"{both_dir}/flat_local_ranges.npy"
    if os.path.exists(r): shutil.copy(r, f"{out_dir}/flat_local_ranges.npy")
    f = np.load(fid_in)
    Sx = f["S"][:, cols].astype(np.float32)
    np.savez(fid_out, S=Sx, perm=f["perm"], patch=f["patch"],
             **({"truth": f["truth"]} if "truth" in f.files else {}))
    print(f"  fid: {f['S'].shape} -> {Sx.shape} -> {fid_out}", flush=True)
    assert np.isfinite(Sx).all() and (Sx >= 0).all(), "bad conv features"
    # the first 800 columns must be the autos, identical to the 'none' arm
    none_cols = fxl.op_feature_columns("none", 4, FEAT)
    fn = f"{S}/fid_raw_none.npz" if frame == "nobnt" else f"{S}/fid_raw_none_bnt.npz"
    if os.path.exists(fn):
        a = np.load(fn)["S"]
        same = np.array_equal(a, Sx[:, :len(none_cols)])
        print(f"  auto block matches the '{'none' if frame=='nobnt' else 'none_bnt'}' arm: {same}")
        assert same, "auto block mismatch -- channel ordering assumption is wrong"
print("\nSLICE CONV DONE")
PY

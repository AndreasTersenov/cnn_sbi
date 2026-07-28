#!/bin/bash
#SBATCH --job-name=bnt_obs_slice
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/bnt_obs_slice_%j.out
# 1) BNT observation datavectors (--mode bnt, BNT ranges + intact _bnt sigma table)
# 2) slice the BNT raw cache + obs into product(10ch) and none(4ch)
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp TF_CPP_MIN_LOG_LEVEL=3
export FLATSKY_FID_CACHE=$S/recovery/full_sphere_cache_fiducial_10deg/nobnt/obs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BOTH=$S/recovery/l1_datavector_cache_bnt/flat_local_both_bnt
cd "$SRC"
echo "######## BNT observation datavectors ########"
srun $N/envs/jaxili/bin/python -u precompute_fiducial_both_datavectors.py \
  --both-cache $BOTH --sigma /lustre/fsn1/projects/rech/prk/ulx34io/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma_bnt.npz --mode bnt --perms 50 \
  --out $S/recovery/fiducial_both_datavectors_bnt.npz || exit 1
echo "######## slice BNT raw cache + obs ########"
$N/envs/jaxili/bin/python -u - <<'PY'
import numpy as np, os, sys, shutil
sys.path.insert(0,"/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts")
import flatsky_cross_l1 as fxl
S="/lustre/fsn1/projects/rech/nzu/ulx34io/recovery"
BOTH=f"{S}/l1_datavector_cache_bnt/flat_local_both_bnt"
d=np.load(f"{S}/fiducial_both_datavectors_bnt.npz")
np.savez(f"{S}/fid_raw_both_bnt.npz", S=d["x"], perm=d["perm"], patch=d["patch"])
print("  fid_raw_both_bnt:", d["x"].shape, flush=True)
for op in ("product","none"):
    cols=fxl.op_feature_columns(op,4,200); out=f"{S}/l1_datavector_cache_bnt/flat_local_{op}_bnt"
    os.makedirs(out,exist_ok=True)
    for sp in ("l1_train","l1_val"):
        z=np.load(f"{BOTH}/{sp}.npz")
        np.savez(f"{out}/{sp}.npz", theta=z["theta"], x=z["x"][:,cols].astype(np.float32))
        print(f"  {op}/{sp}: {z['x'].shape} -> {(z['x'].shape[0],len(cols))}",flush=True)
    shutil.copy(f"{BOTH}/flat_local_ranges.npy", f"{out}/flat_local_ranges.npy")
    np.savez(f"{S}/fid_raw_{op}_bnt.npz", S=d["x"][:,cols].astype(np.float32),
             perm=d["perm"], patch=d["patch"])
print("BNT SLICING DONE")
PY

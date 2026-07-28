#!/bin/bash
#SBATCH --job-name=slice_raw_ops
#SBATCH --account=nzu@cpu --partition=cpu_p1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/slice_raw_%j.out
# Column-slice the raw 16ch 'both' L1 cache into the product(10ch) and none(4ch) caches,
# and the matching raw fiducial datavectors. Cheap: no TFDS pass, no GPU.
set -uo pipefail
module purge; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp
$N/envs/jaxili/bin/python -u - <<'PY'
import numpy as np, os, sys, shutil
sys.path.insert(0,"/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts")
import flatsky_cross_l1 as fxl
S="/lustre/fsn1/projects/rech/nzu/ulx34io/recovery"
BOTH=f"{S}/l1_datavector_cache/flat_local_both"
FEAT=5*40
for op in ("product","none"):
    cols=fxl.op_feature_columns(op,4,FEAT); out=f"{S}/l1_datavector_cache/flat_local_{op}"
    os.makedirs(out,exist_ok=True)
    for split in ("l1_train","l1_val"):
        d=np.load(f"{BOTH}/{split}.npz")
        np.savez(f"{out}/{split}.npz", theta=d["theta"], x=d["x"][:,cols].astype(np.float32))
        print(f"  {op}/{split}: {d['x'].shape} -> {(d['x'].shape[0],len(cols))}",flush=True)
    shutil.copy(f"{BOTH}/flat_local_ranges.npy", f"{out}/flat_local_ranges.npy")
    f=np.load(f"{S}/fid_raw_both.npz")
    np.savez(f"{S}/fid_raw_{op}.npz", S=f["S"][:,cols].astype(np.float32),
             perm=f["perm"], patch=f["patch"])
    print(f"  {op}: fid {f['S'].shape} -> {(f['S'].shape[0],len(cols))}",flush=True)
print("SLICING DONE")
PY

#!/bin/bash
#SBATCH --job-name=env_jaxili
#SBATCH --account=nzu@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/env_build_%j.out

# Rebuild the paper's python env ("jaxili") on Jean-Zay.
# Version pins come from the surviving Learn2Map venv in the transferred tree
# (cnn_sbi/learn2map/lib/python3.11/site-packages) -- the paper's own conda env
# did not survive the Titan failure.
# JAX + TensorFlow + torch all touch the GPU in this pipeline (TF uses
# set_memory_growth; the L1 driver returns a torch.device), so all three are
# installed with CUDA support -- matching the original setup.

set -uo pipefail

export http_proxy=http://prodprox.idris.fr:3128
export https_proxy=http://prodprox.idris.fr:3128

WORKDIR=/lustre/fswork/projects/rech/prk/ulx34io
ENVP=$WORKDIR/envs/jaxili
export PIP_CACHE_DIR=$WORKDIR/.cache/pip     # keep pip cache off $HOME (inode-limited)

module purge
module load miniforge

echo "=== creating $ENVP (python 3.11) ==="
conda create -p "$ENVP" python=3.11 -y || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENVP" || exit 1
python -V

python -m pip install -U pip wheel setuptools

step() { echo; echo "=== pip install $* ==="; pip install "$@" || { echo "FAILED: $*"; return 1; }; }

# numpy first: TF 2.18 requires numpy<2, and the original env pinned 1.26.4
step "numpy==1.26.4"
step "jax[cuda12]==0.5.0"
step "tensorflow==2.18.0" "tensorflow-datasets==4.9.7" "tensorflow-probability==0.25.0"
step "dm-haiku==0.0.13" "optax==0.2.4" "distrax==0.1.5"
step "getdist==1.6.1" "healpy==1.18.0" "h5py==3.13.0"
step scikit-learn matplotlib tqdm wandb
step torch                                    # default PyPI wheel bundles CUDA 12
# --- the three non-obvious dependencies -----------------------------------
step "wl_stats_torch==0.1.0"                  # PyPI, authored by A. Tersenov
step "jaxili @ git+https://github.com/sachaguer/jaxili.git"
step "sbi_lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git"

echo
echo "=== import check (CPU only; GPU verified separately in a GPU job) ==="
python - <<'PY'
mods = ["numpy","jax","jaxlib","tensorflow","tensorflow_datasets",
        "tensorflow_probability","haiku","optax","distrax","getdist",
        "healpy","h5py","sklearn","matplotlib","tqdm","wandb",
        "torch","wl_stats_torch","jaxili","sbi_lens"]
bad = []
for m in mods:
    try:
        mod = __import__(m)
        print(f"  OK   {m:24s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:24s} {type(e).__name__}: {e}")
        bad.append(m)
print("\nFAILED:", bad if bad else "none")
PY

echo
echo "=== pip freeze -> env_freeze.txt ==="
pip freeze > "$WORKDIR/recovery/env_freeze.txt"
echo "done: $ENVP"

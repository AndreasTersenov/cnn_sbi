#!/bin/bash
#SBATCH --job-name=install_env
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/install_env_%j.out

# Offline install of the paper env from the pre-downloaded wheelhouse.
#
# Layout follows guide section 4: env (many small files) on $WORK of the *nzu*
# project -- prk's WORK is at 91.6% of its 500k inode cap -- and the 12 GB
# wheelhouse (bulk I/O, disposable) on $SCRATCH.
#
# Two failures from the previous attempt are fixed here:
#  1. Pins were being overridden. With --find-links, pip picks the NEWEST matching
#     wheel for any unconstrained requirement, so dm-haiku's "numpy>=1.18" pulled
#     numpy 2.4.6 and chex 0.1.92 dragged jax 0.5.0 -> 0.10.2, breaking TF 2.18.
#     Now: one single transaction under a hard constraints file.
#  2. "No space left on device" building the sbi_lens sdist -- that was pip's temp
#     dir on the compute node, not the wheelhouse. TMPDIR now points at SCRATCH.
#
# tf_keras is required by tensorflow-probability 0.25 (and hence distrax) on TF>=2.16.
# jaxili / sbi_lens / wl_stats_torch go in with --no-deps per the repo's own
# guidance (scripts/sbi/l1_vmim_run_commands.txt); their unpinned trees conflict.
# lenstools is deliberately omitted: the rescued-code import closure never uses it.

set -uo pipefail
module purge
module load python/3.11.5

N=/lustre/fswork/projects/rech/nzu/ulx34io
S=/lustre/fsn1/projects/rech/nzu/ulx34io
ENVP=$N/envs/jaxili
WH=$S/recovery/wheelhouse
CONS=$N/recovery/constraints.txt
PIP="$ENVP/bin/pip"
PY="$ENVP/bin/python"

export TMPDIR=$S/recovery/tmp          # compute-node /tmp is too small for torch/sdists
mkdir -p "$TMPDIR"
OFF="--no-index --find-links=$WH -c $CONS"

$PY -V
echo "wheelhouse: $(ls $WH | wc -l) files, $(du -sh $WH | cut -f1)"
echo "constraints:"; sed 's/^/    /' "$CONS"

echo; echo "=== single constrained transaction ==="
$PIP install $OFF \
  "numpy==1.26.4" "ml-dtypes==0.4.1" \
  "jax[cuda12]==0.5.0" \
  "tensorflow==2.18.0" "tensorflow-datasets==4.9.7" \
  "tensorflow-probability==0.25.0" "tf_keras==2.18.0" \
  "dm-haiku==0.0.13" "optax==0.2.4" "chex==0.1.88" "distrax==0.1.5" \
  "getdist==1.6.1" "healpy==1.18.0" "h5py==3.13.0" \
  scikit-learn matplotlib tqdm wandb torch \
  || echo "!!! main transaction FAILED"

echo; echo "=== custom packages (--no-deps, --no-build-isolation) ==="
for pkg in wl_stats_torch jaxili sbi_lens; do
  echo "--- $pkg"
  $PIP install --no-index --find-links=$WH --no-deps --no-build-isolation "$pkg" \
    || echo "!!! FAILED: $pkg"
done

echo; echo "=== import check ==="
$PY - <<'PY'
import warnings, os
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
mods = ["numpy","jax","jaxlib","tensorflow","tensorflow_datasets","tensorflow_probability",
        "haiku","optax","distrax","getdist","healpy","h5py","sklearn","matplotlib",
        "tqdm","wandb","torch","wl_stats_torch","jaxili","sbi_lens"]
bad=[]
for m in mods:
    try:
        mod=__import__(m); print(f"  OK   {m:24s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:24s} {type(e).__name__}: {str(e)[:80]}"); bad.append(m)
print("\nFAILED:", bad if bad else "none")

# the pins that actually matter for correctness
import numpy, jax, tensorflow as tf
print(f"\npin check: numpy={numpy.__version__} (want 1.26.4)  "
      f"jax={jax.__version__} (want 0.5.0)  tf={tf.__version__} (want 2.18.0)")
PY

echo; echo "=== pip check ==="
$PIP check || true
$PIP freeze > /lustre/fswork/projects/rech/prk/ulx34io/recovery/env_freeze.txt
echo "wrote env_freeze.txt"

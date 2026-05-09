#!/usr/bin/env bash
# B1 SBC – harmonic-L1 no-BNT rank-statistics
# Usage: bash run_sbc_b1_harm.sh [smoke|n200|n1000|all]
# Default (no arg): runs all three stages sequentially.
#
# GPU policy: GPU 0 only, XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/sbi/run_sbc_harm_l1_nobnt.py"
OUTPUT_ROOT="$REPO_ROOT/scripts/sbi/results/diagnostics/sbc_harm_l1_nobnt"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN="${1:-all}"

run_sbc() {
    local n_ranks="$1"
    local posterior_samples="$2"
    local seed="$3"
    echo "=== SBC harmonic-L1 no-BNT: n_ranks=${n_ranks} posterior_samples=${posterior_samples} seed=${seed} ==="
    conda run -n jaxili python "$SCRIPT" \
        --n-ranks "$n_ranks" \
        --posterior-samples "$posterior_samples" \
        --rank-seed "$seed" \
        --output-root "$OUTPUT_ROOT" \
        --cuda-visible-devices 0 \
        --xla-mem-fraction 0.45
}

if [[ "$RUN" == "smoke" || "$RUN" == "all" ]]; then
    run_sbc 5 300 101
fi

if [[ "$RUN" == "n200" || "$RUN" == "all" ]]; then
    run_sbc 200 2000 20260507
fi

if [[ "$RUN" == "n1000" || "$RUN" == "all" ]]; then
    run_sbc 1000 2000 20260507
fi

echo "=== B1 harmonic-L1 SBC complete. Outputs in $OUTPUT_ROOT ==="

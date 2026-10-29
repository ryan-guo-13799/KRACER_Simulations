#!/usr/bin/env bash
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

HORIZON=${HORIZON:-10000}
SEED_COUNT=${SEED_COUNT:-20}
VALIDATION_SEED_COUNT=${VALIDATION_SEED_COUNT:-4}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}

mkdir -p "$OUTPUT_ROOT"
run_dir=$(mktemp -d -p "$OUTPUT_ROOT" all_algorithms_XXXXXXXX) || exit 1
echo "Running all eight methods. Results and logs: $run_dir"

# BEGIN RUNNER CLEANUP: one Python entry point schedules every eligible method.
python3 -u src/simulation_main.py \
    --output "$run_dir/results" \
    --horizon "$HORIZON" --seed-count "$SEED_COUNT" \
    --validation-seed-count "$VALIDATION_SEED_COUNT" \
    "$@" 2>&1 | tee "$run_dir/run.log"
# END RUNNER CLEANUP.

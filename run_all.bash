#!/usr/bin/env bash
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Override these environment variables for a small trial, if desired.
HORIZON=${HORIZON:-10000}
SEED_COUNT=${SEED_COUNT:-20}
VALIDATION_SEED_COUNT=${VALIDATION_SEED_COUNT:-4}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}

mkdir -p "$OUTPUT_ROOT"
run_dir=$(mktemp -d -p "$OUTPUT_ROOT" all_algorithms_XXXXXXXX) || exit 1
echo "Running eight methods in parallel. Results and logs: $run_dir"
echo "The four pairwise baselines run only the five K=2 cases."
pids=()

for algorithm in kracer analytical_certificate minimum_eigenvector without_restart
do
    printf -v prefix '%-22s' "$algorithm"
    python3 -u src/simulation_main.py \
        --algorithms "$algorithm" --output "$run_dir/$algorithm" \
        --horizon "$HORIZON" --seed-count "$SEED_COUNT" \
        --validation-seed-count "$VALIDATION_SEED_COUNT" \
        2>&1 | sed -u "s/^/[$prefix] /" | tee "$run_dir/$algorithm.log" &
    pids+=("$!")
done

for algorithm in raes dbgd doubler sparring
do
    printf -v prefix '%-22s' "$algorithm"
    python3 -u src/run_pairwise_baselines.py \
        --algorithm "$algorithm" --output "$run_dir/$algorithm" \
        --horizon "$HORIZON" --seed-count "$SEED_COUNT" \
        --validation-seed-count "$VALIDATION_SEED_COUNT" \
        2>&1 | sed -u "s/^/[$prefix] /" | tee "$run_dir/$algorithm.log" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"
do
    wait "$pid" || failed=1
done

if [ "$failed" -eq 0 ]; then
    echo "All methods finished successfully. Results and logs: $run_dir"
else
    echo "At least one method failed. Check the logs in $run_dir."
fi
exit "$failed"

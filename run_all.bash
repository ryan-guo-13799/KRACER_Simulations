#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# BEGIN PARALLEL RUN: settings affect process scheduling, not the algorithms.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON=${PYTHON:-python}
JOBS=${JOBS:-8}
HORIZON=${HORIZON:-10000}
SEED_COUNT=${SEED_COUNT:-20}
START_SEED=${START_SEED:-6000}
VALIDATION_SEED_COUNT=${VALIDATION_SEED_COUNT:-4}
VALIDATION_START_SEED=${VALIDATION_START_SEED:-120000}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
algorithms=(kracer analytical_certificate minimum_eigenvector without_restart raes dbgd doubler sparring)

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS must be a positive integer." >&2
    exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python not found: $PYTHON. Set PYTHON to your Python executable." >&2
    exit 2
fi
for arg in "$@"; do
    case "$arg" in
        --output|--output=*|--algorithms|--algorithms=*)
            echo "This script manages --output and --algorithms. Use OUTPUT_ROOT for the output location." >&2
            exit 2 ;;
    esac
done
if (( JOBS > ${#algorithms[@]} )); then JOBS=${#algorithms[@]}; fi

mkdir -p "$OUTPUT_ROOT" || exit 1
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd -P) || exit 1
run_dir=$(mktemp -d "$OUTPUT_ROOT/parallel_XXXXXXXX") || exit 1
mkdir -p "$run_dir/results" "$run_dir/logs" "$run_dir/work" || exit 1
echo "Running eight methods, up to $JOBS at a time. Output: $run_dir"
# END PARALLEL RUN.

# BEGIN PARALLEL RUN: each process calls the existing main runner independently.
pids=()
names=()
failures=0

stop_workers() {
    trap - INT TERM
    for pid in $(jobs -pr); do kill "$pid" 2>/dev/null || true; done
    wait
    echo "Stopped. Partial outputs and logs are in $run_dir" >&2
    exit 130
}
trap stop_workers INT TERM

wait_first() {
    local status=0
    wait "${pids[0]}" || status=$?
    if (( status == 0 )); then
        echo "Finished ${names[0]}"
    else
        echo "Failed ${names[0]} (exit $status). See $run_dir/logs/${names[0]}.log" >&2
        failures=$((failures + 1))
    fi
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
}

for algorithm in "${algorithms[@]}"; do
    echo "Starting $algorithm"
    "$PYTHON" -B -u src/simulation_main.py \
        --horizon "$HORIZON" --seed-count "$SEED_COUNT" --start-seed "$START_SEED" \
        --validation-seed-count "$VALIDATION_SEED_COUNT" --validation-start-seed "$VALIDATION_START_SEED" \
        "$@" --algorithms "$algorithm" --output "$run_dir/work/$algorithm" \
        >"$run_dir/logs/$algorithm.log" 2>&1 &
    pids+=("$!")
    names+=("$algorithm")
    if (( ${#pids[@]} >= JOBS )); then wait_first; fi
done
while (( ${#pids[@]} > 0 )); do wait_first; done
trap - INT TERM
# END PARALLEL RUN.

# BEGIN PARALLEL RUN: collect disjoint outputs into the usual case/algorithm tree.
# All move targets are below this run's newly created absolute directory.
for algorithm in "${algorithms[@]}"; do
    work_dir="$run_dir/work/$algorithm"
    [[ -d "$work_dir" ]] || continue
    for case_dir in "$work_dir"/*; do
        [[ -d "$case_dir/$algorithm" ]] || continue
        destination="$run_dir/results/${case_dir##*/}"
        mkdir -p "$destination" || exit 1
        [[ ! -e "$destination/$algorithm" ]] || exit 1
        mv -- "$case_dir/$algorithm" "$destination/" || exit 1
        rmdir -- "$case_dir" || exit 1
    done
    rmdir -- "$work_dir" || exit 1
done
rmdir -- "$run_dir/work" || exit 1
echo "Results: $run_dir/results"
echo "Logs:    $run_dir/logs"
if (( failures > 0 )); then
    echo "$failures method process(es) failed; inspect the logs above." >&2
    exit 1
fi
# END PARALLEL RUN.

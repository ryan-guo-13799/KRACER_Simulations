set -o pipefail
cd /mnt/d/comp_courses/comp3770/K-RACER/manual_simulations || exit 1

# Prevent each Python process from starting many numerical-library threads.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create a new folder for this parallel experiment.
mkdir -p output
run_dir=$(mktemp -d -p output parallel_XXXXXXXX)

pids=()

for algorithm in kracer analytical_certificate minimum_eigenvector without_restart
do
    printf -v prefix '%-22s' "$algorithm"
    python3 -u src/simulation_main.py \
        --algorithms "$algorithm" \
        --output "$run_dir/$algorithm" \
        2>&1 | sed -u "s/^/[$prefix] /" | tee "$run_dir/$algorithm.log" &

    pids+=("$!")
done

echo "Running in parallel. Results and logs: $run_dir"

failed=0
for pid in "${pids[@]}"
do
    wait "$pid" || failed=1
done

if [ "$failed" -eq 0 ]; then
    echo "All methods finished successfully."
else
    echo "At least one method failed. Check the logs in $run_dir."
fi

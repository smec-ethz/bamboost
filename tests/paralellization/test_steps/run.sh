#!/bin/bash

# Check if output directory provided
if [ $# -eq 0 ]; then
    echo "No output directory provided"
    exit 1
fi

out_dir="$1"
cd "$(dirname "$0")"

python3 ./create_test_sims.py $out_dir

for dir in $out_dir/*; do
    if [ -d "$dir" ]; then
        base_name=$(basename "$dir")
        start_time=$(date +%s.%N)
        bash "$dir/$base_name.sh"
        end_time=$(date +%s.%N)
        elapsed_time=$(echo "$end_time - $start_time" | bc)
        echo "$base_name: $elapsed_time"
    fi
done



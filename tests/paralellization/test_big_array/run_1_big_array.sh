#!/bin/bash

for dir in test_1_big_array/*; do
    if [ -d "$dir" ]; then
        base_name=$(basename "$dir")
        start_time=$(date +%s.%N)
        bash "$dir/$base_name.sh"
        end_time=$(date +%s.%N)
        elapsed_time=$(echo "$end_time - $start_time" | bc)
        echo "$base_name: $elapsed_time"
    fi
done

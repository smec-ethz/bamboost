#!/bin/bash

for dir in test/*; do
    if [ -d "$dir" ]; then
        base_name=$(basename "$dir")
        bash "$dir/$base_name.sh"
    fi
done

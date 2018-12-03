#!/usr/bin/env bash

BIN="matching-pii"
INPUT_PATH="input"
OUTPUT_PATH="output"

mkdir $OUTPUT_PATH
for ((n=2; n <=1024; n=n*2)); do
    declare -a coll
    for c in  {1..1000}; do
        coll+=("$(./$BIN $INPUT_PATH/${n}x${n}-$c.txt)")
    done

    sum=0.0
    for i in ${coll[@]}; do
        sum="$(echo "$sum + $i" | bc -l)"
    done

    echo "$sum / ${#coll[@]}" | bc -l > $OUTPUT_PATH/$n.txt
done

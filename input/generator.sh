#!/usr/bin/env bash

for ((i=2; i<=1024; i=i*2))
do
    for (( c=1; c<=1000; c++ ))
    do
        pwd
        ./generator $i $c
    done
done

cd ..

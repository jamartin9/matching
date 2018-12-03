#!/bin/bash

while [ $? -eq 0 ]
do
    ./matching-pii input/$1x$1.txt;
done

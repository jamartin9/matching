#!/bin/bash

while [ $? -eq 0 ]
do
    ./Debug/matching input/$1x$1.txt;
done

#!/bin/bash

for ((i=0; i<=9; i++)); do
    sbatch ./submit.sh $i
done

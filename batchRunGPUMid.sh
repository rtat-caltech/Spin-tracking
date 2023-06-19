#!/bin/bash

declare -i numIter=400
declare -i offset=1000
procID=$SLURM_PROCID
startPoint=$((numIter * procID))
for ((k = 0; k < $numIter; ++k)); do
	seed=$((k+startPoint+offset))
	./gpumain mediumEField.txt /netscratch/dmathews33/mediumEField/gpu$seed.out $seed
done

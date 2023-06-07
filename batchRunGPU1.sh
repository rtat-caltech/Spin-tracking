#!/bin/bash

declare -i numIter=400
declare -i offset=1000
procID=$SLURM_PROCID
startPoint=$((numIter * procID))
for ((k = 0; k < $numIter; ++k)); do
	seed=$((k+startPoint+offset))
	./gpumain integrator1.txt /netscratch/dmathews33/integratorTesting/int1_$seed.out $seed
done

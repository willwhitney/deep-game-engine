#!/bin/bash

#SBATCH --job-name=dataset_generator
#SBATCH --output=dataset_generator.out
#SBATCH --error=dataset_generator.err

./run_gpu breakout

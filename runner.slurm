#!/bin/bash

#SBATCH --job-name=dataset_generator_restart
#SBATCH --output=dataset_generator_restart.out
#SBATCH --error=dataset_generator_restart.err

./run_gpu breakout

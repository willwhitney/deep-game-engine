#!/bin/bash

#SBATCH --job-name=dataset_generator_more_trained
#SBATCH --output=dataset_generator_more_trained.out
#SBATCH --error=dataset_generator_more_trained.err

./run_gpu breakout

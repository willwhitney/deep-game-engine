#!/bin/bash

#SBATCH --job-name=dataset_generator_saving_params
#SBATCH --output=dataset_generator_saving_params.out
#SBATCH --error=dataset_generator_saving_params.err

./run_gpu breakout

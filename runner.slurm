#!/bin/bash

#SBATCH --job-name=DQN_freeway_saving_params
#SBATCH --output=slurm_logs/DQN_freeway_saving_params.out
#SBATCH --error=slurm_logs/DQN_freeway_saving_params.err

./run_gpu freeway

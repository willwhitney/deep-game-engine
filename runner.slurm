#!/bin/bash

#SBATCH --job-name=predict_forward_2frame
#SBATCH --output=prediction_tests/out
#SBATCH --error=prediction_tests/err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=6000
#SBATCH -p gpu

th predict_forward.lua --import z_pred_2frame_coder_rec_mark3_best_version_mark1_learning_rate_-0.00001 --coder rec_mark3_best  --name test2frame

#!/bin/bash
#SBATCH --job-name=LightGBM      # Job name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --mem=30gb                   # Job memory request
#SBATCH --partition=gpupart_48hour          # GPU Partition
#SBATCH --gpus-per-node=2            #number of GPUS
#SBATCH --time=1-23:00:00            # Time limit hrs:min:sec
#SBATCH --output=/home/m1/23CS60R48/MTP/GBM_output_%j.log    # Standard output and error log

#module load anaconda3
conda init
conda activate gpu
python3 Light_BGM.py
# this is workling!!!
#!/bin/bash
#SBATCH --job-name=GBM      # Job name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --mem=30gb                   # Job memory request
#SBATCH --partition=gpupart_p100          # GPU Partition
#SBATCH --gpus-per-node=2            #number of GPUS
#SBATCH --time=1-23:00:00            # Time limit hrs:min:sec
#SBATCH --output=/home/m1/23CS60R48/MTP/local_spatial/GBM_train_%j.log    # Standard output and error log

#module load anaconda3
cd local_spatial
pwd
conda init
conda activate gpu
python3 local_Light_GBM.py
# this is workling!!!
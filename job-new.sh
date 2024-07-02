#!/bin/bash
#SBATCH --job-name=unetSwin_job_4      # Job name
#SBATCH --nodes=2                    # Run all processes on a single node
#SBATCH --ntasks=2                   # Run a single task
#SBATCH --mem=30gb                   # Job memory request
#SBATCH --partition=gpupart_p100          # GPU Partition
#SBATCH --gpus-per-node=2            #number of GPUS
#SBATCH --time=1-23:00:00            # Time limit hrs:min:sec
#SBATCH --output=/home/m1/23CS60R48/MTP/unetSwin_test_%j.log    # Standard output and error log

#module load anaconda3
conda init
conda activate gpu
python3 unet_flair.py
# this is workling!!!
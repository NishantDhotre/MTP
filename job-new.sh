#!/bin/bash
#SBATCH --job-name=unetSwin_job         # Job name
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --mem=30gb                      # Job memory request
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --gpus-per-node=2               # Number of GPUs per node
#SBATCH --partition=gpupart_48hour      # Partition to submit to
#SBATCH --time=1-23:59:00               # Time limit hrs:min:sec
#SBATCH --output=/home/m1/23CS60R48/MTP/output_%j.log   # Standard output and error log

module load anaconda3


source /home/m1/23CS60R48/MTP/.venv/bin/activate  # Activate your virtual environment
# pip install --user monai nibabel tqdm einops torch numpy==1.21.0
srun python /home/m1/23CS60R48/MTP/unetSwin.py

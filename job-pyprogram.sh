#!/bin/bash
#SBATCH --job-name=swinunetr_training   # Job name
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --mem=16gb                      # Job memory request (adjust as needed)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task (adjust as needed)
#SBATCH --gpus-per-node=1               # Number of GPUs (adjust as needed)
#SBATCH --partition=gpupart_48hour      # Partition for 48-hour jobs
#SBATCH --time=1-23:00:00               # Time limit hrs:min:sec (48 hours)
#SBATCH --output=swinunetr_%j.log       # Standard output and error log

# Load necessary modules
# module load python/3.10.12               # Load the correct Python version
# module load cuda/11.1                    # Adjust based on the available CUDA version
# module load cudnn/8.0.5                  # Adjust based on the available cuDNN version

# Activate the virtual environment
# source .venv/bin/activate

# Ensure necessary packages are installed (if running the job for the first time)
pip install --user monai nibabel tqdm einops torch numpy==1.21.0
# pip install --user monai nibabel tqdm einops torch numpy==1.21.0

# Run the Python script
srun python3 /home/m1/23CS60R48/MTP/unetSwin.py
 
#!/bin/bash
#SBATCH --job-name=swin_unetr_training    # Job name
#SBATCH --partition=gpupart_48hour        # Partition name (gpu partition for 48 hours)
#SBATCH --nodes=1                        # Run all processes on a single node
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --mem=32gb                       # Job memory request
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --time=47:59:00                  # Time limit hrs:min:sec (just under 48 hours)
#SBATCH --output=swin_unetr_%j.log       # Standard output and error log

# Load necessary modules
module load python/3.8                   # Adjust based on the available Python version
module load cuda/11.1                    # Adjust based on the available CUDA version
module load cudnn/8.0.5                  # Adjust based on the available cuDNN version

# Activate the virtual environment
source .venv/bin/activate

# Ensure necessary packages are installed
pip install --user monai nibabel tqdm einops torch 

# Run the Python script
srun python3 unetSwin.py

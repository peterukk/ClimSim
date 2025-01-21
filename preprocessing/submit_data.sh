#!/bin/sh 
#SBATCH --job-name=create_data
# name for the queue  
#SBATCH --output=%x.%j.out
# output filename  
#SBATCH --partition=shared 
# use default queue 
#SBATCH --ntasks=1
# No parallelisation
#SBATCH --time=12:00 
# Guide on how long this will run for 
#SBATCH --mem-per-cpu=100gb
# Bigger jobs need more memory allocation

. "/home/u/ukkonen/nobackups/miniforge/etc/profile.d/conda.sh"
conda activate py311_torch_cu121

python create_npy_data.py

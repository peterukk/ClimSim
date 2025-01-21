#!/bin/sh 
#SBATCH --job-name=get_data
# name for the queue  
#SBATCH --partition=shared 
# use default queue 
#SBATCH --ntasks=1
# No parallelisation
#SBATCH --time=2:00:00 
# Guide on how long this will run for 
## SBATCH --mem-per-cpu=2gb
# Bigger jobs need more memory allocation
#SBATCH --array=3-12
#SBATCH --out=logs/download_0001-%a.out

. "/home/u/ukkonen/nobackups/miniforge/etc/profile.d/conda.sh"
which python
conda env list
conda activate py311_torch_cu121
which python

cd /network/group/aopp/predict/HMC009_UKKONEN_CLIMSIM/ClimSim_data/ClimSim_low-res-expanded/train

# Run script for months in array
month=$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")

year=0006
# run for inputs and outputs
echo $year $month 
# mystr=$(printf '%s %s' $year $month)
# echo $mystr
# printf '%s' $mystr

bash get_data_sp.sh $year $month

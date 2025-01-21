#!/bin/sh 
#SBATCH --job-name=concat
# name for the queue  
#SBATCH --output=%x.%j.out
# output filename  
#SBATCH --partition=shared 
# use default queue 
#SBATCH --ntasks=1
# No parallelisation
#SBATCH --time=00:30:00 
# Guide on how long this will run for 
#SBATCH --mem-per-cpu=80gb
# Bigger jobs need more memory allocation


echo $PATH
which python
#source /home/u/ukkonen/.bashrc
echo $PATH
which conda
. "/home/u/ukkonen/nobackups/miniforge/etc/profile.d/conda.sh"
which python
conda env list
conda activate py311_torch_cu121
which python

cd /network/group/aopp/predict/HMC009_UKKONEN_CLIMSIM/ClimSim/preprocessing/

bash -c "python -c \"print('hello world'); import sys; print(sys.version) \""

echo "running python"
python3 concatenate_files.py
echo "got here"
#python create_npy_data.py &
wait

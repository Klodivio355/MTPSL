#!/bin/bash -l

# set the number of nodes
#SBATCH --nodes=1

#SBATCH --partition=shi_gpu

# set name of job
#SBATCH --job-name=mtl

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL 

# send mail to this address
#SBATCH --mail-user=maxime.fontana@kcl.ac.uk

conda activate /scratch_tmp/grp/grv_shi/k21220263/conda/mtl-partial

# run the application
python nyu_mtl_xtc.py --out ./results/nyuv2 --ssl-type randomlabels --dataroot ./data/nyuv2
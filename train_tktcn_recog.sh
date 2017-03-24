#!/bin/bash -l

#SBATCH
#SBATCH --job-name=CV_vanila
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=tkim60@jhu.edu

#### load and unload modules you may need
module restore mymodules

echo "Using GPU Device:"
echo $CUDA_VISIBLE_DEVICES

python /home-4/tkim60@jhu.edu/scratch/dev/nturgbd/train.py --gpu=$CUDA_VISIBLE_DEVICES > /home-4/tkim60@jhu.edu/scratch/dev/nturgbd/CV_TKTCN_D0.5_L3_F25_vanila_$SLURM_JOBID.log
echo "Finished with job $SLURM_JOBID"

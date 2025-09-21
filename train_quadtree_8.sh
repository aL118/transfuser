#!/bin/bash

# Define a base name for the job, experiment, and output files

#SBATCH --job-name=test                            
#SBATCH --output=/fs/nexus-scratch/aliu1237/transfuser/my_dump/slurm_output/%x.out.%j       # indicates a file to redirect STDOUT to; %j is the jobid
#SBATCH --error=/fs/nexus-scratch/aliu1237/transfuser/my_dump/slurm_output/%x.out.%j        # indicates a file to redirect STDERR to; %j is the jobid

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --ntasks=32

# set up notification settings for failures
##SBATCH --mail-user=angelaliu9805@gmail.com
##SBATCH --mail-type=ALL

## Scavenger training config (low priority, unlimited resources)
##SBATCH --time=48:00:00
##SBATCH --qos=scavenger
##SBATCH --account=scavenger
##SBATCH --partition=scavenger

## GAMMA training config
#SBATCH --time=48:00:00     
#SBATCH --qos=huge-long                                    
#SBATCH --account=gamma
#SBATCH --partition=gamma

eval "$(conda shell.bash hook)"
conda activate tfuse

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Number of GPUS: $NUM_GPUS"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 --max_restarts=0 \
 --rdzv_id=1234576890 --rdzv_backend=c10d /fs/nexus-scratch/aliu1237/transfuser/team_code_transfuser/train.py \
 --logdir /fs/nexus-scratch/aliu1237/transfuser/scratch2 --root_dir /fs/nexus-scratch/aliu1237/transfuser/data \
 --load_file /fs/nexus-scratch/aliu1237/transfuser/logdir_quadtree/transfuser/model_40.pth \
 --parallel_training 1 --batch_size 16 --save_every 5 --epochs 100 --backbone 'quadtree' --prefix 'round2_'

## sbatch -J tfuse_qt2 train_quadtree_8.sh
## model_10 = epoch 29
#!/bin/bash

#SBATCH --time=150:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=M_5
#SBATCH --output=newton_output/slurm-%j.out
#SBATCH --constraint=gpu80
######SBATCH --partition=preemptable --qos preemptable


# srun --pty -t72:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 bash
# srun --pty -t50:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 bash
# srun --pty -t48:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 --partition=preemptable --qos preemptable bash


nvidia-smi
nvidia-smi -q |grep -i serial

source ~/.bashrc
module load anaconda/anaconda-2023.09 
CONDA_BASE=$(conda info --base) ;
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate /home/ppathak/my-envs/pathak3
module --ignore_cache load "cuda/cuda-11.3"
module --ignore_cache load gcc/gcc-11.2.0

cho -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
echo $SLURM_JOB_ID $SLURM_JOB_NODELIST
echo $CONDA_DEFAULT_ENV

cd ~/ICCV-CSCI-Person-ReID/
mkdir newton_output


scontrol write batch_script $SLURM_JOB_ID
mv slurm-$SLURM_JOB_ID.sh newton_output/
rsync -a newton_output/slurm-$SLURM_JOB_ID.sh ucf2:~/ICCV-CSCI-Person-ReID/newton_output/
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}


GPUS=0,1
PORT=12345
NUM_GPU=2
BATCH_SIZE=40
RUN_NO=1

ENV='nccl'

############################## CCVID ##############################
ccvid=/datasets/CCVID-lzo/
CONFIG=configs/ccvid_eva02_l_cloth.yml
wt=logs/CCVID/CCVID_IMG/eva02_l_cloth_best.pth
DATASET="ccvid"
ROOT=$ccvid
PORT=12357
COLOR=34



# ############################## MEVID ##############################
# mevid=/datasets/MEVID-lzo
# CONFIG=configs/mevid_eva02_l_cloth.yml
# wt=logs/MEVID/MEVID_IMG2/eva02_l_cloth_best.pth
# DATASET="mevid"
# ROOT=$mevid
# PORT=12362



############################## CCVID ##############################
# ccvid=/data/priyank/synthetic/CCVID
# CONFIG=configs/ccvid_eva02_l_cloth.yml
# DATASET="ccvid"
# ROOT=$ccvid
# PORT=12357

# #################### MEVID ####################
# mevid=/data/priyank/synthetic/MEVID/
# CONFIG=configs/mevid_eva02_l_cloth.yml
# DATASET="mevid"
# ROOT=$mevid
# PORT=12359



# ####### EZ CLIP Baseline (no clothes / colors)
# #### vid-ez E2E (w/ pretrained) NoAd + Motion LOSS
# SEED=1245
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --env $ENV --resume --config_file $CONFIG DATA.ROOT $ROOT \
#     MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt MODEL.MOTION_LOSS True SOLVER.SEED $SEED SOLVER.MAX_EPOCHS 100 >> newton_output/"$DATASET"_4T_NoAd_e2e_pre_ml-RUN-$SEED-EP100.txt



####### EZ CLIP Baseline (no clothes / colors)
#### vid-ez E2E (w/ pretrained) NoAd + Motion LOSS
COLOR=49
SEED=1245
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train_two_step.py --env $ENV --resume --config_file $CONFIG DATA.ROOT $ROOT \
    TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
    TRAIN.DIR_TEACH1 $ROOT TRAIN.TEACH1_MODEL None TRAIN.TEACH1_LOAD_AS_IMG True TRAIN.TEACH_DATASET_FIX 'color_adv' TRAIN.COLOR_ADV True \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED OUTPUT_DIR $DATASET-$COLOR-$SEED SOLVER.MAX_EPOCHS 100 SOLVER.LOG_PERIOD 800 >> newton_output/"$DATASET"_4NAEPM+CO-$COLOR-$SEED-Final.txt
    

        
        
# rsync -a Dump/ccvid-9-1245/ ucf2:~/ICCV-CSCI-Person-ReID/Dump/
        
        
        




rsync -a newton_output/* ucf2:~/ICCV-CSCI-Person-ReID/newton_output/
rm *.pth
rm mAP/* 
rm rank/*
rm train/*

# cd ~/ICCV-CSCI-Person-ReID/
# sbatch Script/image_batch_newton.sh




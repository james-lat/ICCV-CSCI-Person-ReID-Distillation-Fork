#!/bin/bash

#SBATCH --time=150:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=M_2
#SBATCH --output=newton_output/slurm-%j.out
#SBATCH --constraint=gpu80
######SBATCH --partition=preemptable --qos preemptable


# srun --pty -t72:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 bash
# srun --pty -t50:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 bash
# srun --pty -t48:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 --partition=preemptable --qos preemptable bash
# srun --pty -t24:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu32 --partition=preemptable --qos preemptable bash


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


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"




############################## CCVID ##############################
ccvid=/datasets/CCVID-lzo/
# ccvid=/data/priyank/synthetic/CCVID
CONFIG=configs/ccvid_eva02_l_cloth.yml
wt=logs/CCVID/CCVID_IMG/eva02_l_cloth_best.pth
DATASET="ccvid"
ROOT=$ccvid

COLOR=49
SEED=1245



############################## MEVID ##############################
mevid=/datasets/MEVID-lzo
# mevid=/data/priyank/synthetic/MEVID/
CONFIG=configs/mevid_eva02_l_cloth.yml
DATASET="mevid"
wt=logs/MEVID/MEVID_IMG2/eva02_l_cloth_best.pth
DATASET="mevid"
ROOT=$mevid

COLOR=48
SEED=1245


###### #VANILL IMAGE TRAIN  (need this to train EZ-CLIP)
SEED=1244
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT MODEL.DIST_TRAIN True \
    OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> ucf_output/"$DATASET"_img_nocloth-$SEED.txt    


# ####### EZ CLIP Baseline (no clothes / colors)
# #### vid-ez E2E (w/ pretrained) NoAd + Motion LOSS
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --env $ENV --resume --config_file $CONFIG DATA.ROOT $ROOT MODEL.DIST_TRAIN True \
#     MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt MODEL.MOTION_LOSS True \
#     OUTPUT_DIR $DATASET-4TNAE2EPML-$SEED SOLVER.SEED $SEED SOLVER.MAX_EPOCHS 100 >> newton_output/"$DATASET"_4TNAE2EPML-RUN-$SEED-EP100.txt


# ####### EZ CLIP + COLORS
# #### vid-ez E2E (w/ pretrained) NoAd + Motion LOSS
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train_two_step.py --env $ENV --resume --config_file $CONFIG DATA.ROOT $ROOT MODEL.DIST_TRAIN True \
#     TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
#     TRAIN.DIR_TEACH1 $ROOT TRAIN.TEACH1_MODEL None TRAIN.TEACH1_LOAD_AS_IMG True TRAIN.TEACH_DATASET_FIX 'color_adv' TRAIN.COLOR_ADV True \
#     MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED OUTPUT_DIR $DATASET-$COLOR-$SEED SOLVER.MAX_EPOCHS 100 SOLVER.LOG_PERIOD 800 >> newton_output/"$DATASET"_4NAEPM+CO-$COLOR-$SEED-Final.txt
    



    
    
    
    
    
    


# rsync -a Dump/ccvid-9-1245/ ucf2:~/ICCV-CSCI-Person-ReID/Dump/
        
        
rsync -a newton_output/* ucf2:~/ICCV-CSCI-Person-ReID/newton_output/
rm *.pth
rm mAP/* 
rm rank/*
rm train/*

# cd ~/ICCV-CSCI-Person-ReID/
# sbatch Script/image_batch_newton.sh




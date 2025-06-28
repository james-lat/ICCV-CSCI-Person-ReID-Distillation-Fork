#!/bin/bash

#SBATCH --time=150:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=M_4
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --constraint=gpu80
######SBATCH --partition=preemptable --qos preemptable

# srun --pty -t30:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu16 bash
# srun --pty -t30:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu32 bash
# srun --pty -t72:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 bash
# srun --pty -t100:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu32 bash
# srun --pty -t50:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 bash
# srun --pty -t48:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 --partition=preemptable --qos preemptable bash
# srun --pty -t48:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu32 --partition=preemptable --qos preemptable bash
# srun --pty -t48:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu16 --partition=preemptable --qos preemptable bash

# srun --pty -t200:00:00 --gres=gpu:2 --cpus-per-gpu=12 --constraint=gpu80 bash

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
scontrol write batch_script $SLURM_JOB_ID
mv slurm-$SLURM_JOB_ID.sh outputs/
rsync -a outputs/slurm-$SLURM_JOB_ID.sh ucf2:~/MADE_ReID/outputs/
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}

cd ~/MADE_ReID/
GPUS=0,1
PORT=12345
NUM_GPU=2
BATCH_SIZE=40
RUN_NO=1

ENV='nccl'

#################### LTCC ####################
ltcc=/groups/yrawat/LTCC/
CONFIG=configs/ltcc_eva02_l_cloth.yml
DATASET="ltcc"
ROOT=$ltcc
PORT=12345


#################### PRCC ####################
prcc=/groups/yrawat/PRCC/
CONFIG=configs/prcc_eva02_l_cloth.yml
DATASET="prcc"
ROOT=$prcc
PORT=12351




############################## CCVID ##############################
ccvid=/datasets/CCVID-lzo/
CONFIG=configs/ccvid_eva02_l_cloth.yml
wt=logs/CCVID/CCVID_IMG/eva02_l_cloth_best.pth
DATASET="ccvid"
ROOT=$ccvid
PORT=12357



############################## MEVID ##############################
mevid=/datasets/MEVID-lzo
CONFIG=configs/mevid_eva02_l_cloth.yml
wt=logs/MEVID/MEVID_IMG2/eva02_l_cloth_best.pth
DATASET="mevid"
ROOT=$mevid
PORT=12362

casiab=/home/c3-0/datasets/CASIA_B_STAR/
NLR50_Wt=logs/CASIA_B_STAR/CAL_casiab/NLR50_16_224/best_model.pth.tar


# # #### COLOR
# arr=(2 3 4  6 9 12 13 14 16 17 18 21 23 24 25 26 27 28 29 30 31 32 34 35 37 38 39 40 41 42 43 44 46 47 48 49)
# arr=(3 4 6 9 17 18 23 24 25 29 30 31 32 37 38 39 41 42 43 44 46 47 48)
# COLOR=48
# SEED=1244
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     teacher_student.py --resume --config_file $CONFIG DATA.ROOT $ROOT TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
#     TRAIN.DIR_TEACH1 $ROOT TRAIN.TEACH1_MODEL None TRAIN.TEACH1_LOAD_AS_IMG True TRAIN.TEACH_DATASET_FIX 'color_adv' TRAIN.COLOR_ADV True \
#     MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED OUTPUT_DIR $DATASET-$COLOR-$SEED >> outputs/"$DATASET"_4NAEPM+CO-$COLOR-RUN-$SEED-Newton-Final.txt

# # #### COLOR
# COLOR=48
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     teacher_student.py --resume --config_file $CONFIG DATA.ROOT $ROOT TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
#     TRAIN.DIR_TEACH1 $ROOT TRAIN.TEACH1_MODEL None TRAIN.TEACH1_LOAD_AS_IMG True TRAIN.TEACH_DATASET_FIX 'color_adv' TRAIN.COLOR_ADV True \
#     MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR OUTPUT_DIR $DATASET-$COLOR- >> outputs/"$DATASET"_4NAEPM+CO-$COLOR-RUN-NOS-EED-Newton-Final.txt


    
# arr=(2 3 5 14 21 24 39)
# arr=(44 13 25 23 42 47)
# SEED=1244

# arr=(32 43 48 12 17 35)
# arr=(41 18 38 9)
# SEED=1245

# arr=(2 17 18 23 25 35 38 43 44 48)
# SEED=1245

# arr=(21)
# SEED=1244

# for COLOR in "${arr[@]}"
# do
#     IMG_WT=Dump/mevid-$COLOR-$SEED/ez_eva02_vid_hybrid_extra_best.pth 
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         teacher_student.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $IMG_WT SOLVER.SEED $SEED \
#         TRAIN.TRAIN_VIDEO True DATA.DATASET $DATASET \
#         MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR >> outputs/$COLOR-$SEED.txt
# done


    



rsync -a outputs/*.txt ucf2:~/MADE_ReID/outputs/
rsync -a outputs/*.out ucf2:~/MADE_ReID/outputs/

# rsync -a outputs/* ucf2:~/MADE_ReID/outputs/
# rsync -a ~/MADE_ReID/outputs/*.txt ucf2:~/MADE_ReID/outputs/
# rsync -a ucf0:~/MADE_ReID/outputs/*.txt ~/MADE_ReID/outputs/



rm *.pth
rm mAP/* 
rm rank/*
rm train/*

# cd ~/MADE_ReID/
# sbatch Script/image_batch_newton.sh




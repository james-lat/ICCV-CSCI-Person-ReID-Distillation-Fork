#!/bin/bash

#SBATCH --job-name=M_5
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu

#SBATCH -C gmem24
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH -c10

#SBATCH -p gpu --qos=day
##############################SBATCH -p gpu --qos=short
############################SBATCH -C gmem11 --exclude=c1-7
#############################SBATCH -p gpu --qos=preempt -A preempt 
##############################SBATCH -C gmemT48 --gres=gpu:turing:2

##################SBATCH -p gpu -c12 --qos=preempt 
################SBATCH -p gpu -c8 --qos=preempt --exclude=c1-2
################SBATCH --partition preempt
################SBATCH -C "gmem12&infiniband"


# srun --pty --gres=gpu:2 --cpus-per-gpu=8 -C gmem48 --qos=preempt bash 
# srun --pty --gres=gpu:2 --mem-per-cpu=8G -c10 -C gmem12 --qos=preempt bash 
# srun --pty --gres=gpu:2 --mem-per-cpu=8G -c10 -C gmem12 --qos=preempt --gres=gpu:c3-2:1 bash 
# srun --pty --gres=gpu:2 --mem-per-cpu=8G -c10 -C gmem12 --qos=preempt --nodelist=c3-2 bash 
# srun --pty --gres=gpu:2 --qos=preempt --exclude=c3-4,c3-6,c2-0 -w c3-2 bash 
# conda activate C12
# srun --pty --gres=gpu:2 --mem-per-cpu=8G -c10 --qos=preempt --nodelist=c3-2 bash 


nvidia-smi
nvidia-smi -q | grep -i serial

source ~/.bashrc
CONDA_BASE=$(conda info --base) ; 
source $CONDA_BASE/etc/profile.d/conda.sh

echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
echo $SLURM_JOB_ID $SLURM_JOB_NODELIST
echo $CONDA_DEFAULT_ENV

cd ~/ICCV-CSCI-Person-ReID/
mkdir ucf_output
scontrol write batch_script $SLURM_JOB_ID
mv slurm-$SLURM_JOB_ID.sh ucf_output/
rsync -a ucf_output/slurm-$SLURM_JOB_ID.sh ucf2:~/ICCV-CSCI-Person-ReID/ucf_output/
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}



# conda activate bert
conda activate pathak 
GPUS=0,1

NUM_GPU=2
BATCH_SIZE=40 
RUN_NO=1

ENV='nccl'
if [[ "$SLURM_JOB_NODELIST" == "c1-2" ]]; then
        echo " **** USING GLOOO ***** "
        ENV='gloo'
    fi


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"


#################### LTCC ####################
ltcc=/home/c3-0/datasets/LTCC/
CONFIG=configs/ltcc_eva02_l_cloth.yml
DATASET="ltcc"
ROOT=$ltcc
COLOR=44
SEED=1245


#################### PRCC ####################
prcc=/home/c3-0/datasets/PRCC/prcc/
CONFIG=configs/prcc_eva02_l_cloth.yml
DATASET="prcc"
ROOT=$prcc
COLOR=41
SEED=1245


# ###############################################################################################
# ################################ # Img Train ###################################################
# VANILL TRAIN (no color no cloth) (VARRY SEEDS)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> ucf_output/"$DATASET"_img_nocloth-$SEED.txt    


# #### COLOR (VARRY SEEDS)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    OUTPUT_DIR $DATASET+"_Co-$COLOR"-$SEED >> ucf_output/"$DATASET"-CO-$COLOR-$SEED.txt




############################## CCVID ##############################
ccvid=/home/c3-0/datasets/CCVID
CONFIG=configs/ccvid_eva02_l_cloth.yml
DATASET="ccvid"
ROOT=$ccvid
COLOR=49
SEED=1245

wt=logs/CCVID/CCVID_IMG/eva02_l_cloth_best.pth




############################## MEVID ##############################
mevid=/home/c3-0/datasets/MEVID
CONFIG=configs/mevid_eva02_l_cloth.yml
DATASET="mevid"
ROOT=$mevid


###### #VANILL IMAGE TRAIN  (need this to train EZ-CLIP)
SEED=1244
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> ucf_output/"$DATASET"_img_nocloth-$SEED.txt    

####### EZ CLIP Baseline (no clothes / colors)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --resume --config_file $CONFIG_MEVID DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt  >> ucf_output/"$DATASET"_4T_NoAd_e2e_pre.txt


# ####### EZ CLIP + COLORS 
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train_two_step.py --env $ENV --resume --config_file $CONFIG DATA.ROOT $ROOT \
    TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
    TRAIN.DIR_TEACH1 $ROOT TRAIN.TEACH1_MODEL None TRAIN.TEACH1_LOAD_AS_IMG True TRAIN.TEACH_DATASET_FIX 'color_adv' TRAIN.COLOR_ADV True \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED OUTPUT_DIR $DATASET-$COLOR-$SEED SOLVER.MAX_EPOCHS 100 SOLVER.LOG_PERIOD 800 >> ucf_output/"$DATASET"_4NAEPM+CO-$COLOR-$SEED-UCF.txt
    





rsync -a ucf_output/* ucf2:~/ICCV-CSCI-Person-ReID/ucf_output/
# rsync -a ucf0:~/ICCV-CSCI-Person-ReID/ucf_output/* ~/ICCV-CSCI-Person-ReID/ucf_output/



# cd ~/ICCV-CSCI-Person-ReID/
# sbatch Script/image_batch.sh





#!/bin/bash

#SBATCH --job-name=M_1
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu

#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH -c10

##############################SBATCH -p gpu --qos=day
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


scontrol write batch_script $SLURM_JOB_ID
mv slurm-$SLURM_JOB_ID.sh ucf_output/
rsync -a ucf_output/slurm-$SLURM_JOB_ID.sh ucf2:~/ICCV-CSCI-Person-ReID/outputs/

echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}

cd ~/ICCV-CSCI-Person-ReID/
mkdir ucf_output
# conda activate bert
conda activate pathak 
GPUS=0,1


PORT=12355
NUM_GPU=2
BATCH_SIZE=40 
RUN_NO=1

ENV='nccl'
if [[ "$SLURM_JOB_NODELIST" == "c1-2" ]]; then
        echo " **** USING GLOOO ***** "
        ENV='gloo'
    fi


#################### LTCC ####################
ltcc=/home/c3-0/datasets/LTCC/
CONFIG=configs/ltcc_eva02_l_cloth.yml
DATASET="ltcc"
ROOT=$ltcc
PORT=12350
COLOR=26


###############################################################################################
################################ # Img Train ###################################################
# VANILL TRAIN (no color no cloth)
SEED=1244
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> ucf_output/"$DATASET"_img_nocloth-$SEED.txt    


# #### COLOR
SEED=1234
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    OUTPUT_DIR $DATASET+"_Co-$COLOR" >> ucf_output/"$DATASET"-CO-$COLOR-$SEED.txt














# ltcc=/home/c3-0/datasets/LTCC/
# prcc=/home/c3-0/datasets/PRCC/prcc/



# # Img
# SEED=1244
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> outputs/"$DATASET"_img_nocloth-RUN-$SEED.txt    

# SEED=1245
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> outputs/"$DATASET"_img_nocloth-RUN-$SEED.txt    


# SEED=1244
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     SOLVER.SEED $SEED MODEL.NAME 'eva02_base_cloth'  >> outputs/"$DATASET"_E02-Base_nocloth-RUN-$SEED.txt    

# SEED=1245
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     SOLVER.SEED $SEED MODEL.NAME 'eva02_base_cloth'  >> outputs/"$DATASET"_E02-Base_nocloth-RUN-$SEED.txt    


# #### COLOR
arr2=(1245 1244)
arr=(5 9 14 17 18 23 26 27 32 39 44 49) 
PORT=12363
# for SEED in "${arr2[@]}"
# do
#     for COLOR in "${arr[@]}"
#     do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
#         TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#          >> outputs/"$DATASET"-CO-$COLOR-UCF-RUN-$SEED-FINAL.txt
#     done
# done

# # #### COLOR
# arr2=(1245 1244)
# arr2=(1245)
# arr2=(1244)
# arr=(5 9 14 17 18 23 26 27 32 39 44 49) 
# PORT=12364
# for SEED in "${arr2[@]}"
# do
#     for COLOR in "${arr[@]}"
#     do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token_base' \
#         TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#         >> outputs/"$DATASET"-E02-Base-CO-$COLOR-UCF-$SEED.txt
#     done
# done



# ###### NO MSE COLOR DISENTANGLE 
# PORT=12376
# SEED=1244
# SEED=1245
# COLOR=14
# RUN=16
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_no_token_color_mse' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED MODEL.ATT_DIRECT True \
#     OUTPUT_DIR $DATASET+"_Co-DT-$COLOR-$SEED" >> outputs/"$DATASET"-CO-DT-$COLOR-$SEED-DUMP$RUN.txt

# ##### NO MSE COLOR DISENTANGLE 
# COLOR=26
# RUN=16
# PORT=12377
# SEED=1245
# SEED=1244
# SEED=1234
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_no_token_color_mse_project_reid' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED MODEL.ATT_DIRECT True \
#     OUTPUT_DIR $DATASET+"_Co-DT-PR-$COLOR-$SEED" >> outputs/"$DATASET"-CO-DT-PR-$COLOR-$SEED-DUMP$RUN.txt


# #### # Img + GREY (scratch)
# PORT=12356
# SEED=1244
# SEED=1234
# SEED=1245
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     SOLVER.SEED $SEED DATA.GREY_SCALE True MODEL.PRETRAIN False >> outputs/"$DATASET"_img_GREY-$SEED-3.txt    

# # Img Scratch
# PORT=12358 
# SEED=1234
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED MODEL.PRETRAIN False >> outputs/"$DATASET"_Scratch-RUN-$SEED.txt    






PORT=12375
SEED=1245
arr=(26 44)
SEED=1234
arr=(26)
# COLOR=44
# outputs/images/IMG_ONLY/LTCC/Colors/Done/UCF/reported/ltcc_PO--CO-26-UCF-RUN-1234-FINAL.txt
# outputs/images/IMG_ONLY/LTCC/Colors/Done/UCF/reported/ltcc_PO--CO-26-UCF-RUN-1245-FINAL.txt
# outputs/images/IMG_ONLY/LTCC/Colors/Done/UCF/meh/ltcc-CO-44-UCF-RUN-1245-FINAL.txt
# RUN=16
# for COLOR in "${arr[@]}"
# do
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#     OUTPUT_DIR $DATASET+"_Co-$COLOR-NW$RUN-$SEED" >> outputs/"$DATASET"-CO-$COLOR-RUN-$SEED-DUMP$RUN.txt
# done


##### Img + COLOR (Extra Token) [Unified Self-Attention]
PORT=12345
SEED=1244
COLOR=5
arr=(5 9 17)
arr=(14 23 26 27)
arr=(32 39 44 49)
# for COLOR in "${arr[@]}"
# do
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#     MODEL.UNIFIED_DIST True >> outputs/"$DATASET"-CO-$COLOR-SELF-ATTN-$SEED.txt
# done


# Img + POSE (Extra Token) + Masked Self Attention
PORT=12353
SEED=1244
COLOR=5
arr2=(1245 1244)
# arr=(5 9 17 18)
arr=(5 9 14 17 18 23 26 27 32 39 44 49)
# for SEED in "${arr2[@]}"
# do
#     for COLOR in "${arr[@]}"
#     do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
#         TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#         MODEL.MASKED_SEP_ATTN True >> outputs/"$DATASET"-CO-$COLOR-UCF-MASK-SELF-ATTN-$SEED.txt
#     done
# done

#### Clothes Disentanlge
PORT=12352
arr2=(1245 1234)
arr2=(1244)
COLOR=-1
# for SEED in "${arr2[@]}"
# do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token_CL' \
#         TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#         >> outputs/"$DATASET"-CL-UCF-$SEED.txt
# done
















# Img + COLOR (Extra Token) (Cosine)
NAME=32-Sep-COS49
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT OUTPUT_DIR $DATASET"_COLOR" DATA.DATASET $DATASET MODEL.EXTRA_DIM $DIM \
#     MODEL.NAME 'eva02_img_extra_token' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_COSINE True >> outputs/"$DATASET"_Co-ONLY_$NAME.txt    



#### Pose + Color
SEED=1245
PORT=12360
arr2=(1245 1234)

arr=(2 5 14 17 18 26 27 29 34 38 39 43 44 48) 
POSE_TYPE=R_LA_40
arr=(9 43) 
# for SEED in "${arr2[@]}"
# do
#     for COLOR in "${arr[@]}"
#     do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET"_rlq" MODEL.NAME 'eva02_img_extra_token_pose_color' \
#         DATA.POSE_TYPE $POSE_TYPE TRAIN.POSE True TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' \
#         TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED >> outputs/"$DATASET"_PO-$POSE_TYPE-CO-$COLOR-UCF-RUN-$SEED-FINAL.txt
#     done
# done

# #### Pose + Color + Cosine Dist 
# POSE_TYPE=R_LAC_30
# arr=(23) 
# for COLOR in "${arr[@]}"
# do
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET"_rlq" MODEL.NAME 'eva02_img_extra_token_pose_color' \
#     DATA.POSE_TYPE $POSE_TYPE TRAIN.POSE True TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' \
#     TRAIN.COLOR_PROFILE $COLOR TRAIN.COLOR_LOSS "cosine" >> outputs/"$DATASET"_PO-$POSE_TYPE-CO-$COLOR-COS-UCF.txt
# done

# #### Pose + Color + Unified Dist 
# PORT=12376
# POSE_TYPE=N_LAC_20
# arr=(23) 
# for COLOR in "${arr[@]}"
# do
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET"_rlq" MODEL.NAME 'eva02_img_extra_token_pose_color' \
#     DATA.POSE_TYPE $POSE_TYPE TRAIN.POSE True TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' \
#     TRAIN.COLOR_PROFILE $COLOR MODEL.UNIFIED_DIST True >> outputs/"$DATASET"-UniDist-PO-$POSE_TYPE-CO-$COLOR-UCF.txt
# done

# # #### Pose + Color # MASKED ATTN 
# PORT=12379
# POSE_TYPE=R_A_5
# arr=(3 9 23 24 32 35 42 49) 
# arr=(49) 
# for COLOR in "${arr[@]}"
# do
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET"_rlq" MODEL.NAME 'eva02_img_extra_token_pose_color' \
#     DATA.POSE_TYPE $POSE_TYPE TRAIN.POSE True TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' \
#     TRAIN.COLOR_PROFILE $COLOR MODEL.MASKED_SEP_ATTN True >> outputs/"$DATASET"_PO-$POSE_TYPE-CO-$COLOR-MK-ATT-UCF.txt
# done

# ##### Img + COLOR (Extra Token) [FEED COLORS]
# PORT=12352
# # arr=(3 2 5 9 14 17 18 23 24 26 27 29 32 34 35 38 39 42 43 44 48 49) 
# # arr=(14 17 18 23 27 29 32 34)
# # arr=(35 38 39 43 44 48 49) 
# arr=(16 21 25 28 34 40 41 44 47 48 49)
# for COLOR in "${arr[@]}"
# do
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR \
#     MODEL.NAME 'eva02_img_extra_token_feed' MODEL.ATT_AS_INPUT True >> outputs/"$DATASET"-CO-$COLOR-Feed-UCF.txt
# done








#################### PRCC ####################
prcc=/home/c3-0/datasets/PRCC/prcc/
prcc_sil=/home/c3-0/datasets/ID-Dataset/masks/PRCC/jpgs
SIL=$prcc_sil
CONFIG=configs/prcc_eva02_l_cloth.yml
DATASET="prcc"
ROOT=$prcc
PORT=12376

celeb=/home/c3-0/datasets/ID-Dataset/Celeb-reID/
celeb_wt=logs/CELEB/CELEB_IMG/eva02_l_cloth_best.pth

# # Img
# SEED=1244
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> outputs/"$DATASET"_img_nocloth2.txt    

# SEED=1245
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> outputs/"$DATASET"_img_nocloth2-RUN-$SEED.txt    


# SEED=1244
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     SOLVER.SEED $SEED MODEL.NAME 'eva02_base_cloth'  >> outputs/"$DATASET"_E02-Base_nocloth-RUN-$SEED.txt    

# SEED=1245
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT \
#     SOLVER.SEED $SEED MODEL.NAME 'eva02_base_cloth'  >> outputs/"$DATASET"_E02-Base_nocloth-RUN-$SEED.txt    


# # Img + COLOR (Extra Token) 
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT OUTPUT_DIR $DATASET"_COLOR" DATA.DATASET $DATASET MODEL.EXTRA_DIM $DIM \
#     MODEL.NAME 'eva02_img_extra_token' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' >> outputs/"$DATASET"_Co-ONLY_$NAME.txt    

# # Img + COLOR (Extra Token) (Cosine)
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT OUTPUT_DIR $DATASET"_COLOR" DATA.DATASET $DATASET MODEL.EXTRA_DIM $DIM \
#     MODEL.NAME 'eva02_img_extra_token' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_COSINE True >> outputs/"$DATASET"_Co-ONLY_$NAME.txt    


# #### COLOR
arr2=(1245 1234)
arr=(9 11 13 14 16 18)
PORT=12351
# for SEED in "${arr2[@]}"
# do
#     for COLOR in "${arr[@]}"
#     do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
#         TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#          >> outputs/"$DATASET"-CO-$COLOR-UCF-RUN-$SEED-FINAL.txt
#     done
# done


# #### COLOR
arr2=(1245 1244)
arr2=(1245)
arr=(9 11 13 14 16 18)
PORT=12365
# for SEED in "${arr2[@]}"
# do
#     for COLOR in "${arr[@]}"
#     do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token_base' \
#         TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#         >> outputs/"$DATASET"-E02-Base-CO-$COLOR-UCF-$SEED.txt
#     done
# done



# #### Clothes Disentanlge
PORT=12350
arr2=(1244)
COLOR=-1
# for SEED in "${arr2[@]}"
# do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token_CL' \
#         TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
#         >> outputs/"$DATASET"-CL-UCF-$SEED.txt
# done



############################## CELEB  ##############################
celeb=/home/c3-0/datasets/ID-Dataset/Celeb-reID/
CONFIG=configs/celeb_eva02_l_cloth.yml
DATASET="celeb"
ROOT=$celeb
PORT=12347


#################### LaST ####################
last=/home/c3-0/datasets/LaST/
last_sil=/home/c3-0/datasets/ID-Dataset/masks/LaST/jpgs
POSE=Script/Helper/LaST_Pose_Cluster.csv
last_gender=Scripts/Helper/LaST_Gender.csv
SIL=$last_sil
CONFIG=configs/last_eva02_l_cloth.yml
DATASET="last"
ROOT=$last
PORT=12348
celeb=/home/c3-0/datasets/ID-Dataset/Celeb-reID/
celeb_wt=logs/CELEB/CELEB_IMG/eva02_l_cloth_best.pth


# # Img
# PORT=12350
# SEED=1244
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT \
#     OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> outputs/"$DATASET"_img_nocloth-RUN-$SEED.txt    

# SEED=1245
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT \
#     OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> outputs/"$DATASET"_img_nocloth-RUN-$SEED.txt    


# # Img + COLOR (Extra Token) (Cosine)
# SEED=1244
# COLOR=3
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT OUTPUT_DIR $DATASET"_COLOR" DATA.DATASET $DATASET \
#     MODEL.NAME 'eva02_img_extra_token' TRAIN.COLOR_PROFILE $COLOR \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' SOLVER.SEED $SEED >> outputs/"$DATASET"_Co-ONLY_MSE-UCF-RUN-$SEED.txt    


# #### COLOR
arr2=(1245 1234)
arr=(2 3 4 5 6 9 12 13 14 16 17 18 21 23 24 25 26 27 28 29 30 31 32 34 35 37 38 39 40 41 42 43 44 46 47 48 49)
arr=(2 3 12)
arr=(17 32 34)
PORT=12352
# for SEED in "${arr2[@]}"
# do
#     for COLOR in "${arr[@]}"
#     do
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED >> outputs/"$DATASET"-CO-$COLOR-UCF-RUN-$SEED.txt
    
#     done
# done




    



#################### DeepChange ####################
deepchange=/home/c3-0/datasets/DeepChange/
deepchange_sil=/home/c3-0/datasets/ID-Dataset/masks/DeepChangeDataset/jpgs 
SIL=$deepchange_sil
CONFIG=configs/deepchange_eva02_l_cloth.yml
DATASET="deepchange"
ROOT=$deepchange
PORT=12370

celeb=/home/c3-0/datasets/ID-Dataset/Celeb-reID/
celeb_wt=logs/CELEB/CELEB_IMG/eva02_l_cloth_best.pth


# Img
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --env $ENV --config_file $CONFIG DATA.ROOT $ROOT \
#     OUTPUT_DIR $DATASET >> outputs/"$DATASET"_img_nocloth.txt    

# Img + POSE (Extra Token)
# PORT=12350
# for pose in $(seq 10 5 40)
# do  
#     echo $pose
#     POSE_TYPE=N_LAC_$pose
#     NAME=$POSE_TYPE-Sep
#     CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#         train.py --config_file $CONFIG DATA.ROOT $ROOT OUTPUT_DIR $DATASET"_POSE" \
#         MODEL.NAME 'eva02_img_extra_token_pose' \
#         DATA.DATASET $DATASET"_rlq" DATA.POSE_TYPE $POSE_TYPE MODEL.EXTRA_DIM 1024 TRAIN.POSE_ONLY True TRAIN.POSE True >> outputs/$DATASET"_"P-ONLY-$NAME.txt    
# done

# Img + COLOR (Extra Token)
# PORT=12369
# NAME=32-Sep-Cos6
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT OUTPUT_DIR $DATASET"_COLOR" DATA.DATASET $DATASET MODEL.EXTRA_DIM 1024 \
#     MODEL.NAME 'eva02_img_extra_token' \
#     TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' >> outputs/"$DATASET"_Co-ONLY_$NAME.txt    

# Img + Gender Classifier
# PORT=12351
# NAME=Cla-Gen
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG DATA.ROOT $ROOT OUTPUT_DIR $DATASET"_Gender" DATA.DATASET $DATASET \
#     MODEL.NAME 'eva02_img_extra_token_gender2' \
#     DATA.DATASET $DATASET"_rlq" TRAIN.GENDER True >> outputs/"$DATASET"-$NAME.txt    









# rm outputs/ltcc_img_faug.txt
# rm outputs/ltcc_img_nocloth.txt



# rsync -a eval.txt ucf2:~/MADE_ReID/outputs/
rsync -a outputs/*.txt ucf2:~/MADE_ReID/outputs/
rsync -a outputs/*.out ucf2:~/MADE_ReID/outputs/

# rsync -a ucf0:~/MADE_ReID/outputs/*.txt ~/MADE_ReID/outputs/
# rsync -a ucf0:~/MADE_ReID/outputs/*.out ~/MADE_ReID/outputs/




rm *.pth
rm mAP/* 
rm rank/*
rm train/*

# Celeb ReID image eva 


# rsync -r ~/MADE_ReID/logs/ ucf0:~/MADE_ReID/logs/
# cd ~/MADE_ReID/
# sbatch Script/image_batch.sh





conda activate bert
cd ~/ICCV-CSCI-Person-ReID/
NUM_GPU=1
PORT=12355
BATCH_SIZE=40
RUN_NO=1


#################### LTCC ####################
ltcc=/data/priyank/synthetic/LTCC/
CONFIG=configs/ltcc_eva02_l_cloth.yml
DATASET="ltcc"
ROOT=$ltcc
COLOR=26


###############################################################################################    
#################### PRCC ####################
prcc=/data/priyank/synthetic/PRCC/
CONFIG=configs/prcc_eva02_l_cloth.yml
DATASET="prcc"
ROOT=$prcc
COLOR=9


############################## CCVID ##############################
ccvid=/data/priyank/synthetic/CCVID
CONFIG=configs/ccvid_eva02_l_cloth.yml
DATASET="ccvid"
ROOT=$ccvid
PORT=12357




###############################################################################################
################################ # PROPOSED (COLORS) ###################################################
# VANILL TRAIN (no color no cloth)
SEED=1244
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> ucf_output/"$DATASET"_img_nocloth-RUN-$SEED.txt    


# #### COLOR
SEED=1234
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    OUTPUT_DIR $DATASET+"_Co-$COLOR" >> outputs/"$DATASET"-CO-$COLOR-UCF2-RUN-$SEED-FINAL.txt








###############################################################################################
################################ # Training ABLATION ###################################################
##### TRADITIONAL SELF_ATTENTION 
##### Img + COLOR (Extra Token) [Traditional Unified Self-Attention]
SEED=1245
COLOR=5
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    MODEL.UNIFIED_DIST True >> outputs/"$DATASET"-CO-$COLOR-TRAD-SELF-ATTN-$SEED.txt


##### Masked SELF_ATTENTION 
##### Img + COLOR (Extra Token) [Masked Self-Attention]
SEED=1245
COLOR=5
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    MODEL.MASKED_SEP_ATTN True >> outputs/"$DATASET"-CO-$COLOR-MASK-SELF-ATTN-$SEED.txt
    
    

##### FEED COLORS 
##### Img + COLOR (Extra Token) [FEED COLORS]
SEED=1245
COLOR=5
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    MODEL.NAME 'eva02_img_extra_token_feed' MODEL.ATT_AS_INPUT True >> outputs/"$DATASET"-CO-$COLOR-Feed.txt

    
##### GREY
SEED=1245
COLOR=5
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET \
    SOLVER.SEED $SEED DATA.GREY_SCALE True >> outputs/"$DATASET"_img_GREY-$SEED.txt    



################################ # Testing ###################################################    
# Img STATS GFLOP AND NUMBER OF PARAMS  
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT SOLVER.SEED $SEED TEST.MODE True \
    ANALYSIS_STATS True 
    
    
    






















# #### Clothes Disentanlge
PORT=12351
SEED=1244
COLOR=-1
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token_CL' \
    TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED 
    # >> outputs/"$DATASET"-CL-UCF2-RUN-$SEED.txt



    
    
    


# Img + Train Dump + GRAD_CAM 
IMG_WT=logs/prcc+_IMG/eva02_l_cloth_best.pth
SEED=1234
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $IMG_WT SOLVER.SEED $SEED \
    TEST.MODE True DATA.DATASET $DATASET TRAIN_DUMP True GRAD_CAM True AUG.RE_PROB 0.0 AUG.RC_PROB 0.0 AUG.RF_PROB 0.0  
    
# Img Train Dump  
IMG_WT=logs/prcc+_IMG/eva02_l_cloth_best.pth
SEED=1234
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $IMG_WT SOLVER.SEED $SEED \
    TEST.MODE True DATA.DATASET $DATASET TRAIN_DUMP True AUG.RE_PROB 0.0 AUG.RC_PROB 0.0 AUG.RF_PROB 0.0 TAG "PRCC-TRAIN" 
    

###############################################################################################
################################ # Img + COLOR EVAL ###################################################
IMG_WT='logs/PRCC/prcc-9-1245-16/eva02_img_extra_token_best.pth'
COLOR=9
SEED=1234
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $IMG_WT SOLVER.SEED $SEED \
    MODEL.NAME 'eva02_img_extra_token' TRAIN.COLOR_PROFILE $COLOR TEST.MODE True TAG "PRCC-Co-$COLOR" AUX_DUMP True 
    #  >> outputs/"$DATASET"_CO-$COLOR-EVAL_img-RUN-$SEED.txt    
# EVA-attribure.train:  CC :  CMC curve, Rank-1  :65.9%  Rank-5  :75.2%  Rank-10 :79.3%
# EVA-attribure.train:  CC :  mAP Acc. :61.1%
# EVA-attribure.train:  SC:  CMC curve, Rank-1  :100.0%  Rank-5  :100.0%  Rank-10 :100.0%  
# EVA-attribure.train:  SC:  mAP Acc. :98.8%

#### EARLY RETURN
IMG_WT='logs/prcc+_Co-9-NW/eva02_img_extra_token_best.pth'
COLOR=9
SEED=1234
BLOCKS=(0 6 12 18) 
for BLOCK in "${BLOCKS[@]}"
do  
    echo $BLOCK
    CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $IMG_WT SOLVER.SEED $SEED \
    MODEL.NAME 'eva02_img_extra_token' TRAIN.COLOR_PROFILE $COLOR TEST.MODE True TAG "PRCC-Co-$COLOR-$BLOCK" AUX_DUMP True MODEL.RETURN_EARLY $BLOCK
done



# Img + COLOR Train Dump + GRAD_CAM 
IMG_WT='logs/prcc+_Co-9-NW/eva02_img_extra_token_best.pth'
COLOR=9
SEED=1234
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $IMG_WT SOLVER.SEED $SEED DATA.DATASET $DATASET \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR \
    MODEL.NAME 'eva02_img_extra_token' TEST.MODE True TAG "PRCC-Co-$COLOR-TRAIN" TRAIN_DUMP True \
    GRAD_CAM True AUG.RE_PROB 0.0 AUG.RC_PROB 0.0 AUG.RF_PROB 0.0  
     


# Img + COLOR Train Dump  
IMG_WT='logs/prcc+_Co-9-NW/eva02_img_extra_token_best.pth'
COLOR=9
SEED=1234
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $IMG_WT SOLVER.SEED $SEED DATA.DATASET $DATASET \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR \
    MODEL.NAME 'eva02_img_extra_token' TEST.MODE True TAG "PRCC-Co-$COLOR-TRAIN" TRAIN_DUMP True AUG.RE_PROB 0 
    

# #### COLOR + GFLOP AND parameters
COLOR=5
SEED=1234
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    OUTPUT_DIR $DATASET+"_Co-$COLOR" ANALYSIS_STATS True  
# using soft triplet loss for training
#  Model parameters: 311,874,302
# Computational complexity: 78.13 GMac
# Computational complexity: 156.26 GFlops
# Number of parameters: 311.87 M
#  GFLOP USING `thop` 78.05 MACs(G) '# of Params using thop': 302.77M
# FLOP TOTAL : 84737241266
# 84. 74
# FLOP BY MODULES : {'blocks.21.attn.k_proj': 270532608, 'blocks.10.attn.proj': 270532608, 'blocks.9.attn': 1353988096, 'blocks.15.attn.v_proj': 270532608, 'blocks.19.attn.v_proj': 270532608, 'blocks.2.attn.proj': 270532608, 'blocks.8.attn.norm': 1320960, 'blocks.8.attn.k_proj': 270532608, 'blocks.4.attn.v_proj': 270532608, 'blocks.10.attn.v_proj': 270532608, 'blocks.12.attn.k_proj': 270532608, 'blocks.20.attn.k_proj': 270532608, 'blocks.5.attn': 1353988096, 'blocks.1.attn.q_proj': 270532608, 'blocks.17.attn.norm': 1320960, 'blocks.20.attn.proj': 270532608, 'blocks.7.attn.q_proj': 270532608, 'blocks.11.attn': 1353988096, 'blocks.14.attn.k_proj': 270532608, 'blocks.2.attn': 1353988096, 'blocks.21.attn.v_proj': 270532608, 'blocks.21.attn.proj': 270532608, 'blocks.7.attn.norm': 1320960, 'blocks.13.attn.q_proj': 270532608, 'blocks.14.attn.v_proj': 270532608, 'blocks.3.attn.proj': 270532608, 'blocks.13.attn.proj': 270532608, 'blocks.5.attn.v_proj': 270532608, 'blocks.4.attn.k_proj': 270532608, 'blocks.23.attn.q_proj': 270532608, 'blocks.9.attn.q_proj': 270532608, 'blocks.6.attn': 1353988096, 'blocks.9.attn.v_proj': 270532608, 'blocks.11.attn.q_proj': 270532608, 'blocks.7.attn.v_proj': 270532608, 'blocks.7.attn': 1353988096, 'blocks.6.attn.v_proj': 270532608, 'blocks.19.attn.norm': 1320960, 'blocks.1.attn.norm': 1320960, 'blocks.8.attn.v_proj': 270532608, 'blocks.18.attn.k_proj': 270532608, 'blocks.10.attn': 1353988096, 'blocks.16.attn.norm': 1320960, 'blocks.6.attn.proj': 270532608, 'blocks.2.attn.norm': 1320960, 'blocks.14.attn': 1353988096, 'blocks.22.attn.k_proj': 270532608, 'blocks.17.attn': 1353988096, 'blocks.2.attn.k_proj': 270532608, 'blocks.14.attn.norm': 1320960, 'blocks.3.attn.norm': 1320960, 'blocks.22.attn.q_proj': 270532608, 'blocks.23.attn.proj': 270532608, 'blocks.19.attn.proj': 270532608, 'blocks.22.attn.v_proj': 270532608, 'blocks.1.attn.k_proj': 270532608, 'blocks.6.attn.norm': 1320960, 'blocks.12.attn.norm': 1320960, 'blocks.19.attn': 1353988096, 'blocks.3.attn.v_proj': 270532608, 'blocks.0.attn.q_proj': 270532608, 'blocks.22.attn': 1353988096, 'blocks.3.attn.q_proj': 270532608, 'blocks.2.attn.q_proj': 270532608, 'blocks.10.attn.norm': 1320960, 'blocks.21.attn': 1353988096, 'blocks.15.attn': 1353988096, 'blocks.16.attn.q_proj': 270532608, 'blocks.13.attn.k_proj': 270532608, 'blocks.8.attn.q_proj': 270532608, 'blocks.3.attn': 1353988096, 'blocks.11.attn.proj': 270532608, 'blocks.0.attn.norm': 1320960, 'blocks.20.attn': 1353988096, 'blocks.5.attn.proj': 270532608, 'blocks.9.attn.proj': 270532608, 'blocks.11.attn.k_proj': 270532608, 'blocks.3.attn.k_proj': 270532608, 'blocks.0.attn.proj': 270532608, 'blocks.21.attn.q_proj': 270532608, 'blocks.7.attn.k_proj': 270532608, 'blocks.18.attn': 1353988096, 'blocks.16.attn.k_proj': 270532608, 'blocks.12.attn': 1353988096, 'blocks.17.attn.v_proj': 270532608, 'blocks.5.attn.norm': 1320960, 'blocks.23.attn': 1353988096, 'blocks.14.attn.q_proj': 270532608, 'blocks.2.attn.v_proj': 270532608, 'blocks.15.attn.q_proj': 270532608, 'blocks.16.attn': 1353988096, 'blocks.6.attn.k_proj': 270532608, 'blocks.4.attn.proj': 270532608, 'blocks.18.attn.v_proj': 270532608, 'blocks.15.attn.k_proj': 270532608, 'blocks.15.attn.norm': 1320960, 'blocks.15.attn.proj': 270532608, 'blocks.13.attn': 1353988096, 'blocks.16.attn.v_proj': 270532608, 'blocks.7.attn.proj': 270532608, 'blocks.10.attn.k_proj': 270532608, 'blocks.14.attn.proj': 270532608, 'blocks.12.attn.proj': 270532608, 'blocks.6.attn.q_proj': 270532608, 'blocks.1.attn.proj': 270532608, 'blocks.12.attn.q_proj': 270532608, 'blocks.4.attn.norm': 1320960, 'blocks.17.attn.k_proj': 270532608, 'blocks.1.attn.v_proj': 270532608, 'blocks.16.attn.proj': 270532608, 'blocks.9.attn.k_proj': 270532608, 'blocks.18.attn.q_proj': 270532608, 'blocks.18.attn.proj': 270532608, 'blocks.8.attn': 1353988096, 'blocks.11.attn.norm': 1320960, 'blocks.23.attn.norm': 1320960, 'blocks.13.attn.v_proj': 270532608, 'blocks.5.attn.k_proj': 270532608, 'blocks.20.attn.v_proj': 270532608, 'blocks.10.attn.q_proj': 270532608, 'blocks.8.attn.proj': 270532608, 'blocks.1.attn': 1353988096, 'blocks.23.attn.k_proj': 270532608, 'blocks.17.attn.proj': 270532608, 'blocks.20.attn.q_proj': 270532608, 'blocks.4.attn': 1353988096, 'blocks.22.attn.proj': 270532608, 'blocks.11.attn.v_proj': 270532608, 'blocks.9.attn.norm': 1320960, 'blocks.0.attn.k_proj': 270532608, 'blocks.19.attn.q_proj': 270532608, 'blocks.20.attn.norm': 1320960, 'blocks.17.attn.q_proj': 270532608, 'blocks.21.attn.norm': 1320960, 'blocks.0.attn.v_proj': 270532608, 'blocks.13.attn.norm': 1320960, 'blocks.18.attn.norm': 1320960, 'blocks.0.attn': 1353988096, 'blocks.12.attn.v_proj': 270532608, 'blocks.19.attn.k_proj': 270532608, 'blocks.23.attn.v_proj': 270532608, 'blocks.22.attn.norm': 1320960, 'blocks.5.attn.q_proj': 270532608, 'blocks.4.attn.q_proj': 270532608}
# FLOP BY MODULES : {'blocks.7.attn': 1353988096, 'blocks.10.attn': 1353988096, 'blocks.4.attn': 1353988096, 'blocks.18.attn': 1353988096, 'blocks.23.attn': 1353988096, 'blocks.2.attn': 1353988096, 'blocks.0.attn': 1353988096, 'blocks.14.attn': 1353988096, 'blocks.6.attn': 1353988096, 'blocks.19.attn': 1353988096, 'blocks.8.attn': 1353988096, 'blocks.11.attn': 1353988096, 'blocks.20.attn': 1353988096, 'blocks.17.attn': 1353988096, 'blocks.1.attn': 1353988096, 'blocks.5.attn': 1353988096, 'blocks.21.attn': 1353988096, 'blocks.16.attn': 1353988096, 'blocks.3.attn': 1353988096, 'blocks.9.attn': 1353988096, 'blocks.12.attn': 1353988096, 'blocks.13.attn': 1353988096, 'blocks.22.attn': 1353988096, 'blocks.15.attn': 1353988096}
# 'blocks.1.attn': 1353988096









##### Img + COLOR (Extra Token) [FEED COLORS] + GFLOP & NUM PARAMS
COLOR=5
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token_feed' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    OUTPUT_DIR $DATASET+"_Co-$COLOR-FEED" MODEL.ATT_AS_INPUT True ANALYSIS_STATS True  
#  Model parameters: 311,874,302
# Computational complexity: 78.13 GMac
# Computational complexity: 156.26 GFlops
# Number of parameters: 311.87 M
#  GFLOP USING `thop` 78.05 MACs(G) '# of Params using thop': 302.77M




# #### Clothes Disentanlge
COLOR=-1
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token_CL' \
    TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED ANALYSIS_STATS True  
    # >> outputs/"$DATASET"-CL-UCF2-RUN-$SEED.txt
#  Model parameters: 303,783,298
# Computational complexity: 78.13 GMac
# Computational complexity: 156.26 GFlops
# Number of parameters: 303.78 M

# #### COLOR
PORT=12351
SEED=1245
COLOR=9
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    OUTPUT_DIR $DATASET+"_Co-$COLOR" >> outputs/"$DATASET"-CO-$COLOR-UCF2-RUN-$SEED-FINAL.txt











###############################################################################################
#################### MEVID ####################
mevid=/data/priyank/synthetic/MEVID/
CONFIG=configs/mevid_eva02_l_cloth.yml
wt=logs/MEVID/MEVID_IMG2/eva02_l_cloth_best.pth
DATASET="mevid"
ROOT=$mevid
PORT=12359

casiab=/data/priyank/synthetic/CASIA_B_STAR/


###############################################################################################
################################ # Img EVAL ###################################################
IMG_WT=logs/MEVID/MEVID_IMG2/eva02_l_cloth_best.pth
SEED=1234
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $IMG_WT SOLVER.SEED $SEED TEST.MODE True \
    TAG "MEVID-IMG" 

    
# Img Train
SEED=1234
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT SOLVER.SEED $SEED >> outputs/"$DATASET"_img-RUN-$SEED.txt    

# #### COLOR
SEED=1245
COLOR=39
IMG_WT=logs/MEVID/mevid+_Co-39-1245/ez_eva02_vid_hybrid_extra_best.pth
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    teacher_student.py --eval --no-head --resume --config_file $CONFIG DATA.ROOT $ROOT TRAIN.TRAIN_VIDEO True \
    TEST.WEIGHT $IMG_WT TRAIN.TRAIN_VIDEO True DATA.DATASET $DATASET TRAIN.E2E False \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR TEST.MODE True SOLVER.SEED $SEED 
 



# #### COLOR
SEED=1245
COLOR=39
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    teacher_student.py --multi-node --resume --config_file $CONFIG DATA.ROOT $ROOT TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
    TRAIN.DIR_TEACH1 $ROOT TRAIN.TEACH1_MODEL None TRAIN.TEACH1_LOAD_AS_IMG True TRAIN.TEACH_DATASET_FIX 'color_adv' TRAIN.COLOR_ADV True \
    OUTPUT_DIR $DATASET"_COLOR"-$SEED \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR SOLVER.MAX_EPOCHS 100 SOLVER.SEED $SEED  
    # >> outputs/"$DATASET"_CO-$COLOR-$SEED-UCF.txt




    



     
    






#################### CCVID ####################
ccvid=/data/priyank/synthetic/CCVID/
CONFIG=configs/ccvid_eva02_l_cloth.yml
CONFIG=configs/debug_eva02_l_cloth.yml
wt=logs/CCVID/CCVID_IMG/eva02_l_cloth_best.pth
DATASET="ccvid"
ROOT=$ccvid
PORT=12357

casiab=/data/priyank/synthetic/CASIA_B_STAR/
NLR50_Wt=logs/CASIA_B_STAR/CAL_casiab/NLR50_16_224/best_model.pth.tar

EXT_DATA=ltcc
EXT_DATA_PATH=/data/priyank/synthetic/LTCC/

# Img
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    OUTPUT_DIR $DATASET DATA.DATASET 'ccvid_debug' 

#### vid-ez E2E (w/ pretrained) NoAd + Motion LOSS
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --resume --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt MODEL.MOTION_LOSS True >> outputs/"$DATASET"_4T_NoAd_e2e_pre_ml3.txt


# #### COLOR
SEED=1245
COLOR=9
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    teacher_student.py --resume --config_file $CONFIG DATA.ROOT $ROOT TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
    TRAIN.DIR_TEACH1 $ROOT TRAIN.TEACH1_MODEL None TRAIN.TEACH1_LOAD_AS_IMG True TRAIN.TEACH_DATASET_FIX 'color_adv' TRAIN.COLOR_ADV True \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR SOLVER.MAX_EPOCHS 100 SOLVER.SEED $SEED 
    # >> outputs/"$DATASET"_4NAEPM+CO-$COLOR-100EP-RUN-$SEED-Newton-Final.txt
    
# #### COLOR + Masked Attention 
SEED=1245
COLOR=9
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    teacher_student.py --resume --config_file $CONFIG DATA.ROOT $ROOT TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
    TRAIN.DIR_TEACH1 $ROOT TRAIN.TEACH1_MODEL None TRAIN.TEACH1_LOAD_AS_IMG True TRAIN.TEACH_DATASET_FIX 'color_adv' TRAIN.COLOR_ADV True \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR SOLVER.MAX_EPOCHS 100 SOLVER.SEED $SEED \
    MODEL.MASKED_SEP_ATTN True
    # >> outputs/"$DATASET"_4NAEPM+CO-$COLOR-100EP-RUN-$SEED-Newton-Final.txt
    
    
    

    












# ##### NoAd
# # vid-ez 4 frames 
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG_MEVID DATA.ROOT $ROOT \
#     MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True >> outputs/"$DATASET"_4T_NoAd_e2e.txt
# # vid-ez 8 frames 
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --config_file $CONFIG_MEVID DATA.ROOT $ROOT \
#     MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True DATA.F8 True MODEL.TIM_DIM 8 >> outputs/"$DATASET"_8T_NoAd_e2e.txt
# # vid-ez E2E (w/ pretrained)
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --resume --config_file $CONFIG_MEVID DATA.ROOT $ROOT \
#     MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt  >> outputs/"$DATASET"_4T_NoAd_e2e_pre.txt
# # vid-ez only frozen img model ONLY temporal tokens
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
#     train.py --resume --config_file $CONFIG_MEVID DATA.ROOT $ROOT \
#     MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt TRAIN.E2E False  >> outputs/"$DATASET"_4T_NoAd_pre.txt










rsync -r /data/priyank/synthetic/CCVID/CCVID ucf4:/groups/yrawat/

rsync -r ~/gait-sm-ucf/Models/MADE_VID/MADE_ReID/BRIAR_4NAEP* den:~/gait-sm-ucf/Models/MADE_VID/MADE_ReID/
rsync -r den:/data/moved/pr161305/feature_dump/BRIAR_4NAEP* /data/shared/feature_dump/ 

rsync -r ~/gait-sm-ucf/Models/MADE_VID/MADE_ReID/BRIAR_4NAEPM den:~/gait-sm-ucf/Models/MADE_VID/MADE_ReID/
rsync -r ~/gait-sm-ucf/Models/MADE_VID/MADE_ReID/BRIAR_4NAEP den:~/gait-sm-ucf/Models/MADE_VID/MADE_ReID/


rsync -a ucf0:~/MADE_ReID/outputs/*.txt ~/MADE_ReID/outputs/
rsync -a ucf0:~/MADE_ReID/outputs/*.out ~/MADE_ReID/outputs/

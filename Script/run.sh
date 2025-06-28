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

#################### MEVID ####################
mevid=/data/priyank/synthetic/MEVID/
CONFIG=configs/mevid_eva02_l_cloth.yml
DATASET="mevid"
ROOT=$mevid
PORT=12359


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
    
    
    














# #### COLOR
SEED=1245
COLOR=39
IMG_WT=logs/MEVID/mevid+_Co-39-1245/ez_eva02_vid_hybrid_extra_best.pth
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    teacher_student.py --eval --no-head --resume --config_file $CONFIG DATA.ROOT $ROOT TRAIN.TRAIN_VIDEO True \
    TEST.WEIGHT $IMG_WT TRAIN.TRAIN_VIDEO True DATA.DATASET $DATASET TRAIN.E2E False \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR TEST.MODE True SOLVER.SEED $SEED 
 




    teacher_student.py --multi-node 
    MODEL.MOTION_LOSS True TRAIN.TEACH1 $DATASET TEST.WEIGHT $wt TRAIN.HYBRID True \
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

conda activate bert
cd ~/ICCV-CSCI-Person-ReID/
NUM_GPU=1
PORT=12355
BATCH_SIZE=40
RUN_NO=1
SEED=12345

#################### LTCC ####################
ltcc=/data/priyank/synthetic/LTCC/
CONFIG=configs/ltcc_eva02_l_cloth.yml
DATASET="ltcc"
ROOT=$ltcc

COLOR=44
WT=logs/LTCC/ltcc+_Co-44-1245/eva02_img_extra_token_best.pth
# EVA-attribure.train:  CC:  CMC curve, Rank-1  :49.2%  Rank-5  :62.0%  Rank-10 :66.6%  
# EVA-attribure.train:  CC:  mAP Acc. :25.0%
# EVA-attribure.train:  General:  CMC curve, Rank-1  :80.9%  Rank-5  :89.0%  Rank-10 :91.3%  
# EVA-attribure.train:  General:  mAP Acc. :47.0%



#################### PRCC ####################
prcc=/data/priyank/synthetic/PRCC/prcc/
CONFIG=configs/prcc_eva02_l_cloth.yml
DATASET="prcc"
ROOT=$prcc

COLOR=9
WT=logs/PRCC/prcc-9-1245-16/eva02_img_extra_token_best.pth
# EVA-attribure.train:  CC :  CMC curve, Rank-1  :66.8%  Rank-5  :76.0%  Rank-10 :79.8%  
# EVA-attribure.train:  CC :  mAP Acc. :62.9%
# EVA-attribure.train:  SC:  CMC curve, Rank-1  :100.0%  Rank-5  :100.0%  Rank-10 :100.0%  
# EVA-attribure.train:  SC:  mAP Acc. :98.9%


########## IMAGE EVAL ##########
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $WT SOLVER.SEED $SEED \
    MODEL.NAME 'eva02_img_extra_token' TRAIN.COLOR_PROFILE $COLOR TEST.MODE True 
    






    

#################### CCVID ####################
ROOT=/data/priyank/synthetic/CCVID/
CONFIG=configs/ccvid_eva02_l_cloth.yml
DATASET="ccvid"
CLASS=75 

COLOR=34
WT=logs/CCVID/ccvid-34-1244/ez_eva02_vid_hybrid_extra_best.pth
# EVA-attribure: Computing CMC and mAP only for the same clothes setting
# EVA-attribure: top1:100.0% top5:100.0% top10:100.0% top20:100.0% mAP:100.0%
# EVA-attribure: Computing CMC and mAP only for clothes-changing
# EVA-attribure: top1:87.2% top5:90.8% top10:92.2% top20:93.5% mAP:86.6%

COLOR=9
WT=Dump/ccvid-9-1245/ez_eva02_vid_hybrid_extra_best.pth

COLOR=49
SEED=1245

COLOR=49
WT=Dump/ccvid-49-1245/ez_eva02_vid_hybrid_extra_best.pth





#################### MEVID ####################
ROOT=/data/priyank/synthetic/MEVID/
CONFIG=configs/mevid_eva02_l_cloth.yml
DATASET="mevid"
CLASS=104
COLOR=39
WT=logs/MEVID/mevid+_Co-39-1245/ez_eva02_vid_hybrid_extra_best.pth
# EVA-attribure: Overall Results ---------------------------------------------------
# EVA-attribure: top1:70.3% top5:82.0% top10:84.2% top20:87.3% mAP:46.2%

COLOR=39
WT=logs/MEVID/mevid_COLOR-1245/ez_eva02_vid_hybrid_extra_7.pth
# EVA-attribure: top1:70.6% top5:82.6% top10:83.5% top20:87.7% mAP:48.2%





CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $WT SOLVER.SEED $SEED \
    TRAIN.TRAIN_VIDEO True DATA.DATASET $DATASET \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR TEST.MODE True TRAIN.TEACH1_NUMCLASSES $CLASS
    


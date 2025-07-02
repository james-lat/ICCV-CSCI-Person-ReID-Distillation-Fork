conda activate bert
cd ~/ICCV-CSCI-Person-ReID/
NUM_GPU=1
PORT=12355
RUN_NO=1
SEED=12345

#################### LTCC ####################
ltcc=/data/priyank/synthetic/LTCC/
CONFIG=configs/ltcc_eva02_l_cloth.yml
DATASET="ltcc"
ROOT=$ltcc

COLOR=44
WT=logs/LTCC/ltcc+_Co-44-1245/eva02_img_extra_token_best.pth
# EVA-attribure.train:  CC:  CMC curve, Rank-1  :50.3%  Rank-5  :62.0%  Rank-10 :68.6%  
# EVA-attribure.train:  CC:  mAP Acc. :25.9%
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

COLOR=41
WT=logs/PRCC/prcc+_Co-41-1245/eva02_img_extra_token_best.pth
# EVA-attribure.train:  CC :  CMC curve, Rank-1  :66.5%  Rank-5  :74.7%  Rank-10 :78.2%  
# EVA-attribure.train:  CC :  mAP Acc. :62.3%
# EVA-attribure.train:  SC:  CMC curve, Rank-1  :100.0%  Rank-5  :100.0%  Rank-10 :100.0%  
# EVA-attribure.train:  SC:  mAP Acc. :99.4%

########## IMAGE CSCI EVAL  ##########
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $WT SOLVER.SEED $SEED \
    MODEL.NAME 'eva02_img_extra_token' TRAIN.COLOR_PROFILE $COLOR TEST.MODE True 
    




#################### CCVID ####################
ROOT=/data/priyank/synthetic/CCVID/
CONFIG=configs/ccvid_eva02_l_cloth.yml
DATASET="ccvid"

WT=logs/CCVID/CCVID_IMG/eva02_l_cloth_best.pth
# EVA-attribure: Computing CMC and mAP
# EVA-attribure: top1:85.7% top5:88.8% top10:89.9% top20:90.8% mAP:79.1%
# EVA-attribure: Computing CMC and mAP only for clothes-changing
# EVA-attribure: top1:84.4% top5:87.8% top10:89.0% top20:90.1% mAP:77.4%


#################### MEVID ####################
ROOT=/data/priyank/synthetic/MEVID/
CONFIG=configs/mevid_eva02_l_cloth.yml
DATASET="mevid"

WT=logs/MEVID/MEVID_IMG2/eva02_l_cloth_best.pth


########## IMAGE EVAL (BASELINE NO COLORS) ##########
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $WT SOLVER.SEED $SEED \
    TEST.MODE True 




    

#################### CCVID ####################
ROOT=/data/priyank/synthetic/CCVID/
CONFIG=configs/ccvid_eva02_l_cloth.yml
DATASET="ccvid"

COLOR=49
WT=logs/CCVID/ccvid-49-1245/ez_eva02_vid_hybrid_extra_best.pth
# EVA-attribure: Computing CMC and mAP only for the same clothes setting
# EVA-attribure: top1:100.0% top5:100.0% top10:100.0% top20:100.0% mAP:100.0%
# EVA-attribure: Computing CMC and mAP only for clothes-changing
# EVA-attribure: top1:91.0% top5:91.7% top10:93.4% top20:94.0% mAP:90.9%


#################### MEVID ####################
ROOT=/data/priyank/synthetic/MEVID/
CONFIG=configs/mevid_eva02_l_cloth.yml
DATASET="mevid"




COLOR=39
WT=logs/MEVID/mevid+_Co-39-1245/ez_eva02_vid_hybrid_extra_best.pth
# EVA-attribure: Overall Results ---------------------------------------------------
# EVA-attribure: top1:70.3% top5:82.0% top10:84.2% top20:87.3% mAP:46.2%

COLOR=39
WT=logs/MEVID/mevid_COLOR-1245/ez_eva02_vid_hybrid_extra_7.pth
# EVA-attribure: top1:70.6% top5:82.6% top10:83.5% top20:87.7% mAP:48.2%

COLOR=2
WT=Dump/mevid-2-1245/ez_eva02_vid_hybrid_extra_best.pth



NUM_GPU=2
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train_two_step.py --eval --resume --config_file $CONFIG DATA.ROOT $ROOT TEST.WEIGHT $WT SOLVER.SEED $SEED \
    TRAIN.TRAIN_VIDEO True DATA.DATASET $DATASET  \
    MODEL.NAME 'ez_eva02_vid_hybrid_extra' TRAIN.COLOR_PROFILE $COLOR                   




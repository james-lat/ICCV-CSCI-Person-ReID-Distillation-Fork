# ICCV-CSCI-Person-ReID

Implementation of Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement


[Setup](setup.md) Instructions givene here.

## Results
While paper reports average of 2 two runs, we have give best performing model weights, so accuracy will be higher than reported in paper.

| Dataset | COLOR PROFILE | CC (R1) | CC (mAP) | General / SC (R1) | General / SC (mAP) | MODEL Wt. | 
| -------- | ------- | ------- |  ------- | ------- | ------- | ------- |
| LTCC | 44 | 48.2 | 24.4 | 83.4 | 47.8 | - 
| PRCC | 9 | 66.8 | 62.9 | 100.0 | 98.9 | -
| MEVID | - | - | - | - | - | -
| CCVID | - | - | - | - | - | -



## How to run 

We pass color as input as color profile integer with `COLOR >= 50` indicates RGB HISTOGRAM and `COLOR < 50` indicate RGB-uv histogram. Behind the scenes these numbers are translated in various representations. 

#### Train 
Training code is available in `Script/image_batch.sh` and other ablations are available in `Script/run.sh`

Each dataset has some really good performing color profiles Highligted in Logs `outputs/Color Results`.

#### Test 
Evaluating the pretrained model weights is given in `Script/test.sh`


## CITE


## CREDITS 

- Code taken from [MADE-ReID](https://github.com/moon-wh/MADE) 
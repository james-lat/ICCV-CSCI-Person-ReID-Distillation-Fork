# ICCV-CSCI-Person-ReID

Implementation of **Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement** [Paper]() || [Project Page]() || [Arxiv]( )


[Setup](setup.md) instructions given here.

## Results
While paper reports average of 2 two runs, we have give best performing model weights, so accuracy will be higher than reported in paper.

| Dataset | COLOR PROFILE | CC (R1) | CC (mAP) | General / SC (R1) | General / SC (mAP) | MODEL Wt. | 
| -------- | ------- | ------- |  ------- | ------- | ------- | ------- |
| LTCC | 44 | 49.2 | 25.0 | 80.9 | 47.0 | - 
| PRCC | 41 | 66.4 | 60.7 | 100.0 | 98.6 | -
| MEVID | - | - | - | - | - | -
| CCVID | 49 | 91.0 | 90.9 | 100.0 | 100.0 | -



## How to run 

We pass color as input as color profile integer with `COLOR >= 50` (50 51 52 53 54 55 56) indicating RGB HISTOGRAM and `COLOR < 50` indicate RGB-uv histogram. These profiles vary in hyerparameters, and behind the scenes these numbers are translated in various implementations. 

#### CODE 
Training code is available in [Image Train](Script/image_batch.sh) and [Video Train](Script/image_batch_newton.sh) and other ablations are available in [Ablations](Script/run.sh). The best performance from RGB-uv color profile is chosen (> averaged, as reported in paper).    
For Video model, train vanilla image model on video random frames as normal person ReID (no colors) and use those weights in EZ-CLIP to train the final video model.  


Evaluating the pretrained model weights is given in [Test](Script/test.sh)


## CITE


## CREDITS 

- Code taken from [MADE-ReID](https://github.com/moon-wh/MADE) 
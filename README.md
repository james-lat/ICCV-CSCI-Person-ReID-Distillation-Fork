# Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement

Implementation of **Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement** [Paper]() || [Project Page]() || [Arxiv]( )


[Setup](setup.md) instructions given here.

## Results
While paper reports average of 2 two runs, we have give best performing model weights, so accuracy will be higher than reported in paper.

| Dataset | COLOR PROFILE | CC (R1) | CC (mAP) | General / SC (R1) | General / SC (mAP) | MODEL Wt. | 
| -------- | ------- | ------- |  ------- | ------- | ------- | ------- |
| LTCC | 44 | 50.3 | 25.9 | 85.2 | 49.6 | [ckpt](https://github.com/ppriyank/ICCV-CSCI-Person-ReID/releases/download/untagged-48f268abb8ed6a10fba1/CSCI_LTCC.zip) 
| PRCC | 41 | 66.5 | 62.3 | 100.0 | 99.4 | [ckpt-1](https://github.com/ppriyank/ICCV-CSCI-Person-ReID/releases/download/untagged-48f268abb8ed6a10fba1/CSCI_PRCC.zip) & [ckpt-2](https://github.com/ppriyank/ICCV-CSCI-Person-ReID/releases/download/untagged-48f268abb8ed6a10fba1/CSCI_PRCC_2.zip)
| MEVID | - | - | - | - | - | [image-ckpt](https://github.com/ppriyank/ICCV-CSCI-Person-ReID/releases/download/untagged-1dbca424a1a3b1374875/CSCI_MEVID_IMG.zip) & [video-ckpt]()
| CCVID | 49 | 91.0 | 90.9 | 100.0 | 100.0 | [image-ckpt](https://github.com/ppriyank/ICCV-CSCI-Person-ReID/releases/download/untagged-1dbca424a1a3b1374875/CSCI_CCVID_IMG.zip) & [video-ckpt]()



## How to run 

We pass color as input as color profile integer with `COLOR >= 50` (50 51 52 53 54 55 56) indicating RGB HISTOGRAM and `COLOR < 50` indicate RGB-uv histogram. These profiles vary in hyerparameters, and behind the scenes these numbers are translated in various implementations. 

#### CODE 
Training code is available in [All-Train](Script/all_train.sh), [Video-Train](Script/image_batch_newton.sh)
 and other ablations are available in [Ablations](Script/run.sh). The best performance from RGB-uv color profile is chosen (> averaged, as reported in paper).    
For Video model, **FIRST** train vanilla image model on video random frames as normal person ReID (no colors) and then use those weights in EZ-CLIP to train the final video model.  We have provided the image weights above.


Evaluating the pretrained model weights is given in [Test](Script/test.sh)


## CITE


## CREDITS 

- Code taken from [MADE-ReID](https://github.com/moon-wh/MADE) 
#!/bin/bash


## change the list of parameters below to specify your own paths
## expects numpy images to exist in shape: X,Y,Z

MODELPATH="path to saved models"
INPUTDIR="path to test-set images"
LABELDIR="path to ground-truth labels for test images"
OUTPUTDIR="path to save predictions"
WUDIR="path to WU-model predictions"

mkdir ${OUTPUTFILE}

flipflag=0
gpuuse=0  # which gpu card to use

#arch=vnet_small
#arch=vnet_small_res
arch=unet_3d_res1


python ../predict.py\
        -f $INPUTDIR \
        --output-path $OUTPUTDIR \
        --model-path $MODELPATH \
        --model-arch $arch \
        --label-path $LABELDIR \
        --wu-pred $WUDIR \
        --flip-flag    $flipflag \
        --gpu-use         $gpuuse 
 


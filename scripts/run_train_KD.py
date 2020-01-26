#!/bin/bash

#run training


## change the list of parameters below to specify your own paths

INPUTDIR="/home/ubuntu/disk2/DATA/ALL/"
VALDIR="/home/ubuntu/disk2/DATA/ALL/"

MODELPATH="/home/ubuntu/disk2/MODELS/MODELS_BAN/MODELS/"
TEACHERPATH="/home/ubuntu/disk2/MODELS/unet_3D_Res1.pkl"  ## pretrained Teacher

learning_rate=0.0005
epochs=250   
arch=unet_3d_res1 # or vnet_modified or unet_3d_res1
batch=2  # set this to num_gpu_cards_being_used in training   

## Python script starts here
python ../trainKD.py \
        -t $INPUTDIR \
        --model-name $MODELPATH \
        --learning-rate $learning_rate \
        --epoch-count    $epochs \
        --batch-size    $batch \
        --model-teacher    $TEACHERPATH \
        --model-arch   $arch 


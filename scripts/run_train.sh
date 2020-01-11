#!/bin/bash

#run training


## change the list of parameters below to specify your own paths

INPUTDIR="set the path to the training data dir"
VALDIR="set path to val-dir, this is optional, current train code does not support this"

MODELPATH="set path to where model will be saved"

learning_rate=0.0005
epochs=250   
arch=vnet # or vnet_modified or unet_3d_res1
batch=16  # set this to num_gpu_cards_being_used in training   

## Python script starts here
python ../train.py \
        -t $INPUTDIR \
        --model-name $MODELPATH \
        --learning-rate $learning_rate \
        --epoch-count    $epochs \
        --batch-size    $batch \
        --model-arch   $arch 


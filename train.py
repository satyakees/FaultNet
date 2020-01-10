# Train
# TODO: Add logger


import os, sys
sys.path.append('../')
from collections import defaultdict
from copy import deepcopy
import argparse
import logging
import datetime
import time
import numpy as np
import json
import csv
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from loaders import dataloader 
from .loss import dice_random_weight, cross_entropy_3D, CE_plus_Dice
from models import load_models

def compute_iou(preds, trues):

    b,c,_,_,_ = preds.size()
    if c==2:
       preds = np.argmax(preds, axis=1)
    preds = preds.reshape(b,-1)
    trues = trues.reshape(b,-1)
    preds = preds>0.5
    trues = trues>0.5
    inter = trues & preds
    union = trues | preds
    iou = inter.sum(1)/(union.sum(1)+1e-8)
    return iou

def check_folder(data_path):
    """
      helper function to check all training data paths are correct
    """

    # Check if all images and labels exist
    if not os.path.exists(data_path):
        error_message = 'Folder ' + data_path + ' not found'
        raise OSError(error_message)

    image_path = os.path.join(data_path, 'images')
    label_path = os.path.join(data_path, 'labels')

    if not os.path.exists(image_path):
        error_message = 'Folder ' + image_path + ' does not exist'
        raise OSError(error_message)

    if not os.path.exists(label_path):
        error_message = 'Folder ' + label_path + ' does not exist'
        raise OSError(error_message)

    # Build a list of all images and labels
    image_list = []
    label_list = []

    for image in os.listdir(image_path):
        if image.endswith('.jpg'):
            image_list.append(image.split('.')[0])

    for label in os.listdir(label_path):
        if label.endswith('.png'):
            label_list.append(label.split('.')[0])

    image_list.sort()
    label_list.sort()

    if not image_list == label_list:
        file_list = list(set(image_list) & set(label_list))
    else:
        file_list = image_list

    return file_list


def main(args):

    training_data_path = args.training_path
    model_name = args.model_name

    model_arch = args.model_arch
    batch_size = args.batch_size
    nepoch = args.epoch_count
    initial_lr = args.learning_rate

    output_channels = 2

    print(" ---model will be written at %s ---  "%(model_name))
    print("----- model arch is %s -----  "%(model_arch))
    print("----- output_channels %d "%(output_channels))

    file_list_train = check_folder(training_data_path)


    training_data_loader, validation_data_loader = dataloader.LoadData(training_data_path, file_list_train, split=0.001, \
                                                                             batch_size=batch_size, transforms=None)

    parallel_flag=True
    gpu_flag=True
    model = load_models.getModel(model_arch=model_arch,output_channels=output_channels,parallel_flag=parallel_flag)
    use_validation_set = False # toggle to True and change split if needed

    
    print("-----training loader  length is ---------  ",len(training_data_loader))
    print("----- val loader length is -------------  ",len(validation_data_loader))


    step_decay = 30
    decay_factor = 0.5
    optimizer = torch.optim.Adam(model.parameters(),lr=initial_lr, eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_decay, decay_factor) ## decay every 30 epochs by half

    LossFunction = cross_entropy_3D  # default loss, change if needed

    best_train_loss = 100000.; best_train_iou = -100.
    best_val_loss = 100000.; best_val_iou = -100.
    for epoch in range(nepoch):

        train_loss = 0. 
        validation_loss = 0.
        train_accuracy = []
        val_accuracy = []
        model.train()
        for i, (images, labels) in enumerate(training_data_loader):
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                pred_dict = model(images.float().cuda())
                loss = LossFunction(pred_dict, labels.long().cuda(),1)
                loss.backward()
                self.optimizer.step()

            train_loss +=loss.item()
            train_accuracy.append(compute_iou(output_dict['probs'].detach().cpu().numpy(), \
                                              labels.detach().cpu().numpy())) 

        train_loss = train_loss/len(training_data_loader)
        train_iou_mean = np.concatenate(train_accuracy).mean()

        if use_validation_set==False:
            if train_loss < best_train_loss :
                best_train_loss = train_loss
                best_train_iou = train_iou_mean
                save_state = {'epoch': epoch+1,
                              'model_state': self.model.state_dict(),
                              'optimizer_state': self.optimizer.state_dict()}
                modelfile = model_arch + '_'+ str(epoch+1) + '.pkl'
                torch.save(save_state, os.path.join(output_path,modelfile)
        else:
            model.eval()
            for i, (images, labels) in enumerate(validation_data_loader):
                with torch.no_grad():
                    pred_dict = model(images.float().cuda())
                    loss = LossFunction(pred_dict, labels.long().cuda(),1)
                    validation_loss +=loss.cpu().numpy()
                    val_accuracy.append(compute_iou(output_dict['probs'].detach().cpu().numpy(), \
                                                    labels.detach().cpu().numpy()))
          
            validation_loss = validation_loss/len(validation_data_loader)
            val_iou_mean = np.concatenate(val_accuracy).mean()

            if validation_loss < best_val_loss and val_iou_mean < best_val_iou:
                best_val_loss = validation_loss
                best_val_iou =  val_iou_mean
                save_state = {'epoch': epoch+1,
                              'model_state': self.model.state_dict(),
                              'optimizer_state': self.optimizer.state_dict()}
                modelfile = model_arch + '_'+ str(epoch+1) + '.pkl'
                torch.save(save_state, os.path.join(output_path,modelfile)

        lr_scheduler.step()
        print(" at epoch %d train loss %f iou %f"%(epoch, train_loss, train_iou_mean))
 
if __name__=="__main__":
    help_string = "PyTorch Fault picking Model Training"

    parser = argparse.ArgumentParser(description=help_string)

    parser.add_argument('-t', '--training-path', type=str, metavar='DIR', help='Path where training data exists', required=True)
    parser.add_argument('-vl', '--validation-path', type=str, metavar='DIR', help='Path where validation data exists', required=False)  #TODO
    parser.add_argument('-n', '--model-name', type=str, metavar='DIR', help='Name of model to be stored', required=True)

    parser.add_argument('-bs', '--batch-size', type=int, metavar='N', help='Batch size', default=8, required=True)
    parser.add_argument('--epoch-count', type=int, metavar='N', help='Number of epochs (default: 100)', default=100, required=True)
    parser.add_argument('-lr', '--learning-rate', type=float, metavar='LR', help='Initial learning rate (default: 0.0005)', default=0.0005, required=True)
    parser.add_argument('--validation-split', type=float, metavar='VAL', help='Training-validation split (default: 0.15)', default=0.15, required=False)

    parser.add_argument('-arch', '--model-arch', type=str, metavar='ARCH', help='Architecture of the model ', default='vnet', required=True)

    args = parser.parse_args()

    main(args)
    sys.exit(1)

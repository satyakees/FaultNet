import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

def dice_random_weight(inputs, target, random_wt_flag,num_channels=2):
    """
      dynamically class weighted dice score (fscore = tp/(tp+fp))
      usually worse than cross-entropy, if used alone
      should be used in conjunction with CE
      assumes inputs is dictionary with key to logits :'probs'
    """
    if num_channels==1:
        inputs = F.sigmoid(inputs['probs'])
    else:
        inputs = F.softmax(inputs['probs'],dim=1)
        inputs,_ =torch.max(inputs,1)

    smooth = 1
    if random_wt_flag==1:
        beta = random.choice(np.arange(1,2,0.1))
    else:
        beta =1 .

    inp_flat = inputs.contiguous().float().view(-1)
    tar_flat = target.contiguous().float().view(-1)

    intersection = (inp_flat*tar_flat).sum()

    fscore = (((1 + beta ** 2) * intersection) + smooth) / (((beta ** 2 * inp_flat.sum()) + tar_flat.sum()) + smooth)
    return 1-fscore

def cross_entropy_3D(input, target, flagmain, random_weight_flag=False, weight=None, size_average=True):

    """
      class weights degrade the results (thicker faults) in my experiments
    """

    if flagmain==1:
        input_tensor = input['logits']
    else:
        input_tensor = input

    n, c, h, w, s = input_tensor.size()
    log_p = F.log_softmax(input_tensor, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())

    if random_wt_flag==True:
        class_weights = [1.0 for _ in range(c)]
        class_weights[1] = random.choice(np.arange(1,2,0.1))
        class_weights = torch.FloatTensor(my_weights).cuda()
        loss = F.nll_loss(log_p, target, weight=class_weights, size_average=False)
    else:
        loss = F.nll_loss(log_p, target, size_average=False)

    if size_average:
        loss /= float(target.numel())
    return loss

def CE_plus_Dice(input, target, flagmain, random_weight_flag=True, num_channels=2, weight=None, size_average=True):

    """
      CE+Dice loss
    """

    loss_ce = cross_entropy_3D(input, target,flagmain)
    loss_dice = dice_random_weight(input, target, random_weight_flag,num_channels)

    return loss_ce + loss_dice


 


########################

class LossFunction():
    """
    Class to select loss function
    """
    @classmethod
    def select(cls, loss_type):
        if loss_type == 'bce_dice':
            return bce_dice_loss
        if loss_type == 'bce_dice_kl_l2':
            return bce_dice_kl_l2
        if loss_type == 'lovasz':
            return lovasz_hinge
        if loss_type == 'crossentropy3d':
            return cross_entropy_3D
        if loss_type == 'dicerandomwt':
            return dice_random_weight

if __name__ == '__main__':

    loss = bce_dice_loss(torch.FloatTensor(np.array([1,0,1,0,1,0])), torch.FloatTensor(np.array([0,0,1,1,1,0])))
    print(loss)

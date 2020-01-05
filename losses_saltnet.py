import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

"""
 losses for fault 3d
 cross_entropy_3d is preferable
"""

def dice_loss(predicted, actual, weight=None, is_average=True):
    """
        pure dice loss
    """
    num = predicted.size(0)
    predicted = predicted.view(num, -1)
    actual = actual.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        predicted = predicted * w
        actual = actual * w
    intersection = (predicted * actual).sum(1)
    scores = 2. * (intersection + 1) / (predicted.sum(1) + actual.sum(1) + 1)

    if is_average:
        score = scores.sum()/num
        return (1 - torch.clamp(score, 0., 1.))
    else:
        return (1 - scores)



def dice_random_weight(inputs, target, flagmain,random_wt_flag=1):
    """
      dynamically class weighted dice score (fscore = tp/(tp+fp))
      worse than cross-entropy
    """
    if flagmain==1:
        inputs = F.sigmoid(inputs['probs'])
    else:
        inputs = F.sigmoid(inputs)
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
      cross-entropy out performs dice
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
        class_weights[0] = random.choice(np.arange(1,2,0.1))
        class_weights = torch.FloatTensor(my_weights).cuda()
        loss = F.nll_loss(log_p, target, weight=class_weights, size_average=False)
    else:
        loss = F.nll_loss(log_p, target, size_average=False)

    if size_average:
        loss /= float(target.numel())
    return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        
        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

class One_Hot(nn.Module):

    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()



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

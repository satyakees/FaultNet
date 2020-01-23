import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np

###############################################################################
# Collection of Helper Functions 
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class Unetconv_norm_lrelu(nn.Module):
    def __init__(self, feat_in, feat_out, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1), bias=False):
        super(Unetconv_norm_lrelu, self).__init__()
        self.conv_norm_lrelu = nn.Sequential(nn.Conv3d(feat_in, feat_out, kernel_size, init_stride, padding_size, bias=False),
                                             nn.InstanceNorm3d(feat_out),
                                             nn.LeakyReLU(inplace=True),)
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self,inputs):
        outputs = self.conv_norm_lrelu(inputs)
        return outputs

class Unetnorm_lrelu_conv(nn.Module):
    def __init__(self, feat_in, feat_out, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1), bias=False):
        super(Unetnorm_lrelu_conv, self).__init__()
        self.norm_lrelu_conv = nn.Sequential(nn.InstanceNorm3d(feat_in),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Conv3d(feat_in, feat_out, kernel_size, init_stride, padding_size, bias=False),)

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self,inputs):
        outputs = self.norm_lrelu_conv(inputs)
        return outputs

class Unetlrelu_conv(nn.Module):
    def __init__(self, feat_in, feat_out, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1), bias=False):
        super(Unetlrelu_conv, self).__init__()
        self.lrelu_conv = nn.Sequential(nn.LeakyReLU(inplace=True),
                                        nn.Conv3d(feat_in, feat_out, kernel_size, init_stride, padding_size, bias=False),)

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self,inputs):
        outputs = self.lrelu_conv(inputs)
        return outputs

class Unetnorm_lrelu_upscale_conv_norm_lrelu(nn.Module):
    def __init__(self, feat_in, feat_out, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1), bias=False):
        super(Unetnorm_lrelu_upscale_conv_norm_lrelu, self).__init__()
        self.norm_lrelu_upscale_conv_norm_lrelu = nn.Sequential(nn.InstanceNorm3d(feat_in),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Upsample(scale_factor=2, mode='nearest'),
                                             nn.Conv3d(feat_in, feat_out, kernel_size, init_stride, padding_size, bias=False),
                                             nn.InstanceNorm3d(feat_out),
                                             nn.LeakyReLU(inplace=True),)

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self,inputs):
        outputs = self.norm_lrelu_upscale_conv_norm_lrelu(inputs)
        return outputs


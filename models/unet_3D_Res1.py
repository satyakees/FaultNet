import torch
import math
import torch.nn as nn
from .networks_other import init_weights

## pytorch implementation of unet-3d with Residual Blocks 
## No batchnorm, always uses InstanceNorm3d as batch-sizes are small
## LeakyReLU slightly better than ReLU
## initialization via He
## To train models from scratch uncomment all bias=False

class BasicRes(nn.Module):
    """ 
    layout of Residual block (+): Residual connection:
     C ------(+)-IN R    
     |        |
     |        |
     IN R C - -
     
    """ 
    def __init__(self, insize, outsize, kernel=3, init_stride=1, pad=1,last_layer=False):
        super().__init__()

        self.last_layer = last_layer
        self.conv1 = nn.Conv3d(insize,outsize,kernel_size=kernel, stride=init_stride, padding=pad) #,bias=False)

        self.norm_relu_conv = nn.Sequential(nn.InstanceNorm3d(outsize),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv3d(outsize,outsize,kernel_size=kernel,stride=1,padding=1),)

        self.dropout = nn.Dropout3d(p=0.6)
        self.inorm = nn.InstanceNorm3d(outsize)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.last_layer_op = nn.Sequential(nn.InstanceNorm3d(outsize),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Upsample(scale_factor=2, mode='nearest'),
                                           nn.Conv3d(outsize, outsize, kernel_size=3, stride=1, padding=1), #,bias=False),
                                           nn.InstanceNorm3d(outsize),
                                           nn.LeakyReLU(inplace=True),)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')

    def forward(self,x):
        out = self.conv1(x)
        res = out
        out = self.norm_relu_conv(out)
        out = self.dropout(out)
        out +=res
        if self.last_layer==False:
            out = self.lrelu(self.inorm(out))
            return out
        else:
            out = self.last_layer_op(out)
            return out


class BasicDecoder(nn.Module):
    def __init__(self, insize, outsize, kernel=3, init_stride=1, pad=1, last_layer=False):
        super().__init__()
        
        self.last_layer = last_layer
        self.baseconv = nn.Sequential(nn.Conv3d(insize, insize, kernel_size=kernel, stride=init_stride, padding=pad), #,bias=False),
                                      nn.InstanceNorm3d(insize),
                                      nn.LeakyReLU(inplace=True),)

        self.conv1x1 = nn.Conv3d(insize, outsize,kernel_size=1, stride=1, padding=0) #,bias=False)

        self.upconv = nn.Sequential(nn.InstanceNorm3d(outsize),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Upsample(scale_factor=2, mode='nearest'),
                                           nn.Conv3d(outsize, outsize, kernel_size=3, stride=1, padding=1),#bias=False),
                                           nn.InstanceNorm3d(outsize),
                                           nn.LeakyReLU(inplace=True),)

        self.upconv_lastlayer = nn.Sequential(nn.Conv3d(insize, outsize, kernel_size=kernel, stride=init_stride, padding=pad), #,bias=False),
                                               nn.InstanceNorm3d(outsize),
                                               nn.LeakyReLU(inplace=True),)

        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')

    def forward(self,x):
        if self.last_layer==False:
            out = self.baseconv(x)
            out = self.conv1x1(out)
            out = self.upconv(out)
            return out
        else:
            return self.upconv_lastlayer(x)



class unet_3D_Res1(nn.Module):

    def __init__(self, n_classes=2, in_channels=1, verbose=False):
        super(unet_3D_Res1, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.feature_scale = 1

        filters = [16, 32, 64, 256] 
       
        if verbose:
            print("-------------------------")
            for x in range(len(filters)):
                print("--- check filter lens ---",filters[x])
            print("--- num classes to output  -----", self.n_classes)
            print("--- in channels is -----", self.in_channels)
            print("-------------------------")

        # downsampling

        self.dropout = nn.Dropout3d(p=0.6)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.conv0_1 = nn.Conv3d(self.in_channels, filters[0],kernel_size=3,stride=1,padding=1)
        self.conv0_2 = nn.Conv3d(filters[0], filters[0],kernel_size=3,stride=1,padding=1)
        self.conv0_3 = nn.Sequential(nn.InstanceNorm3d(filters[0]),
                                     nn.LeakyReLU(inplace=True),
                                      nn.Conv3d(filters[0], filters[0],kernel_size=3,stride=1,padding=1),)
        self.inorm0 = nn.InstanceNorm3d(filters[0])


        self.conv1 = BasicRes(filters[0], filters[1], kernel=3, init_stride=2)
        self.conv2 = BasicRes(filters[1], filters[2], kernel=3, init_stride=2)
        self.conv3 = BasicRes(filters[2], filters[3], kernel=3, init_stride=2, pad=1, last_layer=True)

        ## upsampling
        self.conv_up1 = BasicDecoder(filters[3]+filters[2],filters[2])
        self.conv_up2 = BasicDecoder(filters[2]+filters[1],filters[1])
        self.conv_up3 = BasicDecoder(filters[1]+filters[0],filters[0], init_stride=1, pad=1, last_layer=True)

        ## logits output
        self.final1 = nn.Conv3d(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        # Downsample

        out = self.conv0_1(x)
        res = out
        out = self.lrelu(out)
        out = self.conv0_2(out)
        out = self.dropout(out)
        out = self.conv0_3(out)
        out +=res
        out = self.inorm0(out)
        conv0 = self.lrelu(out)   ## 1, 16, 128, 128, 128

        conv1 = self.conv1(conv0)   ## 1, 32, 64, 64, 64
        conv2 = self.conv2(conv1)   ## 1, 64, 32, 32, 32
        conv3 = self.conv3(conv2)   ## 1, 256, 32, 32, 32

        # Upsample
        out = torch.cat((conv3,conv2),dim=1)
        up1 = self.conv_up1(out)        # 1, 64, 64, 64

        out = torch.cat((up1,conv1),dim=1)
        up2 = self.conv_up2(out)          # 1, 32, 128, 128, 128

        out = torch.cat((up2,conv0),dim=1)
        up3 = self.conv_up3(out)       #  1, 16, 128, 128, 128

        final1 = self.final1(up3)  # 1,2, 128, 128, 128

        return {'logits':final1}   
















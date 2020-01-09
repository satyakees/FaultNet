import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks_other import init_weights

def passthrough(x, **kwargs):
    return x


class InputTransition(nn.Module):
    """
      modified compared to original vnet
      res connection added differently
    """
    def __init__(self, outChans):
        super(InputTransition, self).__init__()

        self.dropout = nn.Dropout3d(p=0.6)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv3d(1, outChans, kernel_size=3, stride=1,padding=1, bias=False)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=3, stride=1,padding=1, bias=False)
        self.conv3 = nn.Sequential(nn.InstanceNorm3d(outChans),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv3d(outChans, outChans,kernel_size=3,stride=1,padding=1),)
        self.bn = nn.InstanceNorm3d(outChans)

        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        out = self.conv1(x)
        res = out
        out = self.lrelu(out)
        out = self.dropout(self.conv2(out))
        out = self.conv3(out)
        out +=res
        out = self.lrelu(self.bn(out))

        return out
        
class DownTransition(nn.Module):
    """
      modified compared to original vnet
      conv block orders similar to Resnet layout
    """
    def __init__(self, inChans, dropout=True, last_layer=False):
        super(DownTransition, self).__init__()

        self.last_layer = last_layer
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=3, stride=2, padding=1,bias=False)
        self.down_conv2 = nn.Conv3d(outChans, outChans, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn1 = nn.InstanceNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = nn.LeakyReLU(inplace=True) 

        self.dropout = nn.Dropout3d(p=0.6)
        self.last_layer_op = nn.Sequential(nn.InstanceNorm3d(outChans),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Upsample(scale_factor=2, mode='nearest'),
                                           nn.Conv3d(outChans, outChans, kernel_size=3, stride=1, padding=1,bias=False),
                                           nn.InstanceNorm3d(outChans),
                                           nn.LeakyReLU(inplace=True),)

        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        down = self.down_conv(x)
        res = down
        down = self.down_conv2(self.relu1(self.bn1(down)))
        down = self.dropout(down)
        down = self.relu1(self.bn1(self.down_conv2(down)))
        down = self.down_conv2(down)
        down +=res
        if self.last_layer==False:
            down = self.relu1(self.bn1(down))
            return down
        else:
            down = self.last_layer_op(down)
            return down


class BasicDecoder(nn.Module):
    def __init__(self, insize, outsize, kernel=3, init_stride=1, pad=1, last_layer=False):
        super().__init__()

        self.last_layer = last_layer
        self.baseconv = nn.Sequential(nn.Conv3d(insize, insize, kernel_size=kernel, stride=init_stride, padding=pad,bias=False),
                                      nn.InstanceNorm3d(insize),
                                      nn.LeakyReLU(inplace=True),)

        self.conv1x1 = nn.Conv3d(insize, outsize,kernel_size=1, stride=1, padding=0,bias=False)

        self.upconv = nn.Sequential(nn.InstanceNorm3d(outsize),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Upsample(scale_factor=2, mode='nearest'),
                                           nn.Conv3d(outsize, outsize, kernel_size=3, stride=1, padding=1,bias=False),
                                           nn.InstanceNorm3d(outsize),
                                           nn.LeakyReLU(inplace=True),)

        self.upconv_lastlayer = nn.Sequential(nn.Conv3d(insize, outsize, kernel_size=kernel, stride=init_stride, padding=pad, bias=False),
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


class VNet_modified(nn.Module):
    def __init__(self, elu=True, nll=False,n_classes=2):
        super(VNet_modified, self).__init__()

        filters = [16,32,64,128,256]
        self.n_classes = n_classes

        #changed downsampling compared to VNet original implementation
        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16)
        self.down_tr64 = DownTransition(32)
        self.down_tr128 = DownTransition(64, dropout=True,last_layer=False)
        self.down_tr256 = DownTransition(128, dropout=True, last_layer=True)
       
        #changed decoder compared to VNet original implementation
        self.up_tr256 = BasicDecoder(filters[4]+filters[3], filters[3])
        self.up_tr128 = BasicDecoder(filters[3]+filters[2], filters[2])
        self.up_tr64 = BasicDecoder(filters[2]+filters[1], filters[1])
        self.up_tr32 = BasicDecoder(filters[1]+filters[0], filters[0], init_stride=1, pad=1, last_layer=True)

        self.final1 = nn.Conv3d(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)

        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):

        # Downsample
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        # Upsample
        out = torch.cat((out256,out128), dim=1)
        out = self.up_tr256(out) 
        out = torch.cat((out, out64),dim=1)
        out = self.up_tr128(out)
        out = torch.cat((out, out32),dim=1)
        out = self.up_tr64(out)
        out = torch.cat((out, out16),dim=1)
        out = self.up_tr32(out)

        out = self.final1(out)

        #print("--- final out shape --", out.shape)
        return {'probs':out}

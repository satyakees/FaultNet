###  PyTorch implementation of Fault picking with 3D-UNets .


#### Model1: Basic 3D model with simple residual connection ( unet_3d_res1 in models dir). The residual block is set up as:

Conv1 ---------------(+) --IN --LReLU  
|			|  
|			|   
IN -- LReLU ---Conv2 --      

IN: instancenorm3d (Used IN throughout due to training with small batch size, each card gets only 1 image).  
LReLU: LeakyReLU  (Observed slight improvement over ReLU)  
Conv1: (3X3) conv3d with  stride=2  
Conv2: (3x3) conv3d with stride=1  
(+): shortcut connection of Res blocks  
Memory permitting additional Conv operations can be added following Conv2 in the side branch (lower branch)  

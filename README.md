#### Fault picking with 3D-Unets

#### Meta-overview
PyTorch implementation of various 3d-unets for fault picking. We use [Xinming Wu's](https://github.com/xinwucwp/faultSeg) repository to get the training data (200 images and labels of size 128x128x128). We set aside 20 images as a holdout test set for accuracy benchmarking. These are the 20 images in the validation set of Wu's original implementation, but we keep these now as unseen test-set. The current best model in this repo (unet_3d_res1) siginificantly outperforms the original Wu model and a standard Vnet model in terms of iou on the test set.  

<p align="center"><img width="80%"  src="images/iou_comparison1.PNG" /></p>

#### Current best model IOUs

| Filename | WU | VNet(CE+F1) | Modified VNet | UNet_Res1(CE) | Iter-Unet(CE) |
| -- | -- | -- | -- | -- | -- |
| 0 | 0.695 | 0.672 | 0.000 | 0.000 | 0.000 |
| 1 | 0.701 | 0.721 | 0.000 | 0.000 | 0.000 |
| 2 | 0.774 | 0.672 | 0.000 | 0.000 | 0.000 |  
| 3 | 0.686 | 0.672 | 0.000 | 0.000 | 0.000 |
| 4 | 0.623 | 0.672 | 0.000 | 0.000 | 0.000 |
| 5 | 0.611 | 0.672 | 0.000 | 0.000 | 0.000 | 
| 6 | 0.651 | 0.672 | 0.000 | 0.000 | 0.000 | 
| 7 | 0.631 | 0.672 | 0.000 | 0.000 | 0.000 | 
| 8 | 0.698 | 0.672 | 0.000 | 0.000 | 0.000 | 
| 9 | 0.598 | 0.672 | 0.000 | 0.000 | 0.000 | 
| 10 | 0.706 | 0.724 | 0.000 | 0.000 | 0.000 | 
| 11 | 0.632 | 0.654 | 0.000 | 0.000 | 0.000 | 
| 12 | 0.609 | 0.614 | 0.000 | 0.000 | 0.000 |
| 13 | 0.493 | 0.490 | 0.000 | 0.000 | 0.000 |
| 14 | 0.598 | 0.588 | 0.000 | 0.000 | 0.000 | 
| 15 | 0.561 | 0.575 | 0.000 | 0.000 | 0.000 |
| 16 | 0.652 | 0.636 | 0.000 | 0.000 | 0.000 |
| 17 | 0.672 | 0.727 | 0.000 | 0.000 | 0.000 | 
| 18 | 0.576 | 0.585 | 0.000 | 0.000 | 0.000 | 
| 19 | 0.529 | 0.537 | 0.000 | 0.000 | 0.000 | 

* WU model results are grabbed from the predictions bu [Wu at](https://github.com/xinwucwp/faultSeg/tree/master/data/validation/predict)
* VNet(CE+F1) is the original VNet implementation from [here](https://github.com/mattmacy/vnet.pytorch). The model is trained with cross-entropy3D (CE) + F1-score loss and outperforms the same model trained only with CE or F1


* unet_3d_res1 is a standard 3D-UNet with Residual blocks. The primary performace uplift comes from using a simple Res-block which is structured as:  


&nbsp;&nbsp;&nbsp;&nbsp; Conv1-----------------(+) IN -- LR  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|     
&nbsp;&nbsp;&nbsp;&nbsp; IN -- LR --Conv2 -------

&nbsp;&nbsp;&nbsp;&nbsp; conv1: (3x3) conv3d with stride = 2        

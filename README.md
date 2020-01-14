#### Fault picking with 3D-Unets

#### Meta-overview
PyTorch implementation of various 3d-unets for fault picking. We use [Xinming Wu's](https://github.com/xinwucwp/faultSeg) repository to get the training data (200 images and labels of size 128x128x128). We set aside 20 images as a holdout test set for accuracy benchmarking. These are the 20 images in the validation set of Wu's original implementation, but we keep these now as unseen test-set. The models with  IOU reported are:
1. VNet original
2. VNet custom modification
3. UNet-Res1
4. UNet-Res2  

The current best model, with code released, in this repo is (UNet-Res1) which siginificantly outperforms the original Wu model and both VNets in terms of iou on the test set. The unreleased model UNet-Res2 is **SOTA** for the test-set. Code to be released.

#### Current model IOUs

| Filename | WU | VNet(CE+Dice) | VNet-Modified | UNet_Res1(CE) | Unet-Res2(**SOTA**) | Unet-Res2(TTA) **SOTA** |  
| -- | -- | -- | -- | -- | -- | -- |  
| 0 | 0.695 | 0.672 | 0.711 | 0.764 | 0.801 | 0.814 | 
| 1 | 0.701 | 0.721 | 0.761 | 0.751 | 0.809 | 0.824 |
| 2 | 0.774 | 0.672 | 0.810 | 0.822 | 0.864 | 0.861 | 
| 3 | 0.686 | 0.672 | 0.737 | 0.738 | 0.787 | 0.801 |
| 4 | 0.623 | 0.672 | 0.630 | 0.661 | 0.686 | 0.699 |
| 5 | 0.611 | 0.672 | 0.675 | 0.693 | 0.714 | 0.721 |
| 6 | 0.651 | 0.672 | 0.730 | 0.750 | 0.789 | 0.786 |
| 7 | 0.631 | 0.672 | 0.702 | 0.723 | 0.750 | 0.749 |
| 8 | 0.698 | 0.672 | 0.731 | 0.744 | 0.761 | 0.758 |
| 9 | 0.598 | 0.672 | 0.669 | 0.649 | 0.691 | 0.693 |
| 10 | 0.706 | 0.724 | 0.799 | 0.789 | 0.834 | 0.843 |
| 11 | 0.632 | 0.654 | 0.653 | 0.703 | 0.709 | 0.732 |
| 12 | 0.609 | 0.614 | 0.639 | 0.665 | 0.708 | 0.721 |
| 13 | 0.493 | 0.490 | 0.531 | 0.556 | 0.580 | 0.625 |
| 14 | 0.598 | 0.588 | 0.641 | 0.658 | 0.695 | 0.689 |
| 15 | 0.561 | 0.575 | 0.636 | 0.622 | 0.688 | 0.699 |
| 16 | 0.652 | 0.636 | 0.661 | 0.684 | 0.729 | 0.747 |
| 17 | 0.672 | 0.727 | 0.729 | 0.738 | 0.763 | 0.770 |
| 18 | 0.576 | 0.585 | 0.650 | 0.671 | 0.684 | 0.703 |
| 19 | 0.529 | 0.537 | 0.594 | 0.625 | 0.685 | 0.701 |

* WU model results are grabbed from the predictions by [Wu](https://github.com/xinwucwp/faultSeg/tree/master/data/validation/predict). 


* VNet is the original VNet implementation from [here](https://github.com/mattmacy/vnet.pytorch). The only change from VNet-original is all batchnorm is replaced with InstanceNorm and ReLUs with LeakyReLU. The model is trained with cross-entropy (CE) + dyanmically weighted Dice loss and outperforms the same model trained only with CE or Dice. 

* Modified VNet is a modification made to Vnet's downsampling and upsampling block and outperforms the original VNet and Wu. Unfortunately the modifications means, we are not very faithful to the original VNet model layout. This model is trained with pure CE loss. Not tested CE+Dice(dynamic weighting) loss for this model.

* UNet-Res1 is custom 3D UNet with Residual Blocks. The Res-block  is based on the Kaggle 2017 Data Science Bowl 2nd place winner, but has beedn modified to follow more closely a standard ResNet's Res-block layout. The model is trained with CE and outperforms all the previous models on the test set.

* UNet-Res2 **( code not released)** This is a custom 3D UNet with additional tricks in the model architecture. This model is the **SOTA** in this repo. Arxiv paper is in the works for this and the code will be released with that. TTA results are included for this model to show additional uplift with TTA. Again combo loss improves compared to training with CE loss only. 

* Curios thing: Training with standard class weighting consistently produced much thicker faults than desired. Thus we switched to a combo-loss for the weaker models (VNet) where the weihting is done dynamically (or stochastic weighting) only for the Dice loss term. CE continues to be non weighted. 

* Models are trained on V100 AWS machines. For Unet-Res2 model memory requirement is on the high side. All models are trained with batch-size=num_gpu_cards_on_machine 

### Directory layout

* models : contains the 3 model scripts
* loaders: simple custom loader scripts to efficiently feed the data (TODO:augmentations)
* scripts: to run the trainining (train.py) and prediction (predict.py) scripts. Both only work on GPU enabled devices.
* zoo : trained model files. 
* data: training and test data in npy format.

### Training and Test Data

* Download training and test dataset from data dir. All data is in numpy format with shape X,Y,Z. The TestSet images have been used to benchmark the models. 

 
 

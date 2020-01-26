# Prediction on images in TestDir (total of 20 images)
# predictions using the dat files in Wu's repo
# TestDdata dir is incomplete in my DATA dir
import os, sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import OrderedDict
from models import load_models


def computeiou(pred, actual):
    pred = pred>0.5
    actual = actual>0.5
    inter = actual & pred
    union = actual | pred

    iou = inter.sum()/(union.sum()+1e-8)
    return iou 

def main(args):
    """
    does the prediction for all images in TestDir (the test set) in Wu's repo
    prints out iou and saves preds to user specified dir  
    assumes network output is 2 channels
    """

    image_folder = args.file_path
    label_folder = args.label_path
    wu_folder = args.wu_pred
    output_folder = args.output_path
    flipflag = args.flip_flag   ## For TTA, TODO
    model_arch = args.model_arch
    model_path = args.model_path
    save_flag = 1 #args.save_flag
    verbose = False
    print(" output folder is ", output_folder)


    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_use # set GPU to use

    print(" start loading models with arch: ",model_arch)
    parallel_flag=False
    gpu_flag=True
    output_channels = 2
    model = load_models.getModel(model_arch=model_arch,output_channels=output_channels,parallel_flag=parallel_flag)
    curr_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['model_state']  # this is wrapped in DataParallel so need to undo it
    model_dict = OrderedDict()
    for k, v in curr_dict.items():
        name = k[7:]
        model_dict[name] = v
    model.load_state_dict(model_dict)
    print("---------loaded state dict----")

    if verbose==True:
        print(" ------ printing model on screen --------- ")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(" ----------------------------------------- ")
    model.eval()

    toc = time.time(); count_files=0	
    for files in os.listdir(image_folder):
        filename, ext = os.path.splitext(files)
        image_file = os.path.join(image_folder, filename) +".dat"
        label_file = os.path.join(label_folder, filename) +".dat"

        gx,m1,m2,m3 = np.fromfile(image_file,dtype=np.single),128,128,128
        n1=128; n2=128; n3=128
        image_load = np.reshape(gx,(n1,n2,n3))
        #filename_out = "test_"+filename
        #output_image = os.path.join(output_image_folder,filename_out)+".npy"
        #np.save(output_image, image_load) # turn these 3 lines on to save test images in npy format ( X,Y,Z)
                                           # have to specify output_image_folder path and add to code
        gx,m1,m2,m3 = np.fromfile(label_file,dtype=np.single),128,128,128
        label_load = np.reshape(gx,(n1,n2,n3))

        label_load = label_load.transpose(2,1,0)

        image_load = (image_load - np.mean(image_load))/np.std(image_load)
        image_load = image_load.transpose(2,1,0)
        image_load = np.expand_dims(image_load,0)
        image_load = np.expand_dims(image_load,0)  # 1,1,Z,X,Y

        with torch.no_grad():
            image_tensor = torch.from_numpy(image_load).float().cuda()
            preds = model(image_tensor)

        logits = preds['logits']
        probits = F.softmax(logits,dim=1).data.cpu().numpy()
        pred_argmax = np.argmax(probits[0,:,:,:,:], axis=0).astype(np.float32)
        iou_preds = computeiou(pred_argmax,label_load)
        #iou_wu = computeiou(wu_pred,label_load)
        #print(" ---- for filename %s iou preds: %f wu-model: %f ----"%(filename,iou_preds,iou_wu))
        print(" ---- for filename %s iou preds: %f  ----"%(filename,iou_preds))

        if save_flag==True:
            filename_out = "Preds_"+filename
            output_file = os.path.join(output_folder,filename_out)+".npy"
            np.save(output_file, pred_argmax.transpose(2,1,0)) # save in X,Y,Z npy format same as test labels/images

        count_files +=1

    tic = time.time()	
    print("=====================================================")
    print(" Done prediction for %d files in %f s" %(count_files, tic-toc))
    print("=====================================================")

if __name__ == '__main__':
    help_string = "PyTorch Fault prediction"

    parser = argparse.ArgumentParser(description=help_string)

    parser.add_argument('-f', '--file-path', type=str, metavar='DIR', help='Path where test images is located', required=True)
    parser.add_argument('-l', '--label-path', type=str, metavar='DIR', help='Path where test data labels is located', required=True)
    parser.add_argument('-w', '--wu-pred', type=str, metavar='DIR', help='Path where WU-predictions is located', required=True)
    parser.add_argument('-o', '--output-path', type=str, metavar='DIR', help='Path where predictions will be written out', required=True)
    parser.add_argument('-m', '--model-path', type=str, metavar='DIR', help='Path where trained model is stored', required=True)
    parser.add_argument('-arch', '--model-arch', type=str, metavar='ARCH', help='Architecture of the model (default: vnet)', default='linknet34', required=False)

    parser.add_argument('-flipflag', '--flip-flag', type=int, metavar='W', help='flip image during test for TTA 0 or 1', default=0, required=True)
    parser.add_argument('-gpu-use', '--gpu-use', type=str, metavar='H', help='which gpu to use(default: 0)', default=0, required=True)

    args = parser.parse_args()

    main(args)

    sys.exit(1)

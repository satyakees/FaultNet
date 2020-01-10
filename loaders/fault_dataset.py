"""
simple preprocess the fault training data and prep for loader
Assumes data folder (root_path) has two dirs: images(with all the images), labels (with all the labels). Filenames should be same in each dir
All images and labels are numpy arrays (format : X,Y,Z; Z is the depth dim)
"""

import os
import numpy as np
import imageio

import torch
from torch.utils import data

class FaultPrep(data.Dataset):
    
    def __init__(self, root_path, file_list, transforms=None):
        """
        """
        self.root = root_path
        self.n_classes = 2
            
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
            
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root, "images")
        image_path = os.path.join(image_folder, file_id + ".npy")
        
        label_folder = os.path.join(self.root, "labels")
        label_path = os.path.join(label_folder, file_id + ".npy")

        image = np.load(image_path)   # X,Y,Z
        image = (image - np.mean(image))/np.std(image)
        image = image.transpose(2,1,0)  # Z,X,Y
        image = np.expand_dims(image,0) # 1,Z,X,Y

        label = np.load(label_path)
        label = label.transpose(2,1,0)
        label = np.expand_dims(label,0)

        return image, label

if __name__ == "__main__":
    pass

# simple loader for fault data

import os
from sklearn.model_selection import train_test_split

from torch.utils import data

from .fault_dataset import FaultPrep

def LoadData(training_data_path, file_list, image_size, split=0.15, workers=4, batch_size=1, transforms=None):
    
    if not os.path.exists(training_data_path):
        error_message = 'Folder ' + os.path.abspath(training_data_path) + ' does not exist.'
        raise OSError(error_message)
    

    training_samples, validation_samples = train_test_split(file_list, test_size=split, random_state=393939)   # do the train-val split here
    
    training_dataset = FaultPrep(training_data_path, training_samples, transforms=transforms)
    validation_dataset = FaultPrep(training_data_path, validation_samples, transforms=None)
    
    training_loader = data.DataLoader(training_dataset, batch_size=batch_size, num_workers=workers, shuffle=True,pin_memory=True)
    validation_loader = data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    
    return training_loader, validation_loader
    
    
    

from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import re
import numpy as np
import torch

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        label = []
        image = []
        target = []
        c, k = 0, 0
        if split == 'train':
            fdata = np.loadtxt('Caltech101/train.txt', dtype=str)
        else:
            fdata = np.loadtxt('Caltech101/test.txt', dtype=str)
        for i in range(0, len(fdata)):
            y = fdata[i].split('/')
            if y[0] != 'BACKGROUND_Google':
                target.append(y[0])
                image.append(y[1])
                label.append(c)
                if (target[k] != target[k-1]) & (k > 0):
                    c+=1
                k+=1

        self.image = image
        self.label = label
        self.target = target

       
    def __getitem__(self, index):
        path = os.path.join(self.root, self.target[index], self.image[index])
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        image = pil_loader(path) # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)
        label = torch.from_numpy(np.asarray(self.label[index]))
        label = label.long()
        return image, label, path, self.target[index]

    def __len__(self):

        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.label)
        return length
    

from torchvision.datasets import VisionDataset
from PIL import Image
import numpy as np
import os
import os.path
import sys
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from collections import OrderedDict

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def train_test_split(split):
    image=[]
    label_name=[]

    if split == "train":
        im_folder = np.loadtxt('Caltech101/train.txt', dtype=str)   
    elif split == "test":
        im_folder = np.loadtxt('Caltech101/test.txt', dtype=str)

    for i in range(0, len(im_folder)):
        y = im_folder[i].split('/')
        if y[0] != "BACKGROUND_Google":
            label_name.append(y[0])
            image.append(y[1])
    return image, label_name


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):

        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.split = split 
        image, label_name = train_test_split(self.split)                             
        label_ind = [{val : key for key, val in enumerate(OrderedDict.fromkeys(label_name))}[ele] for ele in label_name]

        self.label_name = label_name
        self.label_ind = label_ind
        self.image = image
        
    def __getitem__(self, index):
        path = os.path.join(self.root, self.label_name[index], self.image[index])
        label = self.label_ind[index]
        image = pil_loader(path) 

        # Applies preprocessing when accessing the image:
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        length = len(self.image)
        return length

from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.split = split 
    
        if self.split== "train":
            im_folder = np.loadtxt('train.txt', dtype=str)
            
        if self.split== "test":
            im_folder = np.loadtxt('test.txt', dtype=str)
        
        image=[]
        l_name=[]
        
        for i in range(0, len(im_folder)):
            y = im_folder[i].split('/')
            if y[0] != "BACKGROUND_Google":
                l_name.append(y[0])
                image.append(y[1])
            #im_path.append(os.path.join(root, y[0], y[1]))
            
        self.l_name = l_name
        self.image = image
    
        #self.im_path = im_path
    
    def __getitem__(self, index):
        path = os.path.join(self.root, self.label[index], self.image[index])
        
        label = self.l_name[index]
        image = pil_loader(path) # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)
    
        return image, label

    def __len__(self):

        length = len(self.image)
        return length

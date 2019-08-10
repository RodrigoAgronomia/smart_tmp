import cv2
import numpy as np
import torch.utils.data


#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================


class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, imList, labelList, transform=None):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = np.load(image_name)
        image = np.moveaxis(image, 2, 0)
        label = np.load(label_name)
        label = np.moveaxis(label, 2, 0)
        
        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label)

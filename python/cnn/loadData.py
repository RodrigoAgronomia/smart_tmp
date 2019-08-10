import cv2
import pickle
import numpy as np


#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

class LoadData:
    '''
    Class to laod the data
    '''
    def __init__(self, data_dir, classes, cached_data_file, normVal=1.10):
        '''
        :param data_dir: directory where the dataset is kept
        :param classes: number of classes in the dataset
        :param cached_data_file: location where cached file has to be stored
        :param normVal: normalization value, as defined in ERFNet paper
        '''
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.ones(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.cached_data_file = cached_data_file
        
    def readFile(self, fileName, trainStg=False):
        '''
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        '''
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                line_arr = line.split(',')
                img_file = (line_arr[0].strip()).strip()
                label_file = (line_arr[1].strip()).strip()
                
                if trainStg == True:
                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)
        return 0

    def processData(self):
        '''
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        print('Processing training data')
        return_val = self.readFile('train.txt', True)

        print('Processing validation data')
        return_val1 = self.readFile('val.txt')

        print('Pickling data')
        data_dict = dict()
        data_dict['trainIm'] = self.trainImList
        data_dict['trainAnnot'] = self.trainAnnotList
        data_dict['valIm'] = self.valImList
        data_dict['valAnnot'] = self.valAnnotList

        data_dict['mean'] = self.mean
        data_dict['std'] = self.std
        data_dict['classWeights'] = self.classWeights

        pickle.dump(data_dict, open(self.cached_data_file, "wb"))
        return data_dict





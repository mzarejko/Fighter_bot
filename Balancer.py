import os
from sklearn.model_selection import train_test_split
import Settings
import cv2
from Data_loader import Data_loader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from distutils.dir_util import copy_tree
import pickle
from collections import Counter

class Balancer:

    def __init__(self, loader):
        self.__train_valid_ratio = 0.2
        self.__train_test_ratio = 0.1
        self.dataset_x_name = 'dataset_x'
        self.dataset_y_name = 'dataset_y'
        self.__loader = loader


    def split(self, x_paths, y_paths):
        train_valid_x, test_x, train_valid_y,  test_y = train_test_split(x_paths, y_paths, test_size=self.__train_test_ratio)
        train_x, valid_x,  train_y, valid_y = train_test_split(train_valid_x, train_valid_y, test_size=self.__train_valid_ratio)
        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


    def save_split_x(self, paths, dst):
        for path in tqdm(paths):
            self.__loader.copy_folder(path, dst+str(path).split('/')[-1])

    def save_split_y(self, paths, dst):
        for path in tqdm(paths):
            self.__loader.copy_file(path, dst+str(path).split('/')[-1])


    def prepare_data(self, x_path, y_path):
        x = self.__loader.get_paths(x_path)
        y = self.__loader.get_paths(y_path)
        '''
        out = []
        for i in y:
            with open(i, 'rb') as f:
                out.append(pickle.load(f))
        x = [tuple(element) for element in out]
        freq = Counter(x)
        '''
        #IMPORTANT, without this file read in bad order, features not the same as labels
        x,y = shuffle(x, y)
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = self.split(np.array(x), np.array(y))
        self.save_split_x(train_x, Settings.TRAIN_X_PATH)
        self.save_split_x(valid_x, Settings.VALID_X_PATH)
        self.save_split_x(test_x, Settings.TEST_X_PATH)

        self.save_split_y(train_y, Settings.TRAIN_Y_PATH)
        self.save_split_y(valid_y, Settings.VALID_Y_PATH)
        self.save_split_y(test_y, Settings.TEST_Y_PATH)

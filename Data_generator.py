import os
import numpy as np
import random
import pickle
import Settings
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
'''
    For generate datasets from directory to GPU
'''

class Data_generator:

    def __init__(self, path_x, path_y,  batch_size):
        self.path_x = path_x
        self.path_y = path_y
        self.__list_x = sorted(os.listdir(path_x))
        self.__list_y = sorted(os.listdir(path_y))

        self.all_x = []
        self.all_y = []

        self.size_data = len(self.__list_x)
        self.__it = batch_size

    def __shuffle_data(self, data_x, data_y):
        x, y = shuffle(data_x, data_y)

        return x, y

    def __normalize(self, data):
        dataset = []
        for d in data:
            dataset.append(d/255)

        return np.array(dataset)


    def __open_pickel(self, data):
        with open(data, 'rb') as f:
            datasets = pickle.load(f)

        return np.array(datasets)

    def load_dataset(self, path):

        data = os.listdir(path)

        datasets = []
        for i in range(len(data)):
            seq_path = path + str(data[i])
            sequence = os.listdir(seq_path)
            sequences = []
            for j in range(len(sequence)):
                img_path = seq_path + sequence[j]
                img = cv2.imread(img_path)
                sequences.append(img)
                datasets.append(sequences)
        return datasets

    def batch_dispatch(self):
        counter = 0

        self.__list_x, self.__list_y= self.__shuffle_data(self.__list_x, self.__list_y)
        while counter <= self.size_data:
            batch_x = []
            batch_y = []
            for i in range(self.__it):
                image_seqs = []
                labels = []
                path_img = os.path.join(self.path_x, self.__list_x[counter])
                path_label = os.path.join(self.path_y, self.__list_y[counter])
                for img_name in os.listdir(path_img):
                    if Settings.CHANNELS == 1:
                        img = cv2.imread(path_img + '/' + img_name, 0)
                    else:
                        img = cv2.imread(path_img + '/' + img_name)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img =cv2.resize(img, (Settings.WIDTH_HP_DATA, Settings.HEIGHT_HP_DATA))
                    image_seqs.append(img)

                image_seqs = self.__normalize(image_seqs)

                #If CHANNEL = 1 this should be True
                if len(image_seqs.shape) != 4:
                    image_seqs = image_seqs.reshape((image_seqs.shape[0], image_seqs.shape[1], image_seqs.shape[2], Settings.CHANNELS))




                labels.append(self.__open_pickel(path_label))

                labels = np.array(labels)
                batch_x.append(image_seqs)
                batch_y.append(labels)
                counter += 1
                if counter >= self.size_data:
                    counter = 0
                    self.__list_x, self.__list_y = self.__shuffle_data(self.__list_x, self.__list_y)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y).reshape((-1,2))


            yield batch_x, batch_y
import Settings
from mss import mss
import cv2
import numpy as np
import pickle
import os
import cv2
import time
import shutil
import matplotlib.pyplot as plt



class Data_loader:

    '''
    class for saving, creating, loading data
    updating work base on Data_updater
    '''

    def __init__(self, label_idx=None):
        self.__name_screenshots = "sequence"
        self.__name_screenshots_folder = "screenshot"
        self.__label = [0 for _ in range(len(Settings.LABEL_CLASS.values()))]
        if label_idx != None:
            self.__label[label_idx] = 1
        else:
            self.__label = None

        self.__seq_delay = 0.01
        self.__first_delay = 0.3

    # loop for finding last number of folder with image sequence
    def __get_next_folder(self, path):
        offset_folder = [0]
        for folder in os.listdir(path):
            num = int(folder.split('_')[-1])
            offset_folder.append(num)
        # if all dir empty then create first folder
        if not os.path.isdir(path+self.__name_screenshots_folder+'_'+str(max(offset_folder))):
            return path + self.__name_screenshots_folder + '_' + str(max(offset_folder))+ '/'
        # for checking if last folder is full
        if len(os.listdir(path+self.__name_screenshots_folder+'_'+str(max(offset_folder)))) == Settings.TIME_STEP:
            dir = path+self.__name_screenshots_folder+'_'+str(max(offset_folder)+1) +'/'
        else:
            dir = path + self.__name_screenshots_folder + '_' + str(max(offset_folder)) + '/'
        return dir


    def __get_next_file(self, path, type='png'):
        offset_file = [-1]
        for file in os.listdir(path):
            num = int(file.split('_')[-1].split('.')[0])
            offset_file.append(num)
        return path + self.__name_screenshots+'_'+str(max(offset_file)+1)+'.'+type

    def save_images(self, data, path):
        for img in data:
            if not os.path.isdir(path):
                os.mkdir(path)
            folder = self.__get_next_folder(path)
            if not os.path.isdir(folder):
                os.mkdir(folder)
            dir = self.__get_next_file(folder)
            cv2.imwrite(dir, img)



    def picke_data(self, data, path):
        dir = self.__get_next_file(path, type='pickle')
        with open(dir, 'wb') as f:
            pickle.dump(data, f)


    def __get_img(self):
        with mss() as sct :
            time.sleep(self.__seq_delay)
            img = sct.grab(Settings.MONITOR)
            img = np.array(img)
            img = cv2.resize(img, (Settings.WIDTH_SCREEN_DATA, Settings.HEIGHT_SCREEN_DATA))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_bar = img[Settings.RESIZE_MONITOR_HP['top']:Settings.RESIZE_MONITOR_HP['height']+Settings.RESIZE_MONITOR_HP['top'],
                      Settings.RESIZE_MONITOR_HP['left']:Settings.RESIZE_MONITOR_HP['width']+Settings.RESIZE_MONITOR_HP['left']]
            img_bar = cv2.cvtColor(img_bar, cv2.COLOR_BGRA2BGR)
        return img_bar

    def load_sequence(self):
        screen_data = []
        time.sleep(self.__first_delay)
        for step in range(Settings.TIME_STEP):
            img = self.__get_img()
            screen_data.append(img)

        return np.array(screen_data)

    def update_data(self, labels_path, features_path):
        if self.__label is None:
            raise Exception('idx label None!')

        #this delay is to reduce number of TIME STEPS
        screen_data = self.load_sequence()
        self.save_images(screen_data, labels_path)
        self.picke_data(self.__label, features_path)
        print(self.__label)


    def check_label_idx(self):
        if self.__label is None:
            raise Exception('idx label None!')
        return list(Settings.LABEL_CLASS.keys())[np.argmax(self.__label)]

    def copy_folder(self, src, dst):
        shutil.copytree(str(src),str(dst))

    def copy_file(self, src, dst):
        shutil.copy(src,dst)

    def get_paths(self, path):
        paths = []
        for p in os.listdir(path):
            paths.append(path+p)
        return sorted(paths)



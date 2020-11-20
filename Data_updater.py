import Settings
from tqdm import tqdm
import time
from pynput.keyboard import Key, Listener, Controller
from pynput import keyboard as pad
import os

class Data_updater:

    '''
    if button from pad is pressed then Data_loader get info about updating dataset
    '''

    def __init__(self, loader, pad_controller):
        self.__loader = loader
        self.__pad = pad_controller
        self.__pad.daemon=True
        self.__data_size = Settings.DATASET_SIZE
        self.__end_loop=False

    def __update_miss(self):
        self.__loader.update_data(Settings.FEATURES_PATH, Settings.LABELS_PATH)

    def __update_hit(self):
        self.__end_loop=False
        while not self.__end_loop:
            if self.__pad.check_input(): 
                print('hit')
                self.__end_loop=True
                break

                  
    def start_updating(self): 
        self.__pad.start()
        while True:
            if self.__loader.check_label_idx() == 'miss':
                self.__update_miss()
            elif self.__loader.check_label_idx() == 'hit':
                self.__update_hit()


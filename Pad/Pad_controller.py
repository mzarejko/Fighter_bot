from inputs import get_gamepad
from inputs import devices
import threading
import Settings
from tqdm import tqdm
import time
from pynput.keyboard import Key, Listener, Controller
import os
import serial
import uinput
import struct
from Pad import Pad_config
from enum import Enum, auto
import numpy as np
import cv2
from Pad.Key_watch import Key_watch
import pandas as pd  

class Key_State(Enum):
    PRESSED= auto()
    RELEASED = auto()


class Mode_State(Enum):
    RECORD = auto()
    PLAY = auto()
    MONITOR = auto()


class Pad_controller(threading.Thread):

    '''
    class work with active gamepad and linux kernel
    
    struct input_event {
        struct timeval time; = {long seconds, long microseconds}
        unsigned short type;
        unsigned short code;
        unsigned int value;

    '''
    

    def __init__(self, mode = Mode_State.MONITOR):
        threading.Thread.__init__(self)
        # code of key for gamepad  
        self.active_inputs = [Pad_config.X, Pad_config.Y, Pad_config.B, 
                              Pad_config.A, Pad_config.RB, Pad_config.RT, Pad_config.LB, Pad_config.LT]
        self.__isPressed = False
        self.__stream_bytes = struct.calcsize(Pad_config.STREAM_TYPE)
        self.__key_state = Key_State.RELEASED
        self.__mode_state = mode
        self.__last_pressed = False

        self.__event = None
        self.__code = None
        self.__clock = Key_watch()

    def __change_state_key(self):
        if self.__key_state == Key_State.PRESSED:
            self.__key_state = Key_State.RELEASED
        else:
            self.__key_state = Key_State.PRESSED

    def __update_input(self, event, code):
        if event == Pad_config.EVENTS['Keys'] and code in self.active_inputs:
            self.__code= code
            self.__event = event
            self.__last_pressed = True

    def __reset_input(self):
        self.__last_pressed = False
        self.__event = None
        self.__code = None
        

    def check_input(self):
        if self.__last_pressed:
            self.__change_state_key()
            self.__reset_input()
            if self.__key_state == Key_State.RELEASED:                
                return True
            else:
                return False
        else:
            return False
 
    def __generate_records(self):
        with open(Pad_config.RECORD_INPUT_PATH) as rec:
            lines = []
            for line in tqdm(rec):
                line = line.rstrip()
                line = line[1:-1]
                line = list(line.split(', '))
                packed = self.__encode(line[0], line[1], line[2], line[3])
                lines.append([packed, line[0]])
            return lines 
            
    def __play(self):
        lines = self.__generate_records()
        self.__clock.start()
        for inp, t in lines:
            print(float(t),' > ', self.__clock.get_time())
            while float(t) > self.__clock.get_time():
                pass
            with open(Pad_config.SERIAL, 'wb') as pl:
                pl.write(inp)

    # prepare data for sending to kernel
    def __encode(self, time, event, code, value):
        sec, mili = divmod(float(time), 1.0)
        return struct.pack(Pad_config.STREAM_TYPE, int(sec), int(mili*1000000.0), int(event), int(code), int(value))


    # decode data from kernel
    def __decode(self, data):
        sec, mili, event, code, value = struct.unpack(Pad_config.STREAM_TYPE, data)
        time = float(sec) + float(mili)/1000000.0
        return time, event, code, value


    def __write_inputs(self, time, event, code, value):
        df = pd.DataFrame(time, event, code, value)
        df.to_csv(Pad_config.RECORD_INPUT_PATH, mode='a',  header=False)


    def __record(self):
        self.__clock.start()
        with open(Pad_config.SERIAL, 'rb') as f:
            while True:
                data = f.read(self.__stream_bytes)
                _, event, code, value = self.__decode(data)
                self.__write_inputs(float(self.__clock.get_time()), int(event), int(code), int(value))

    def __monitor_inputs(self):
        with open(Pad_config.SERIAL, 'rb') as f:
            while True:
                data = f.read(self.__stream_bytes)
                _, event, code, _ = self.__decode(data)
                self.__update_input(event, code) 
        
    def run(self):
        try:
            if self.__mode_state == Mode_State.RECORD:
                self.__record()
            elif self.__mode_state == Mode_State.PLAY:
                self.__play() 
            elif self.__mode_state == Mode_State.MONITOR:
                self.__monitor_inputs()
        except Exception as error:
            print(str(error))

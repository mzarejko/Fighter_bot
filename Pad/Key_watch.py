import time
from Pad import Pad_config

class Key_watch:

    def __init__(self):
        self.__start_time = None
            

    def start(self):
        self.__start_time = time.time()
    
    def get_time(self):
        return float(time.time() - self.__start_time) 


                     
             
    
        
        

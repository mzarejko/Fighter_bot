from Data_updater import Data_updater
from Data_loader import Data_loader
from Pad.Pad_controller import Pad_controller
from Pad.Pad_controller import Mode_State
from Balancer import Balancer
import Settings
from CNN_LSTM import CNN_LSTM
import os
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def record():
    loader = Data_loader(label_idx=Settings.LABEL_CLASS['hit'])    
    controller = Pad_controller(mode=Mode_State.RECORD)  
    updater = Data_updater(loader, controller)
    updater.start_updating()

def gather_data():
    loader = Data_loader(label_idx=Settings.LABEL_CLASS['miss'])
    updater = Pad_controller(record_mode=True)
    updater.start()

def balance_data():
    loader = Data_loader()
    balancer = Balancer(loader)
    balancer.prepare_data(Settings.FEATURES_PATH, Settings.LABELS_PATH)

def train_model():
    cnn = CNN_LSTM()
    model = cnn.create_model()
    model = cnn.train_model(model)
    score = cnn.test_model(model)
    print(score)
    cnn.save_model(model, 'model10.h5')

def predict():
    loader = Data_loader()
    cnn = CNN_LSTM()
    model = cnn.load_model(Settings.MODEL_PATH + 'model10.h5')
    while True:
        seq = loader.load_sequence()
        pred = cnn.predict(model, seq)
        print(pred)


if __name__ == '__main__':
    record()

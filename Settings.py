#settings for Controller
LABEL_CLASS = {'hit': 0, 'miss': 1}

MONITOR = {"top": 10, "left": 0, "width": 800, "height": 600}
#Position of HP bar depends on size of resize game window 800x600
RESIZE_MONITOR_HP = {"top": 85,"left": 440, "width": 280, "height": 25}


#for CNN-RNN
#dont change this settings, releted with pos of MONITOR_HP
WIDTH_SCREEN_DATA = 800
HEIGHT_SCREEN_DATA = 600
WIDTH_HP_DATA = 50
HEIGHT_HP_DATA = 4
CHANNELS =3
TIME_STEP = 6
DATASET_SIZE = 20
BATCH_SIZE=8
EPOCHS = 40

#names of files
TRAIN_X_PATH = 'datasets/train/x/'
TRAIN_Y_PATH = 'datasets/train/y/'
VALID_X_PATH = 'datasets/valid/x/'
VALID_Y_PATH = 'datasets/valid/y/'
TEST_X_PATH = 'datasets/test/x/'
TEST_Y_PATH = 'datasets/test/y/'
MODEL_PATH = 'models/'
 
FEATURES_PATH = 'features/'
LABELS_PATH = 'labels/'








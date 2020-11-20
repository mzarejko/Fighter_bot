from tensorflow.keras.layers import  Input, LSTM, TimeDistributed , Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import Model
import time
import Settings
from Data_generator import Data_generator
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt


'''
Class for training model CNN_LSTM
All configutations are in Settings.py
IMPORTATN: check GPU temperature
'''



class CNN_LSTM:

    def __init__(self):
        self.tensorboard = TensorBoard(log_dir=f"logs/{time.time()}")
        self.__shape = (Settings.TIME_STEP, Settings.HEIGHT_HP_DATA, Settings.WIDTH_HP_DATA , Settings.CHANNELS)


    def create_model(self, num_categories = len(Settings.LABEL_CLASS), learning_rate=0.0001):
        input = Input(shape=self.__shape)

        cnn = TimeDistributed(Conv2D(32, (2, 2), strides=(1, 1), padding='same', activation='relu'))(input)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(cnn)


        cnn = TimeDistributed(Conv2D(64, (2, 2), strides=(1, 1), padding='same', activation='relu'))(cnn)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(cnn)
        cnn = TimeDistributed(Dropout(0.25))(cnn)


        cnn = TimeDistributed(Conv2D(128, (2, 2), strides=(1, 1), padding='same', activation='relu'))(cnn)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(Conv2D(128, (2, 2), strides=(1, 1), padding='same', activation='relu'))(cnn)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(cnn)
        cnn = TimeDistributed(Dropout(0.25))(cnn)

        cnn = TimeDistributed(Conv2D(256, (2, 2), strides=(1, 1), padding='same', activation='relu'))(cnn)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(Conv2D(256, (2, 2), strides=(1, 1), padding='same', activation='relu'))(cnn)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(cnn)
        cnn = TimeDistributed(Dropout(0.25))(cnn)


        cnn = TimeDistributed(Flatten())(cnn)

        cnn_rnn = LSTM(128, return_sequences=True, dropout=0.5)(cnn)
        cnn_rnn = LSTM(128, return_sequences=False, dropout=0.5)(cnn_rnn)

        cnn_rnn = Dense(1024, activation='relu')(cnn_rnn)
        cnn_rnn = Dropout(0.5)(cnn_rnn)
        cnn_rnn = Dense(512, activation='relu')(cnn_rnn)
        cnn_rnn = Dropout(0.5)(cnn_rnn)
        cnn_rnn = Dense(128, activation='relu')(cnn_rnn)
        cnn_rnn = Dense(num_categories, activation="sigmoid")(cnn_rnn)

        full_model = Model([input], cnn_rnn)

        full_model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate),
                           metrics=['binary_accuracy'])
        print(full_model.summary())

        return full_model





    def train_model(self, model):



        train_generator = Data_generator(Settings.TRAIN_X_PATH, Settings.TRAIN_Y_PATH,  Settings.BATCH_SIZE)
        train_generator = train_generator.batch_dispatch()
        valid_generator = Data_generator(Settings.VALID_X_PATH, Settings.VALID_Y_PATH, Settings.BATCH_SIZE)
        valid_generator = valid_generator.batch_dispatch()


        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        #checkpoint = ModelCheckpoint(Settings.CHECKPOINT_FILES, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

        #model.fit(np.array(x), np.array(y), batch_size=batch_size, validation_split=0.2, class_weight=class_weights,  callbacks=[early_stop, self.tensorboard], verbose=1,  epochs=epochs)
        with tf.device('/gpu:0'):
            model.fit(train_generator, epochs=Settings.EPOCHS, steps_per_epoch=len(os.listdir(Settings.TRAIN_X_PATH))//Settings.BATCH_SIZE, validation_data=valid_generator,  validation_steps=1,callbacks=[early_stop,self.tensorboard])


        return model

    def test_model(self, model):
        test_generator = Data_generator(Settings.TEST_X_PATH, Settings.TEST_Y_PATH, Settings.BATCH_SIZE)
        test_generator = test_generator.batch_dispatch()

        score = model.evaluate(test_generator, steps=1)
        return score

    def save_model(self, model, name):
        model.save(name)

    def load_model(self, path):
        model = load_model(path)
        return model


    def predict(self, model, feature):
        output=[]
        for f in feature:
            f =  cv2.resize(f, (Settings.WIDTH_HP_DATA, Settings.HEIGHT_HP_DATA))
            plt.imshow(f)
            if Settings.CHANNELS == 1:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            else:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = f/255
            plt.imshow(f)
            output.append(f)
        output = np.array(output)
        if len(output.shape) !=5:
            if Settings.CHANNELS == 1:
                output = output.reshape((1, output.shape[0], output.shape[1], output.shape[2], 1))
            else:
                output = output.reshape((1, output.shape[0], output.shape[1], output.shape[2], 3))

        pred = model.predict(output)[0,:]
        value= np.argmax(pred)
        keys = list(Settings.LABEL_CLASS.keys())
        return keys[value]
"""
This code is to build and train 2D U-Net
"""
import numpy as np
import sys
import subprocess
import argparse
import os

from keras.models import Model
from keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from layers import ConvOffset2D
from keras import losses

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import pickle

from utils import *
from data import load_train_data

K.set_image_data_format('channels_last')  # Tensorflow dimension ordering

# ----- paths setting -----
data_path = "D:/UnetDataBase" + "/"
model_path = data_path + "models/"
log_path = data_path + "logs/"


# ----- params for training and testing -----
batch_size = 1
cur_fold = 0
plane = 'Z'
epoch = 10
init_lr = 1e-2


# ----- Dice Coefficient and cost function for training -----
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return  -dice_coef(y_true, y_pred)


def get_unet(img_rows, img_cols, flt=8, pool_size=(2, 2, 2), init_lr=1.0e-5):
    """build and compile Neural Network"""

    print("start building NN")
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(flt*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=init_lr), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def get_deform_unet(img_rows, img_cols, flt=8, pool_size=(2, 2, 2), init_lr=1.0e-5):
    print("start building NN")
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(flt, (3, 3), padding='same')(inputs)
    conv1 = ConvOffset2D(flt)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(flt, (3, 3), padding='same')(conv1)
    conv1 = ConvOffset2D(flt)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(flt * 2, (3, 3), padding='same')(pool1)
    conv2 = ConvOffset2D(flt * 2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(flt * 2, (3, 3), padding='same')(conv2)
    conv2 = ConvOffset2D(flt * 2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(flt * 4, (3, 3), padding='same')(pool2)
    conv3 = ConvOffset2D(flt * 4)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(flt * 4, (3, 3), padding='same')(conv3)
    conv3 = ConvOffset2D(flt * 4)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(flt * 8, (3, 3), padding='same')(pool3)
    conv4 = ConvOffset2D(flt * 8)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(flt * 8, (3, 3), padding='same')(conv4)
    conv4 = ConvOffset2D(flt * 8)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(flt * 16, (3, 3), padding='same')(pool4)
    conv5 = ConvOffset2D(flt * 16)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(flt * 16, (3, 3), padding='same')(conv5)
    conv5 = ConvOffset2D(flt * 16)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = concatenate([Conv2DTranspose(flt * 16, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(flt * 8, (3, 3), padding='same')(up6)
    conv6 = ConvOffset2D(flt * 8)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(flt * 8, (3, 3), padding='same')(conv6)
    conv6 = ConvOffset2D(flt * 8)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt * 4, (3, 3), padding='same')(up7)
    conv7 = ConvOffset2D(flt * 4)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(flt * 4, (3, 3), padding='same')(conv7)
    conv7 = ConvOffset2D(flt * 4)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt * 2, (3, 3), padding='same')(up8)
    conv8 = ConvOffset2D(flt * 2)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(flt * 2, (3, 3), padding='same')(conv8)
    conv8 = ConvOffset2D(flt * 2)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (3, 3), padding='same')(up9)
    conv9 = ConvOffset2D(flt)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(flt, (3, 3), padding='same')(conv9)
    conv9 = ConvOffset2D(flt)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=init_lr), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train(fold, plane, batch_size, nb_epoch,init_lr):
    """
    train an Unet model with data from load_train_data()

    Parameters
    ----------
    fold : string
        which fold is experimenting in 4-fold. It should be one of 0/1/2/3

    plane : char
        which plane is experimenting. It is from 'X'/'Y'/'Z'

    batch_size : int
        size of mini-batch

    nb_epoch : int
        number of epochs to train NN

    init_lr : float
        initial learning rate
    """

    print("number of epoch: ", nb_epoch)
    print("learning rate: ", init_lr)

    # --------------------- load and preprocess training data -----------------
    print('-'*80)
    print('         Loading and preprocessing train data...')
    print('-'*80)

    imgs_train, imgs_mask_train = load_train_data(fold, plane)

    imgs_row = imgs_train.shape[1]
    imgs_col = imgs_train.shape[2]

    print(imgs_row, imgs_col)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

    # ---------------------- Create, compile, and train model ------------------------
    print('-'*80)

    print('		Creating and compiling model...')
    print('-'*80)

    model = get_deform_unet(imgs_row, imgs_col, pool_size=(2, 2, 2), init_lr=init_lr)
    print(model.summary())

    print('-'*80)
    print('		Fitting model...')
    print('-'*80)

    ver = 'unet_deform_fd%s_%s_ep%s_lr%s.csv'%(cur_fold, plane, epoch, init_lr)
    csv_logger = CSVLogger(log_path + ver)
    model_checkpoint = ModelCheckpoint(model_path + ver + ".h5",
                                       monitor='loss',
                                       save_best_only=False,
                                       period=10)

    history = model.fit(imgs_train, imgs_mask_train,
                        batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True,
                        callbacks=[model_checkpoint, csv_logger])

    # save_process_result(history)


def save_process_result(history):
    if os.path.exists('result.pkl'):
        os.makedirs('result.pkl')

    file = open('result.pkl', 'wb')
    pickle.dump(history.history, file)
    file.close()


if __name__ == "__main__":

    train(cur_fold, plane, batch_size, epoch, init_lr)

    # drawModel(cur_fold, plane, batch_size, epoch, init_lr)

    print("training done")

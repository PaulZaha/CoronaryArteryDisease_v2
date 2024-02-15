import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice


def model_compiler(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  #"sparse_categorical_crossentropy",
                  metrics="accuracy")


def model_fitter(model,epochs,batchsize,xdata,ydata,valdata):
    hist = model.fit(xdata,ydata
              ,epochs=epochs
              ,batch_size=batchsize
              ,verbose=1
              ,validation_data = valdata
              )
    return hist

def main():
    pass


if __name__ == "__main__":
    main()


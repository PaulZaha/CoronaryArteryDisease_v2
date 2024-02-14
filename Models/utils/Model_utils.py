import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2


def model_compiler(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  #"sparse_categorical_crossentropy",
                  metrics="accuracy")


def model_fitter(model,epochs,batchsize,xdata,ydata):
    hist = model.fit(xdata,ydata
              ,epochs=epochs
              ,batch_size=batchsize,
              verbose=1
              )
    return hist

def main():
    pass


if __name__ == "__main__":
    main()


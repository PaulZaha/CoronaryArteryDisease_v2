import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

from utils.Pipeline import *


def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice

path = os.path.join(os.getcwd(),'..')
checkpoint_path = os.path.join(path,'model.h5')
model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor='val_io_u',
    mode='max',
    save_best_only=True,
    verbose=1
)

def model_compiler(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  #"sparse_categorical_crossentropy",
                  metrics=[
                      tf.keras.metrics.IoU(num_classes=2,target_class_ids=[1],sparse_y_true = True, sparse_y_pred = False)
                      ])


def model_fitter(train_generator,model,epochs
                 ,validation_generator
                 ):
    hist = model.fit(train_generator
              ,epochs=epochs
              ,steps_per_epoch=250
              ,verbose=1
              ,validation_steps=50
              ,validation_data = validation_generator
              ,class_weight={0: 1, 1: 100}
              ,callbacks=[model_callback]
              )
    return hist


def model_evaluater(test_generator,model):
    results = model.evaluate(test_generator)
    print(results)




def main():
    pass


if __name__ == "__main__":
    main()


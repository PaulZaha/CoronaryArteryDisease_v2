import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

# from Pipeline import *
# from Predictor import *


from Pipeline import *
from Predictor import *

#tf.config.run_functions_eagerly(True)

def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

path = os.path.join(os.getcwd(),'..')
checkpoint_path = os.path.join(path,'model.h5')
model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-2,
    decay_steps=250,
    decay_rate=0.96,
    staircase=True
)


def model_compiler(model):
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate_decay),
                  #loss=tf.keras.losses.BinaryCrossentropy(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  run_eagerly=True,
                  #"sparse_categorical_crossentropy",
                  metrics=[
                      tf.keras.metrics.IoU(num_classes=2,target_class_ids=[1],sparse_y_true = True, sparse_y_pred = False,name='IoU_White')
                      ,tf.keras.metrics.IoU(num_classes=2,target_class_ids=[0],sparse_y_true = True, sparse_y_pred = False,name='IoU_Black')
                      #,f1_metric
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

#Funktioniert nicht :/
def f1_metric(y_true,y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    f1 = f1score(y_true,y_pred)
    return f1

def main():
    pass


if __name__ == "__main__":
    main()


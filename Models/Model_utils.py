import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

# from Pipeline import *
# from Predictor import *


from Pipeline import *
from Predictor import *


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
    decay_rate=0.95,
    staircase=True
)


def model_compiler(model):
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate_decay),
                  #loss=tf.keras.losses.BinaryCrossentropy(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  #run_eagerly=True,
                  #"sparse_categorical_crossentropy",
                  metrics=[
                      tf.keras.metrics.IoU(num_classes=2,target_class_ids=[1],sparse_y_true = True, sparse_y_pred = False,name='IoU_White')
                      ,tf.keras.metrics.IoU(num_classes=2,target_class_ids=[0],sparse_y_true = True, sparse_y_pred = False,name='IoU_Black')
<<<<<<< HEAD
                      ,tf.keras.metrics.SparseCategoricalAccuracy()
                      #,f1_metric
=======
>>>>>>> origin/main
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
              ,class_weight={0: 1, 1: 80}
              ,callbacks=[model_callback]
              )
    return hist


def main():
    pass


if __name__ == "__main__":
    main()


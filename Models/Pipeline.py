import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2


def generators(targetsize,batchsize,
               train_image_dir,train_mask_dir
               ,val_image_dir,val_mask_dir
               ,test_image_dir,test_mask_dir
               ):
    """
    Takes targetsize, batchsize and directories for train, validation, test images and masks. Returns zipped generators.
    """
    class_mode = None
    colormode = 'grayscale'

    seed=42
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255
                                                                ,horizontal_flip=True
                                                                ,zoom_range=0.3
                                                                ,rotation_range=40)
    
    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255
                                                              ,horizontal_flip=True
                                                              ,zoom_range=0.3
                                                              ,rotation_range=40)
    
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_image_generator = train_gen.flow_from_directory(directory=train_image_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=batchsize
                                                          )
    train_mask_generator = train_gen.flow_from_directory(directory=train_mask_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=batchsize
                                                          )
    train_generator = zip(train_image_generator,train_mask_generator)


    val_image_generator = val_gen.flow_from_directory(directory=val_image_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=batchsize)
    val_mask_generator = val_gen.flow_from_directory(directory=val_mask_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=batchsize)
    validation_generator = zip(val_image_generator,val_mask_generator)


    test_image_generator = test_gen.flow_from_directory(directory=test_image_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=1)
    test_mask_generator = test_gen.flow_from_directory(directory=test_mask_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=1)
    test_generator = zip(test_image_generator,test_mask_generator)


    return train_generator,validation_generator,test_generator



def main():
    pass


if __name__ == "__main__":
    main()



import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

from utils.Model_utils import *
from utils.Pipeline import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose

def double_conv_block(x,n_filters):
    x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x


def downsample_block(x,n_filters):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(2)(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f,p


def upsample_block(x, conv_features, n_filters):
   x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   x = tf.keras.layers.concatenate([x, conv_features])
   x = tf.keras.layers.Dropout(0.3)(x)
   x = double_conv_block(x, n_filters)
   return x


def build_model(size):
    shape = size + (1,)
    inputs = tf.keras.layers.Input(shape=shape)



    #Encoder Structure
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    
    #Bottleneck feature map
    bottleneck = double_conv_block(p4, 1024)
    
    #Decoder Structure
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    
    # outputs
    #outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    outputs = tf.keras.layers.Conv2D(2, (1,1), padding="same", activation = "sigmoid")(u9)


    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="U-Net")
    unet_model.trainable = True
    return unet_model


def load_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # Lade das Bild mit 3 Farbkanälen (RGB)
    image = tf.cast(image, tf.float32) / 255.0  # Normalisieren der Pixelwerte auf den Bereich [0, 1]

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)  # Lade die Maske mit einem Farbkanal (Graustufen)
    mask = tf.cast(mask, tf.float32) / 255.0  # Normalisieren der Pixelwerte auf den Bereich [0, 1]
    
    return image, mask





def main():
    #Todos:

    #Dice loss implementieren

    size =(512,512)
    epochs = 1
    batch_size = 4
    steps_per_epoch = 250

    #unet = build_model(size)
    net = build_model(size)
    model_compiler(net)


    os.chdir(os.path.join(os.getcwd(),'..'))
    train_image_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train','images')
    train_mask_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train','masks')
    val_image_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','val','images')
    val_mask_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','val','masks')
    test_image_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test','images')
    test_mask_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test','masks')


    train_generator,validation_generator,test_generator = generators(targetsize=size,batchsize=batch_size,
                                                                     train_image_dir=train_image_dir,train_mask_dir=train_mask_dir,
                                                                     val_image_dir=val_image_dir,val_mask_dir=val_mask_dir,
                                                                     test_image_dir=test_image_dir,test_mask_dir=test_mask_dir
                                                                     )

    



    
    model_fitter(train_generator=train_generator,model=net,epochs=epochs
                 #,validation_generator=train_generator
                 )

    #model_evaluater(test_generator=test_generator,model=unet)
    tf.keras.saving.save_model(net,os.path.join(os.getcwd(),'model.h5'),save_format='h5')
    #predictor('1.png',unet,'images',size)
    #pred = unet.predict(test_img)
if __name__ == "__main__":
    main()
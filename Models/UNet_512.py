import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

from utils.Model_utils import *
from utils.Pipeline import *
from utils.Predictor import *

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
    shape = size + (3,)
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
    outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model


   
def main():
    size =(512,512)


    unet = build_model(size)
    
    model_compiler(unet)
    os.chdir(os.path.join(os.getcwd(),'..'))
    train_image_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train','images')
    train_mask_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train','masks')
    

    test_img = img_to_array('images','21.png',size)
    test_img = np.expand_dims(test_img,axis=0)

    x_train = load_images('images',train_image_dir,size)
    print(x_train.shape)
    y_train = load_images('masks',train_mask_dir,size)
    print(y_train.shape)

    unet.fit(x_train,y_train,epochs=3,batch_size=32,verbose=1)
    #hist = model_fitter(unet,3,32,x_train,y_train)

    #tf.keras.saving.save_model(unet,os.getcwd())
    predictor('1.png',unet,'images',size)
    #pred = unet.predict(test_img)
if __name__ == "__main__":
    main()
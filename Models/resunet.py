from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, Softmax, Add, Dropout, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.layers import Concatenate as concatenate
import tensorflow as tf


import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from utils.Model_utils import *
from utils.Pipeline import *


def residual_block(input1):
    #Input
    input1 = BatchNormalization()(input1)

    
    conv_1 = Conv2D(input1.shape[3], (3, 3), activation = 'relu', padding='same')(input1)
    conv_2 = Conv2D(input1.shape[3], (3, 3), activation = 'relu', padding='same')(conv_1)
    batch_1 = BatchNormalization()(conv_2)
    add = Add()([input1,batch_1])

    conv_1 = Conv2D(input1.shape[3], (3, 3), activation = 'relu', padding='same')(add)
    conv_2 = Conv2D(input1.shape[3], (3, 3), activation = 'relu', padding='same')(conv_1)
    batch_1 = BatchNormalization()(conv_2)
    add = Add()([input1,batch_1])

    conv_1 = Conv2D(input1.shape[3], (3, 3), activation = 'relu', padding='same')(add)
    conv_2 = Conv2D(input1.shape[3], (3, 3), activation = 'relu', padding='same')(conv_1)
    batch_1 = BatchNormalization()(conv_2)
    
    add = Add()([input1,batch_1])
    batch = BatchNormalization()(add)
    return batch

def model():
    input11 = Input(shape=(512, 512, 1))
    conv = Conv2D(16, (3, 3), padding='same')(input11)
    
    #Encoder
    res_1 = residual_block(conv)
    conv_1 = Conv2D(32, (2, 2), strides = 2, padding='same')(res_1)
    res_2 = residual_block(conv_1)
    conv_2 = Conv2D(64, (2, 2), strides = 2, padding='same')(res_2)
    res_3 = residual_block(conv_2)
    conv_3 = Conv2D(128, (2, 2), strides = 2, padding='same')(res_3)
    res_4 = residual_block(conv_3)
    conv_4 = Conv2D(256, (2, 2), strides = 2, padding='same')(res_4)
    
    #Bottleneck
    res_7 = residual_block(conv_4)
    res_8 = residual_block(res_7)
    res_9 = residual_block(res_8)
    
    #Decoder
    trconv_3 = Conv2DTranspose(128, (2, 2), strides = 2, padding='same')(res_9)
    res_12 = residual_block(trconv_3)
    #Skip connection 1
    add_3 = Add()([res_12, res_4])
    trconv_4 = Conv2DTranspose(64, (2, 2), strides = 2, padding='same')(add_3)
    res_13 = residual_block(trconv_4)
    #Skip connection 2
    add_4 = Add()([res_13, res_3])
    trconv_5 = Conv2DTranspose(32, (2, 2), strides = 2, padding='same')(add_4)
    res_14 = residual_block(trconv_5)
    #Skip connection 3
    add_5 = Add()([res_14, res_2])
    trconv_6 = Conv2DTranspose(16, (2, 2), strides = 2, padding='same')(add_5)
    res_15 = residual_block(trconv_6)
    #Skip connection 4
    add_6 = Add()([res_15, res_1])
    
    conv_7 = Conv2D(16, (3, 3), activation = 'relu', padding='same')(add_6)
    batch_1 = BatchNormalization()(conv_7)
    conv_8 = Conv2D(16, (3, 3), activation = 'relu', padding='same')(batch_1)
    batch_2 = BatchNormalization()(conv_8)
    out1 = Conv2D(2, (1, 1), activation = 'softmax', padding='same', name='seg')(batch_2)
    
    model = keras.Model(inputs=[input11], outputs=[out1], name= "model1") 
    
    return model




def main():

    size =(512,512)
    epochs = 1
    batch_size = 4

    net = model()
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
    
    #gen_insepctor(validation_generator)



    
    model_fitter(train_generator=train_generator,model=net,epochs=epochs
                 ,validation_generator=train_generator
                 )

    #model_evaluater(test_generator=test_generator,model=unet)
    tf.keras.saving.save_model(net,os.path.join(os.getcwd(),'model.h5'),save_format='h5')

if __name__ == "__main__":
    main()
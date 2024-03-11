import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
import math
import matplotlib.image 

from Pipeline import *
from Model_utils import *

def img_to_array(images_type,name,size):
    """
    Convert images to numpy arrays with shape=(size,size,1)
    """
    #Open image in greyscale mode
    img=Image.open(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test',images_type,'img',name)).convert('L')

    #Resize imgs
    img = img.resize(size)

    #Convert img to numpy array (shape=(512,512,1))
    img_array = np.array(img)

    #Normalize img values to 0-1 range
    img_array = img_array /255.

    return img_array



def mask_predictions(model,size,image_type):
    """
    Creates predicted masks for test split and returns mean f1 score over all predictions
    """

    #Iterate over all images in test split
    for name in os.listdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test',image_type,'img')):
        
        #Convert img to numpy array
        predict_array = img_to_array(image_type,name,size)

        #transform shape of array to (None,size,size,1) to match model input shape (4th dimension due to batch size needed)
        predict_array = np.expand_dims(predict_array,axis=0)

        #Make prediction
        prediction = model.predict(predict_array)

        #Remove first axis again, shape is now (size,size,2) due to softmax output (percentages that the pixel belongs to foreground/background)
        prediction = np.squeeze(prediction, axis=0)

        #Choose higher percentage, 0=background, 1=foreground. Shape is now (size,size,1)
        prediction = np.argmax(prediction, axis=-1)


        #Convert and rescale numpy array to image
        prediction_img = Image.fromarray(np.uint8(prediction*255),mode='L')
        

        #print(prediction_img)

        #Save images
        os.chdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test','predictions_'+image_type,'img'))
        prediction_img.save(name[:-4] + "_pred.jpg")
        os.chdir(os.path.join(os.getcwd(),'..','..','..','..','..','..'))



def main():
    size =(512,512)
    model = tf.keras.saving.load_model(os.path.join(os.getcwd(),'model.h5'))
    mask_predictions(model,size,'images_prewitt')

if __name__ == "__main__":
    main()


import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
import math

# from Pipeline import *
# from Model_utils import *

from Pipeline import *
from Model_utils import *

def img_to_array(imgormask,name,size):
    """
    Convert images to numpy arrays with shape=(size,size,1)
    """
    #Open image in greyscale mode
    img=Image.open(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test',imgormask,'img',name)).convert('L')

    #Resize imgs
    img = img.resize(size)

    #Convert img to numpy array (shape=(512,512,1))
    img_array = np.array(img)

    #Normalize img values to 0-1 range
    img_array = img_array /255.

    return img_array



def predictor(name,model,imgormask,size):
    """
    Saves predicted mask as .jpg and returns f1 score
    """

    #Convert img to numpy array
    predict_array = img_to_array(imgormask,name,size)

    #transform shape of array to (None,size,size,1) to match model input shape (4th dimension due to batch size needed)
    predict_array = np.expand_dims(predict_array,axis=0)

    #Make prediction
    prediction = model.predict(predict_array)
    
<<<<<<< HEAD
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.argmax(prediction, axis=-1)

=======
    #Remove first axis again, shape is now (size,size,2) due to softmax output (percentages that the pixel belongs to foreground/background)
    prediction = np.squeeze(prediction, axis=0)
>>>>>>> origin/main

    #Choose higher percentage, 0=background, 1=foreground. Shape is now (size,size,1)
    prediction = np.argmax(prediction, axis=-1)

    #Convert and rescale numpy array to image
    prediction_img = Image.fromarray(np.uint8(prediction*255)) 

<<<<<<< HEAD
    prediction_img.save(name[:-4] + "_pred.jpg")

    true_mask = img_to_array_pred('masks',name,size)
    f1 = f1score(true_mask,prediction)
    print(f1)
=======
    #Save prediction mask
    prediction_img.save(name[:-4] + "_pred.jpg")

    #Convert true mask to numpy array (shape=(size,size,1))
    true_mask = img_to_array('masks',name,size)
>>>>>>> origin/main

    #Compute f1 score
    f1 = f1score(true_mask,prediction)

    return f1

def model_evaluate(model,size):
    """
    Creates predicted masks for test split and returns mean f1 score over all predictions
    """
    mean_f1 = []

    #Iterate over all images in test split
    for name in os.listdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test','images','img')):
        
        #Convert img to numpy array
        predict_array = img_to_array('images',name,size)

        #transform shape of array to (None,size,size,1) to match model input shape (4th dimension due to batch size needed)
        predict_array = np.expand_dims(predict_array,axis=0)

        #Make prediction
        prediction = model.predict(predict_array)
        
        #Remove first axis again, shape is now (size,size,2) due to softmax output (percentages that the pixel belongs to foreground/background)
        prediction = np.squeeze(prediction, axis=0)

        #Choose higher percentage, 0=background, 1=foreground. Shape is now (size,size,1)
        prediction = np.argmax(prediction, axis=-1)

        #Convert and rescale numpy array to image
        prediction_img = Image.fromarray(np.uint8(prediction*255)) 

        #Save images
        os.chdir(os.path.join(os.getcwd(),'test_imgs'))
        prediction_img.save(name[:-4] + "_pred.jpg")
        os.chdir(os.path.join(os.getcwd(),'..'))

        #convert true mask to numpy array
        true_mask = img_to_array('masks',name,size)

        #compute f1 score for every image
        f1 = f1score(true_mask,prediction)
        print(name)
        print("F1: " + str(f1))
        mean_f1.append(f1)
<<<<<<< HEAD
    print(mean_f1)
    arr_f1 = np.array(mean_f1)
    meanf1 = np.nanmean(arr_f1)
    print(meanf1)
    return meanf1
=======

    #compute mean f1 score
    print(sum(mean_f1)/len(mean_f1))
    return (sum(mean_f1)/len(mean_f1))
>>>>>>> origin/main


def f1score(y_true,y_pred):
    """
    computes f1 score between true and predicted masks
    """
    
    #compute true positive rate
    TP = np.sum(y_true*y_pred)

    #compute precision
    precision = TP/(np.sum(y_pred)+1e-10)

    #compute recall
    recall = TP/(np.sum(y_true)+1e-10)
    
    #compute f1 from precision and recall
    f1 = (2*precision*recall)/(precision+recall)

    return f1



def main():
    size =(512,512)
    model = tf.keras.saving.load_model(os.path.join(os.getcwd(),'model.h5'))
    #predictor('7.png',model,'images',size)
    model_evaluate(model,size)

if __name__ == "__main__":
    main()


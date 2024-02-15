import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

from utils.Pipeline import *


def predictor(name,model,imgormask,size):
    predict_array = img_to_array(imgormask,name,size)
    predict_array = np.expand_dims(predict_array,axis=0)
    #print(prediction.shape)
    prediction = model.predict(predict_array)
    
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.nanargmax(prediction, axis=-1)
    prediction = Image.fromarray(np.uint8(prediction*255)) 
    
    #prediction = prediction.resize(size)
    prediction.save(name[:-4] + "_pred.jpg")






def main():
    pass


if __name__ == "__main__":
    main()


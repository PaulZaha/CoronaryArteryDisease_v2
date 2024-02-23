import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

from Pipeline import *


def predictor(name,model,imgormask,size):
    predict_array = img_to_array(imgormask,name,size)
    predict_array = np.expand_dims(predict_array,axis=0)
    #print(prediction.shape)
    prediction = model.predict(predict_array)
    
    #np.savetxt('prediction.txt',prediction)
    prediction = np.squeeze(prediction, axis=0)
    print(prediction)
    prediction = np.argmax(prediction, axis=-1)
    print(prediction)
    np.savetxt('prediction.txt',prediction)
    prediction = Image.fromarray(np.uint8(prediction*255)) 
    np.savetxt('prediction_scaled.txt',prediction)
    #prediction = prediction.resize(size)
    prediction.save(name[:-4] + "_pred.jpg")






def main():
    size =(512,512)
    model = tf.keras.saving.load_model(os.path.join(os.getcwd(),'model.h5'))
    predictor('21.png',model,'images',size)

if __name__ == "__main__":
    main()


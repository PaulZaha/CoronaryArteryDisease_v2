import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

from Pipeline import *

def img_to_array_pred(imgormask,name,size):
    img=Image.open(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test',imgormask,'img',name)).convert('L')
    img = img.resize(size)
    img_array = np.array(img)

    
    #img_array = np.expand_dims(img_array[:,:1], axis=0)
    img_array = img_array /255.
    binary_array = np.round(img_array)
    
    #np.savetxt('output.txt',binary_array)
    return binary_array

def predictor(name,model,imgormask,size):
    predict_array = img_to_array_pred(imgormask,name,size)
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


def predictor_test(model,size):
    for name in os.listdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test','images_kirsch','img')):
        predict_array = img_to_array_pred('images',name,size)
        predict_array = np.expand_dims(predict_array,axis=0)
        #print(prediction.shape)
        prediction = model.predict(predict_array)
        
        #np.savetxt('prediction.txt',prediction)
        prediction = np.squeeze(prediction, axis=0)

        prediction = np.argmax(prediction, axis=-1)

        
        prediction = Image.fromarray(np.uint8(prediction*255)) 

        #prediction = prediction.resize(size)
        os.chdir(os.path.join(os.getcwd(),'test_imgs'))
        prediction.save(name[:-4] + "_pred.jpg")
        os.chdir(os.path.join(os.getcwd(),'..'))



def main():
    size =(512,512)
    model = tf.keras.saving.load_model(os.path.join(os.getcwd(),'model.h5'))
    predictor('7.png',model,'images',size)
    #predictor_test(model,size)
if __name__ == "__main__":
    main()


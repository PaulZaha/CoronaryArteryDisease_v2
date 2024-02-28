import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

from Pipeline import *
from Model_utils import *

# from utils.Pipeline import *
# from utils.Model_utils import *

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

    prediction_img = Image.fromarray(np.uint8(prediction*255)) 

    #prediction = prediction.resize(size)
    prediction_img.save(name[:-4] + "_pred.jpg")

    true_mask = img_to_array_pred('masks',name,size)
    f1score(true_mask,prediction)

def predictor_test(model,size):
    mean_f1 = []
    for name in os.listdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test','images','img')):
        predict_array = img_to_array_pred('images',name,size)
        predict_array = np.expand_dims(predict_array,axis=0)
        #print(prediction.shape)
        prediction = model.predict(predict_array)
        
        #np.savetxt('prediction.txt',prediction)
        prediction = np.squeeze(prediction, axis=0)

        prediction = np.argmax(prediction, axis=-1)

        
        prediction_img = Image.fromarray(np.uint8(prediction*255)) 

        #prediction = prediction.resize(size)
        os.chdir(os.path.join(os.getcwd(),'test_imgs'))
        prediction_img.save(name[:-4] + "_pred.jpg")
        os.chdir(os.path.join(os.getcwd(),'..'))

        true_mask = img_to_array_pred('masks',name,size)
        f1 = f1score(true_mask,prediction)
        print(name)
        print("F1: " + str(f1))
        mean_f1.append(f1)
    print(sum(mean_f1)/len(mean_f1))
    return mean_f1


def f1score(y_true,y_pred):
    TP = np.sum(y_true*y_pred)
    precision = TP/np.sum(y_pred)
    recall = TP/np.sum(y_true)
    
    f1 = (2*precision*recall)/(precision+recall)

    return f1



def main():
    size =(512,512)
    model = tf.keras.saving.load_model(os.path.join(os.getcwd(),'model2.h5'))
    #predictor('7.png',model,'images',size)
    predictor_test(model,size)
if __name__ == "__main__":
    main()


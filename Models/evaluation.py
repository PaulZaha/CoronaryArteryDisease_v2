import os
import numpy as np
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt

from Predictor import *

def evaluation(images_type,size):
    f1_list = []
    for name in os.listdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test','images','img')):
        true_mask_path = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','test',images_type,'img',name)
        true_mask = img_to_array('masks',name,size)
        pred_name = name[:-4] + '_pred.jpg'
        pred_mask = img_to_array('predictions_'+images_type,pred_name,size)

        f1 = f1score(true_mask,pred_mask)
        f1_list.append(f1)
    f1_arr = np.array(f1_list)
    print(f1_arr)
    nan_mask = np.isnan(f1_arr)
    f1_arr[nan_mask] = 0
    
    
    return f1_arr

def f1score(y_true,y_pred):
    """
    computes f1 score between true and predicted masks
    """
    
    #compute true positive rate
    TP = np.sum(y_true*y_pred)

    #compute precision
    precision = TP/(np.sum(y_pred)+1e-7)

    #compute recall
    recall = TP/(np.sum(y_true)+1e-7)
    
    #compute f1 from precision and recall
    f1 = (2*precision*recall)/(precision+recall)

    return f1

def histogram(array):
    plt.hist(array,bins=30,color='blue',edgecolor='black')

    plt.xlabel("F1 Score")
    plt.ylabel('Number of images')
    plt.title('Histogram')

    plt.show()


def main():
    size = (512,512)

    f1_array = evaluation('images_prewitt',size)

    #Calculate mean f1
    mean_f1 = np.mean(f1_array)
    print('Mean F1: ' + str(mean_f1))


    histogram(f1_array)


if __name__ == "__main__":
    main()
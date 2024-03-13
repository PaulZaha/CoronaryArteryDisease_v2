import os
import numpy as np
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt
import re
from Predictor import *
from scipy.stats import friedmanchisquare, norm,ttest_rel

#Alphanumerical sorting 
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def evaluation(images_type,size):
    f1_list = []

    img_dir = os.path.join(os.getcwd(), 'Dataset', 'arcade', 'stenosis', 'test', 'images', 'img')
    img_files = sorted(os.listdir(img_dir), key=natural_sort_key)

    for name in img_files:
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

def normalize(array):
    normal_distribution = norm(loc=0,scale=1)
    mean = np.mean(array)
    std_dev = np.std(array)

    standardized = (array-mean)/std_dev
    transformed = normal_distribution.ppf(np.linspace(0.01,0.99,len(array)))
    transformed = transformed *std_dev + mean
    return transformed

def paired_t_test(array1,array2,name):
    statistic,p_value = ttest_rel(array1,array2)
    print("T-Statistik "+name,statistic)
    print("P-Wert "+name,p_value)


def main():
    size = (512,512)

    f1_images = evaluation('images',size)
    f1_kirsch = evaluation('images_kirsch',size)
    f1_prewitt = evaluation('images_prewitt',size)
    f1_sobel = evaluation('images_sobel',size)


    
    


    paired_t_test(f1_kirsch,f1_images,"Kirsch")
    paired_t_test(f1_prewitt,f1_images,"Prewitt")
    paired_t_test(f1_sobel,f1_images,"Sobel")






if __name__ == "__main__":
    main()
import os
import numpy as np
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt
import re
from Predictor import *
from scipy.stats import friedmanchisquare, norm,ttest_rel

from cluster_analysis import *

#Alphanumerical sorting 
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

#Main evaluation function, returning f1, clusters, overlaps
def evaluation(images_type,size):
    f1_list = []
    nonoverlaps_list = []
    truesize_list = []
    predsize_list = []
    overlapssize_list = []

    img_dir = os.path.join(os.getcwd(), 'Dataset', 'arcade', 'stenosis', 'test', 'images', 'img')
    img_files = sorted(os.listdir(img_dir), key=natural_sort_key)

    #Iterate over all test images
    for name in img_files:
        #convert true_mask to array
        true_mask = img_to_array('masks',name,size)
        true_mask = np.where(true_mask>=0.5,1,true_mask)
        true_mask = np.where(true_mask<0.5,0,true_mask)

        #Convert prediction mask to array
        pred_name = name[:-4] + '_pred.jpg'
        pred_mask = img_to_array('predictions_'+images_type,pred_name,size)
        pred_mask = np.where(pred_mask>=0.5,1,pred_mask)
        pred_mask = np.where(pred_mask<0.5,0,pred_mask)

        #get values for overlaps and sizes
        nonoverlaps,truesize,predsize,overlapssize = count_overlapping_clusters(true_mask,pred_mask)
        nonoverlaps_list.append(nonoverlaps)
        truesize_list.append(truesize)
        predsize_list.append(predsize)
        overlapssize_list.append(overlapssize)

        #get f1 values
        f1 = f1score(true_mask,pred_mask)
        f1_list.append(f1)
    f1_arr = np.array(f1_list)
    nonoverlap_arr = np.array(nonoverlaps_list)

    nan_mask = np.isnan(f1_arr)
    f1_arr[nan_mask] = 0
    
    
    return f1_arr,nonoverlaps_list,truesize_list,predsize_list,overlapssize_list

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

#Not used anymore
def histogram(array):
    plt.hist(array,bins=30,color='blue',edgecolor='black')

    plt.xlabel("F1 Score")
    plt.ylabel('Number of images')
    plt.title('Histogram')

    plt.show()

#Not used
def normalize(array):
    normal_distribution = norm(loc=0,scale=1)
    mean = np.mean(array)
    std_dev = np.std(array)

    standardized = (array-mean)/std_dev
    transformed = normal_distribution.ppf(np.linspace(0.01,0.99,len(array)))
    transformed = transformed *std_dev + mean
    return transformed


def paired_t_test(array1,array2,name):
    """
    Significance testing 
    """
    statistic,p_value = ttest_rel(array1,array2)
    print("T-Statistik "+name,statistic)
    print("P-Wert "+name,p_value)


def main():
    size = (512,512)

    f1_images,overlaps_images,truesize,predsize_images,overlaps_size_images = evaluation('images',size)
    f1_kirsch,overlaps_kirsch,a,predsize_kirsch,overlaps_size_kirsch = evaluation('images_kirsch',size)
    f1_prewitt,overlaps_prewitt,b,predsize_prewitt,overlaps_size_prewitt = evaluation('images_prewitt',size)
    f1_sobel,overlaps_sobel,c,predsize_sobel,overlaps_size_sobel = evaluation('images_sobel',size)

    truth_clusters = true_clusters(size)
    pred_clusters_images = pred_clusters('images',size)
    pred_clusters_kirsch = pred_clusters('images_kirsch',size)
    pred_clusters_prewitt = pred_clusters('images_prewitt',size)
    pred_clusters_sobel = pred_clusters('images_sobel',size)

    non_overlaps_images = pred_clusters_images-overlaps_images
    non_overlaps_kirsch = pred_clusters_kirsch-overlaps_kirsch
    non_overlaps_prewitt = pred_clusters_prewitt-overlaps_prewitt
    non_overlaps_sobel = pred_clusters_sobel-overlaps_sobel


    print("Mean Clusters: ")
    print(np.mean(pred_clusters_images))
    print(np.mean(pred_clusters_kirsch))
    print(np.mean(pred_clusters_prewitt))
    print(np.mean(pred_clusters_sobel))

    cluster_list = [pred_clusters_images,pred_clusters_kirsch-pred_clusters_prewitt,pred_clusters_sobel]

    print("Mean Non-Overlaps:")
    print(np.mean(non_overlaps_images))
    print(np.mean(non_overlaps_kirsch))
    print(np.mean(non_overlaps_prewitt))
    print(np.mean(non_overlaps_sobel))

    nonoverlaps_list = [non_overlaps_images,non_overlaps_kirsch,non_overlaps_prewitt,non_overlaps_sobel]

    print("Mean Overlaps: ")
    print(np.mean(overlaps_images))
    print(np.std(overlaps_images))
    print(np.mean(overlaps_kirsch))
    print(np.std(overlaps_kirsch))
    print(np.mean(overlaps_prewitt))
    print(np.std(overlaps_prewitt))
    print(np.mean(overlaps_sobel))
    print(np.std(overlaps_sobel))

    print("Cluster sizes: ")
    print(np.mean(truesize))
    print(np.mean(predsize_images))
    print(np.mean(predsize_kirsch))
    print(np.mean(predsize_prewitt))
    print(np.mean(predsize_sobel))

    print("Non Overlaps size: ")
    print(np.mean(overlaps_size_images))
    print(np.mean(overlaps_size_kirsch))
    print(np.mean(overlaps_size_prewitt))
    print(np.mean(overlaps_size_sobel))


    paired_t_test(f1_kirsch,f1_images,"Kirsch")
    paired_t_test(f1_prewitt,f1_images,"Prewitt")
    paired_t_test(f1_sobel,f1_images,"Sobel")


if __name__ == "__main__":
    main()
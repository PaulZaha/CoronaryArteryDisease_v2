import os
import numpy as np
import re

from Predictor import *
from evaluation import *


from skimage import morphology, measure
from sklearn.cluster import KMeans



def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def true_clusters(size):
    true_clusters_list = []

    img_dir = os.path.join(os.getcwd(), 'Dataset', 'arcade', 'stenosis', 'test', 'images', 'img')
    img_files = sorted(os.listdir(img_dir), key=natural_sort_key)

    for name in img_files:

        true_mask = img_to_array('masks',name,size)
        true_mask = np.where(true_mask>=0.5,1,true_mask)
        true_mask = np.where(true_mask<0.5,0,true_mask)
        

        true_clusters = measure.label(true_mask,connectivity=2)
        num_true_clusters = np.max(true_clusters)

        true_clusters_list.append(num_true_clusters)
    
    true_clusters_list = np.array(true_clusters_list)
    return true_clusters_list

def pred_clusters(images_type,size):
    pred_clusters_list = []

    img_dir = os.path.join(os.getcwd(), 'Dataset', 'arcade', 'stenosis', 'test', 'images', 'img')
    img_files = sorted(os.listdir(img_dir), key=natural_sort_key)

    for name in img_files:
        pred_name = name[:-4] + '_pred.jpg'
        pred_mask = img_to_array('predictions_'+images_type,pred_name,size)
        pred_mask = np.where(pred_mask>=0.5,1,pred_mask)
        pred_mask = np.where(pred_mask<0.5,0,pred_mask)

        pred_clusters = measure.label(pred_mask,connectivity=1)
        num_pred_clusters = np.max(pred_clusters)

        # Filtern der Cluster nach Mindestgröße
        cluster_sizes = np.bincount(pred_clusters.flat)
        clusters_to_remove = np.where(cluster_sizes < 96)[0]

        for cluster_label in clusters_to_remove:
            pred_mask[pred_clusters == cluster_label] = 0

        # Erneutes Etikettieren der Cluster, nachdem kleine Cluster entfernt wurden
        pred_clusters = measure.label(pred_mask, connectivity=1)
        num_pred_clusters = np.max(pred_clusters)


        pred_clusters_list.append(num_pred_clusters)

    pred_clusters_list = np.array(pred_clusters_list)
    return pred_clusters_list


def main():
    size=(512,512)
    truth_clusters = true_clusters(size)
    unique,counts=np.unique(truth_clusters,return_counts=True)
    anzahl = dict(zip(unique,counts))
    print(anzahl)
    print(np.mean(truth_clusters))
    # print(truth_clusters)

    pred_clusters_images = pred_clusters('images',size)
    print(np.mean(pred_clusters_images))
    #print(pred_clusters_images)

    pred_clusters_kirsch = pred_clusters('images_kirsch',size)
    print(np.mean(pred_clusters_kirsch))
    #print(pred_clusters_kirsch)

    pred_clusters_prewitt = pred_clusters('images_prewitt',size)
    print(np.mean(pred_clusters_prewitt))
    #print(pred_clusters_prewitt)

    pred_clusters_sobel = pred_clusters('images_sobel',size)
    print(np.mean(pred_clusters_sobel))
    #print(pred_clusters_sobel)

    paired_t_test(pred_clusters_images,pred_clusters_kirsch,'Kirsch')
    paired_t_test(pred_clusters_images,pred_clusters_prewitt,'Prewitt')
    paired_t_test(pred_clusters_images,pred_clusters_sobel,'Sobel')


if __name__ == "__main__":
    main()


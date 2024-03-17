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
        clusters_to_remove = np.where(cluster_sizes < 1)[0]

        for cluster_label in clusters_to_remove:
            pred_mask[pred_clusters == cluster_label] = 0

        # Erneutes Etikettieren der Cluster, nachdem kleine Cluster entfernt wurden
        pred_clusters = measure.label(pred_mask, connectivity=1)
        num_pred_clusters = np.max(pred_clusters)


        pred_clusters_list.append(num_pred_clusters)

    pred_clusters_list = np.array(pred_clusters_list)
    return pred_clusters_list


def count_overlapping_clusters(true_mask, pred_mask):
    truesize_list = []
    predsize_list = []
    overlapping_cluster_sizes = []
    true_clusters = measure.label(true_mask, connectivity=2)
    pred_clusters = measure.label(pred_mask, connectivity=2)

    true_cluster_sizes = np.bincount(true_clusters.flat)[1:]
    truesize_full = sum(true_cluster_sizes)
    truesize_list.append(truesize_full)
    pred_cluster_sizes = np.bincount(pred_clusters.flat)[1:]
    predsize_full = sum(pred_cluster_sizes)
    predsize_list.append(predsize_full)
    

    true_cluster_pixels = (true_clusters > 0).astype(int)
    pred_cluster_pixels = (pred_clusters > 0).astype(int)


    true_cluster_ids = np.unique(true_clusters * true_cluster_pixels)
    pred_cluster_ids = np.unique(pred_clusters * pred_cluster_pixels)

    overlapping_cluster_ids = np.intersect1d(true_cluster_ids, pred_cluster_ids)

    for cluster_id in overlapping_cluster_ids:
        pred_cluster_size = np.sum(pred_clusters == cluster_id)
        overlapping_cluster_sizes.append(pred_cluster_size)
    overlapping_cluster_sizes_value = sum(overlapping_cluster_sizes[1:])

    

    overlapping_clusters_count = len(overlapping_cluster_ids)

    return overlapping_clusters_count,truesize_list,predsize_list,overlapping_cluster_sizes_value

    

def main():
    pass
    
if __name__ == "__main__":
    main()


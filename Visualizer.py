import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


def show_image(split,name):
    """
    Shows images with bounding boxes. Args:[name: 'image_id.jpg']
    """

    #Navigating to train_images folder
    os.chdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'images'))

    #Instantiate subplots
    fig,ax=plt.subplots()
    ax.imshow(mpimg.imread(name),cmap='gray')
    plt.axis('off')

    #Navigate back to main folder
    os.chdir(os.path.join(os.getcwd(),'..','..','..','..','..'))

    #Add bounding box
    #rect = boundingboxsplit,name,fig,ax)
    #ax.add_patch(rect)
    patches = mask_overlay(split,name,fig,ax)
    for patch in patches:

        ax.add_patch(patch)
    plt.show()


def boundingbox(name,fig,ax):
    #Navigate to train annotation json
    os.chdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'annotations'))

    #Read json data
    with open(split + ".json","r") as f:
        json_data = json.load(f)


    #Iterate through annotations, looking for image_id:name and save values in bbox_values list
    for annotation in json_data['annotations']:
        if annotation['image_id'] == int(name[:-4]):
            bbox_values = annotation['bbox']

    #Instantiate rectangle patch and return to show_image function for bounding box layover
    bbox = patches.Rectangle([bbox_values[0],bbox_values[1]],width=bbox_values[2],height=bbox_values[3],linewidth = 1,edgecolor='r',facecolor='none')
    return bbox


def mask_overlay(split,name,fig,ax):
    #Navigate to train annotation json
    os.chdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'annotations'))
    print(os.getcwd())
    #Read json data
    with open(split + ".json","r") as f:
        json_data = json.load(f)
    os.chdir(os.path.join(os.getcwd(),'..','..','..','..','..'))
    #Create empty lists segmentation_values and patch_list
    segmentation_values = []
    plt_patch_list = []
    id_list = []

    #Get list of ids, which is used as image_id in annotations
    for images in json_data['images']:
        if images['file_name'] == name:
            id_list.append(images['id'])


    #Iterate through annotations, looking for image_id:name and save values in segmentation_values list
    for annotation in json_data['annotations']:
        if annotation['image_id'] in id_list:
            segmentation_values.append(annotation['segmentation'][0])

    for value in segmentation_values:
        #Create np arrays from values
        mask_array = np.array(value)

        #Create overlay with plt.patches.polygon. Reshape value array into x and y pairs.
        plt_patch_list.append(patches.Polygon(np.array(mask_array).reshape(-1, 2), closed=True, edgecolor='r', facecolor='none'))
    return plt_patch_list




def main():
    show_image('train','1.png')





if __name__ == "__main__":
    main()
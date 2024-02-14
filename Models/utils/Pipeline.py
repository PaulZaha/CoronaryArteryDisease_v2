import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

def img_to_array(imgormask,name,size):
    img=Image.open(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train',imgormask,name)).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img)

    
    #img_array = np.expand_dims(img_array[:,:1], axis=0)
    img_array = img_array /255.
    binary_array = np.round(img_array)
    
    #np.savetxt('output.txt',binary_array)
    return binary_array


def load_images(imgormask,dir,size):
    images = []
    for filename in os.listdir(dir):
        img_path = os.path.join(dir,filename)
        img_array = img_to_array(imgormask,filename,size)
        # if img_array.shape==(128,128):
        #     img_array = np.expand_dims(img_array,axis=-1)
        #     img_array = np.tile(img_array,(1,1,3))
        images.append(img_array)
    images = np.array(images)
    return images


def main():
    pass


if __name__ == "__main__":
    main()



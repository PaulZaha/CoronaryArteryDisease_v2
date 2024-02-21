import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def canny_detection(split):
    """
    Creates canny images for different data splits
    """

    #Set img path
    img_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'images','img')

    #Iterative over images in dir
    for image in os.listdir(img_dir):
        os.chdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'images','img'))

        #Read img
        img = cv.imread(image,cv.IMREAD_GRAYSCALE)
        os.chdir(os.path.join(os.getcwd(),'..','..','..','..','..','..'))

        assert img is not None, "not found"

        #Perform canny edge detection and save img to output path
        edges = cv.Canny(img,50,100)
        output_path = os.path.join(os.getcwd(), 'Dataset', 'arcade', 'stenosis',split,'images_canny','img')
        os.chdir(output_path)
        cv.imwrite(image,edges)
        os.chdir(os.path.join(os.getcwd(),'..','..','..','..','..','..'))



def main():
    canny_detection('val')


if __name__ == "__main__":
    main()
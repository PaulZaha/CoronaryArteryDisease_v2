import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def canny_detection(name):
    os.chdir(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','val','images','img'))
    img = cv.imread(name,cv.IMREAD_GRAYSCALE)
    os.chdir(os.path.join(os.getcwd(),'..','..','..','..','..','..'))

    assert img is not None, "not found"
    edges = cv.Canny(img,50,100)
    output_path = os.path.join(os.getcwd(), 'Dataset', 'arcade', 'stenosis', 'val','images_canny','img')
    os.chdir(output_path)
    cv.imwrite(name,edges)
    os.chdir(os.path.join(os.getcwd(),'..','..','..','..','..','..'))

def main():

    img_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','val','images','img')
    for image in os.listdir(img_dir):
        canny_detection(image)
    print('done')


if __name__ == "__main__":
    main()
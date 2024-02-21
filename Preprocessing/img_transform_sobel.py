import os
import cv2 as cv
import numpy as np

def sobel_detection(split):
    path = os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'images','img')


    for image in os.listdir(path):
        os.chdir(path)
        img = cv.imread(image,cv.IMREAD_GRAYSCALE)
        os.chdir(os.path.join(os.getcwd(),'..','..'))
        sobel_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
        sobel_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)

        abs_grad_x = cv.convertScaleAbs(sobel_x)
        abs_grad_y = cv.convertScaleAbs(sobel_y)

        grad = cv.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
        os.chdir(os.path.join(os.getcwd(),'images_sobel','img'))
        cv.imwrite(image,grad)


def main():
    sobel_detection('train')


if __name__ == "__main__":
    main()
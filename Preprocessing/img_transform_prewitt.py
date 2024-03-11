import os
import cv2 as cv
import numpy as np



def prewitt_detection(split):
    path = os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'images','img')


    for image in os.listdir(path):
        os.chdir(path)
        img = cv.imread(image,cv.IMREAD_GRAYSCALE)
        os.chdir(os.path.join(os.getcwd(),'..','..'))
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

        img_prewittx = cv.filter2D(img,-1,kernelx)
        img_prewitty = cv.filter2D(img,-1,kernely)

        abs_grad_x = cv.convertScaleAbs(img_prewittx)
        abs_grad_y = cv.convertScaleAbs(img_prewitty)

        grad = cv.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
        os.chdir(os.path.join(os.getcwd(),'images_prewitt','img'))
        cv.imwrite(image,grad)


def main():
    prewitt_detection('val')


if __name__ == "__main__":
    main()
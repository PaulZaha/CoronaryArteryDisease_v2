import os
import cv2 as cv
import numpy as np



def kirsch_detection(split):
    path = os.path.join(os.getcwd(),'Dataset','arcade','stenosis',split,'images','img')


    for image in os.listdir(path):
        os.chdir(path)
        img = cv.imread(image,cv.IMREAD_GRAYSCALE)
        img = cv.GaussianBlur(img,(3,3),0)
        os.chdir(os.path.join(os.getcwd(),'..','..'))
        kernel1 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
        kernel2 = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
        kernel3 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
        kernel4 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
        kernel5 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
        kernel6 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
        kernel7 = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
        kernel8 = np.array([[-3,-3,5],[-3,0,5],[-3,5,5]])

        img_kirsch1 = cv.filter2D(img,-1,kernel1)
        img_kirsch2 = cv.filter2D(img,-1,kernel2)
        img_kirsch3 = cv.filter2D(img,-1,kernel3)
        img_kirsch4 = cv.filter2D(img,-1,kernel4)
        img_kirsch5 = cv.filter2D(img,-1,kernel5)
        img_kirsch6 = cv.filter2D(img,-1,kernel6)
        img_kirsch7 = cv.filter2D(img,-1,kernel7)
        img_kirsch8 = cv.filter2D(img,-1,kernel8)

        abs_grad_1 = cv.convertScaleAbs(img_kirsch1)
        abs_grad_2 = cv.convertScaleAbs(img_kirsch2)
        abs_grad_3 = cv.convertScaleAbs(img_kirsch3)
        abs_grad_4 = cv.convertScaleAbs(img_kirsch4)
        abs_grad_5 = cv.convertScaleAbs(img_kirsch5)
        abs_grad_6 = cv.convertScaleAbs(img_kirsch6)
        abs_grad_7 = cv.convertScaleAbs(img_kirsch7)
        abs_grad_8 = cv.convertScaleAbs(img_kirsch8)



        grad = cv.addWeighted(abs_grad_1,0.5,abs_grad_2,0.5,0)
        os.chdir(os.path.join(os.getcwd(),'images_kirsch','img'))
        cv.imwrite(image,grad)


def main():
    kirsch_detection('val')


if __name__ == "__main__":
    main()
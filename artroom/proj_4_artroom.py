#Brendan Neal
#ENPM673 Project 4 - Stereo Vision

##Artroom code

##----------------Importing Libraries-----------------##
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


##--------------Creating Calibration Matricies from Text File-----------------------##
K_0 = np.array([[1733.74, 0, 792.27],[0, 1733.74, 541.89], [0,0,1]])
K_1 = np.array([[1733.74, 0, 792.27],[0, 1733.74, 541.89], [0,0,1]])

##------------------------------Read Images-----------------------------------------##
OG_Image_0 = cv.imread('im0.png')
OG_Image_1 = cv.imread('im1.png')


##---------------------------Converting Images to Gray------------------------------##

G_Image0 = cv.cvtColor(OG_Image_0, cv.COLOR_BGR2GRAY)
G_Image1 = cv.cvtColor(OG_Image_1, cv.COLOR_BGR2GRAY)

##---------------------------Create ORB Feature Extractor---------------------------##

ORB = cv.ORB_create()

##-------------------------------Find Key Points--------------------------------------##

keypoints1 = ORB.detect(G_Image0, None)
keypoints2 = ORB.detect(G_Image1, None)

##-------------------------Attach Descriptors to Key Points----------------------------##
keypoints1, descriptors1 = ORB.compute(G_Image0, keypoints1)
keypoints2, descriptors2 = ORB.compute(G_Image1, keypoints2)

##-----------------------------------Draw Key Points-----------------------------------##

ORB_Image_1 = cv.drawKeypoints(G_Image0, keypoints1, None, color = (0,255,0), flags = 0)
plt.imshow(ORB_Image_1)
plt.show()
ORB_Image_2 = cv.drawKeypoints(G_Image1, keypoints2, None, color = (0,255,0), flags = 0)
plt.imshow(ORB_Image_2)
plt.show()
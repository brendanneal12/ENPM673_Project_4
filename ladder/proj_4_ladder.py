#Brendan Neal
#ENPM673 Project 4 - Stereo Vision

#Ladder Code

##----------------Importing Libraries-----------------##
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

##--------------Creating Calibration Matricies from Text File-----------------------##
K_0 = np.array([[1734.16, 0, 333.49],[0, 1734.16, 958.05], [0,0,1]])
K_1 = np.array([[1734.16, 0, 333.49],[0, 1734.16, 958.05], [0,0,1]])

##------------------------------Read Images-----------------------------------------##
OG_Image_0 = cv.imread('/home/brendanneal12/Documents/GitHub/ENPM673_Project_4/ladder/im0.png')
OG_Image_1 = cv.imread('/home/brendanneal12/Documents/GitHub/ENPM673_Project_4/ladder/im1.png')


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

##--------------------------------Matching Features------------------------------------##
BF = cv.BFMatcher()
IM1_IM2_Matches = BF.match(descriptors1,descriptors2)
matches = sorted(IM1_IM2_Matches, key = lambda x :x.distance)
BestMatches = matches[0:100]

Corr_List_IM1_IM2 = []

for match in IM1_IM2_Matches:
        (x1_IM12, y1_IM12) = keypoints1[match.queryIdx].pt
        (x2_IM12, y2_IM12) = keypoints2[match.trainIdx].pt
        Corr_List_IM1_IM2.append([x1_IM12, y1_IM12, x2_IM12, y2_IM12])

Corr_Matrix_IM1_IM2 = np.matrix(Corr_List_IM1_IM2)

##-------------------------------Drawing Matched Features------------------------------##
Match1_2_IMG = cv.drawMatches(G_Image0,keypoints1,G_Image1,keypoints2,BestMatches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(Match1_2_IMG)
plt.show()

##----------------------RANSAC Estimation of Fundemental Matrix------------------------##

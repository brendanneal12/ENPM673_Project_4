#Brendan Neal
#ENPM673 Project 4 - Stereo Vision

##Artroom code

##----------------Importing Libraries-----------------##
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

##========================================Function Definitions=======================================##
'''Below you will find function definitions in order to accomplish the Stero Vision Tasks'''

##----------------------Defining my "Calculate F" Function Pipeline------------------------##
''' You need a minimum of 8 points to estimate fundemental matrix, and since we are 
using RANSAC to calculate this, I am using logic to make some assumptions'''

def normalize_matrix(matrix):
      matrix_bar = np.mean(matrix, axis = 0)
      ubar = matrix_bar[0]
      vbar = matrix_bar[1]

      ucap = matrix[:,0] - ubar
      vcap = matrix[:,1] - vbar
      
      scale = (2/np.mean(ucap**2 + vcap**2))**0.5

      Tscale = np.diag([scale,scale,1])
      Ttrans = np.array([[1,0,-ubar],[0,1,-vbar],[0,0,1]])
      T = Tscale.dot(Ttrans)
      norm_matrix_temp = np.column_stack((matrix, np.ones(len(matrix))))
      normalized_matrix = (T.dot(norm_matrix_temp.T)).T

      return normalized_matrix, T


def GetFundementalMatrix(Matched_Feature_List):
    norm = True
    FirstPoints = Matched_Feature_List[:,0:2]
    SecondPoints = Matched_Feature_List[:,2:4]

    if FirstPoints.shape[0] > 7:
        if norm == True:
            FirstPoints_Norm, T1 = normalize_matrix(FirstPoints)
            SecondPoints_Norm, T2 = normalize_matrix(SecondPoints)
        else:
            FirstPoints_Norm = FirstPoints
            SecondPoints_Norm = SecondPoints

        Amatrix = np.zeros((len(FirstPoints_Norm),9))
        for i in range(0, len(FirstPoints_Norm)):
             X1 = FirstPoints_Norm[i][0]
             Y1 = FirstPoints_Norm[i][1]
             X2 = SecondPoints_Norm[i][0]
             Y2 = SecondPoints_Norm[i][1]
             Amatrix[i] = np.array([X1*X2, X2*Y1,X2,Y2*X1,Y2*Y1,Y2,X1,Y1,1])

        u,s,v_t = np.linalg.svd(Amatrix, full_matrices=True)

        F = v_t.T[:,-1]
        F = F.reshape(3,3)

        U,S,V_T = np.linalg.svd(F)
        S = np.diag(S)
        S[2,2] = 0
        F = np.dot(U, np.dot(S,V_T))

        if norm:
             F = np.dot(T2.T,np.dot(F,T1))
        return F
    else:
         return None
    
def CalcRANSACError(IterF, AllFeatures):
     error_array = []
     for feature in AllFeatures:
        X1 = feature[0:2]
        X2 = feature[2:4]
        X1_Temp = np.array([X1[0], X1[1], 1]).T
        X2_Temp = np.array([X2[0], X2[1], 1])
        error_raw = np.dot(X1_Temp, np.dot(IterF, X2_Temp))
        error = np.abs(error_raw)
        error_array.append(error)

     return error_array

def Calculate_F_RANSAC(Feature_List):
    iter_max = math.inf #Generate Temporary Max Iteration. This will change later
    iteration = 0 #Init First Iterationj
    max_inliers = 0 #Init max_inliers
    best_model = None #create best model variable
    prob_outlier = 0 #I want 0 outlier probability
    prob_des = 0.95 #I wanta  95% Accuracy Rate
    n = len(Feature_List) #Init N
    inlier_count = 0 #Init Inlier Count
    threshold = 0.5

    while iteration < iter_max: #While iteration number is less than calculated max
        num_rows = Feature_List.shape[0]
        rand_idxs = np.random.choice(num_rows,size = 8)
        Feature_Sample = Feature_List[rand_idxs,:]
        iteration_model = GetFundementalMatrix(Feature_Sample)
        error = CalcRANSACError(iteration_model, Feature_List)
        error = np.array(error)
        inlier_count = np.count_nonzero(error < threshold)

        if inlier_count > max_inliers: #If the number of inliers is greater than the current max:
            max_inliers = inlier_count #Update the Max Inliers
            print("Max Inliers:", max_inliers)
            best_model = iteration_model #Update the current best model
            prob_outlier = 1-(inlier_count/n) #Calculate the probability of an outlier

        if prob_outlier > 0: #If the probability of an outlier is greater than 0:
            iter_max = math.log(1-prob_des)/math.log(1-(1-prob_outlier)**8) #Recalculate the new number of max iteration number
            print("Max Iterations:", iter_max, "Current Iteration:", iteration, "Current Max Inlier Count:", max_inliers)
            iteration+=1 #Increase Iteration Number
    return best_model

        
























##=========================================="Main" Function==========================================##
''' Here is the Image Processing Pipeline and Application of Functions to Solve Stereo Vision'''
##--------------Creating Calibration Matricies from Text File-----------------------##
K_0 = np.array([[1733.74, 0, 792.27],[0, 1733.74, 541.89], [0,0,1]])
K_1 = np.array([[1733.74, 0, 792.27],[0, 1733.74, 541.89], [0,0,1]])

##------------------------------Read Images-----------------------------------------##
OG_Image_0 = cv.imread('/home/brendanneal12/Documents/GitHub/ENPM673_Project_4/artroom/im0.png')
OG_Image_1 = cv.imread('/home/brendanneal12/Documents/GitHub/ENPM673_Project_4/artroom/im1.png')


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

Corr_Matrix_IM1_IM2 = np.array(Corr_List_IM1_IM2).reshape(-1, 4)

##-------------------------------Drawing Matched Features------------------------------##
Match1_2_IMG = cv.drawMatches(G_Image0,keypoints1,G_Image1,keypoints2,BestMatches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(Match1_2_IMG)
plt.show()

##----------------------RANSAC Estimation of Fundemental Matrix------------------------##
print("RANSAC Starting!")
F_Matrix = Calculate_F_RANSAC(Corr_Matrix_IM1_IM2)
print("\n The F Matrix is:\n", F_Matrix)
print("\n The rank of the F Matrix is:", np.linalg.matrix_rank(F_Matrix))
print("\nThe determinent of the F Matrix is:", np.linalg.det(F_Matrix))


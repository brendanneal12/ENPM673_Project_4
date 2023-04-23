#Brendan Neal
#ENPM673 Project 4 - Stereo Vision

##Artroom code

##----------------Importing Libraries-----------------##
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import timeit

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
     selected_feature_idxs = []
     error_array = []
     for feature in AllFeatures:
        X1 = feature[0:2]
        X2 = feature[2:4]
        X1_Temp = np.array([X1[0], X1[1], 1]).T
        X2_Temp = np.array([X2[0], X2[1], 1])
        error_raw = np.dot(X1_Temp, np.dot(IterF, X2_Temp))
        error = np.abs(error_raw)
        error_array.append(error)
        if error < 0.03:
             selected_feature_idxs.append(np.where(AllFeatures == feature)[0][0])

     return error_array, selected_feature_idxs

def Calculate_F_RANSAC(Feature_List):
    iter_max = math.inf #Generate Temporary Max Iteration. This will change later
    iteration = 0 #Init First Iterationj
    max_inliers = 0 #Init max_inliers
    best_model = None #create best model variable
    prob_outlier = 0 #I want 0 outlier probability
    prob_des = 0.95 #I wanta  95% Accuracy Rate
    n = len(Feature_List) #Init N
    inlier_count = 0 #Init Inlier Count
    threshold = 0.03

    while iteration < iter_max: #While iteration number is less than calculated max
        num_rows = Feature_List.shape[0]
        rand_idxs = np.random.choice(num_rows,size = 8)
        Feature_Sample = Feature_List[rand_idxs,:]
        iteration_model = GetFundementalMatrix(Feature_Sample)
        error, feature_idxs = CalcRANSACError(iteration_model, Feature_List)
        error = np.array(error)
        inlier_count = np.count_nonzero(error < threshold)

        if inlier_count > max_inliers: #If the number of inliers is greater than the current max:
            max_inliers = inlier_count #Update the Max Inliers
            best_iter_feature_idxs = feature_idxs
            print("Max Inliers:", max_inliers)
            best_model = iteration_model #Update the current best model
            prob_outlier = 1-(inlier_count/n) #Calculate the probability of an outlier

        if prob_outlier > 0: #If the probability of an outlier is greater than 0:
            iter_max = math.log(1-prob_des)/math.log(1-(1-prob_outlier)**8) #Recalculate the new number of max iteration number
            print("Max Iterations:", iter_max, "Current Iteration:", iteration, "Current Max Inlier Count:", max_inliers)
            iteration+=1 #Increase Iteration Number  

    best_features = Feature_List[best_iter_feature_idxs,:] #Needed for Later

    return best_model, best_features


##----------------------Defining my "Calculate E" Function-----------------------------------##

def Calc_E_Matrix(K1, K2, F):
     E_Temp = K2.T.dot(F).dot(K1)
     u, s, v = np.linalg.svd(E_Temp)
     s = [1,1,0]
     E = np.dot(u,np.dot(np.diag(s),v))
     return E

##----------------------Defining my "Decompose E" Pipeline-----------------------------------##

def Decompose_E_Matrix(E):
     u, s, v = np.linalg.svd(E)
     W_Matrix = np.array([[0,-1,0],[1,0,0],[0,0,1]])

     Rot_Array = []
     Trans_Array = []

     Rot_Array.append(np.dot(u,np.dot(W_Matrix,v)))
     Rot_Array.append(np.dot(u,np.dot(W_Matrix,v)))
     Rot_Array.append(np.dot(u,np.dot(W_Matrix.T,v)))
     Rot_Array.append(np.dot(u,np.dot(W_Matrix.T,v)))

     Trans_Array.append(u[:,2])
     Trans_Array.append(-u[:,2])
     Trans_Array.append(u[:,2])
     Trans_Array.append(-u[:,2])

     for i in range(4):
          if (np.linalg.det(Rot_Array[i]) < 0):
               Rot_Array[i] = -Rot_Array[i]
               Trans_Array[i] = -Trans_Array[i]

     return Rot_Array, Trans_Array

def Generate3dPoints(K1,K2,bestfeatures, TestRotMatrices, TestTransMatrices):
     Points_3D = []
     TestRot1 = np.identity(3)
     TestTrans1 = np.zeros((3,1))
     I = np.identity(3)

     FirstPoint = np.dot(K1,np.dot(TestRot1,np.hstack((I,-TestTrans1.reshape(3,1)))))

     for i in range(len(TestTransMatrices)):
          X1 = bestfeatures[:,0:2].T
          X2 = bestfeatures[:,2:4].T

          SecondPoint = np.dot(K2,np.dot(TestRotMatrices[i],np.hstack((I,-TestTransMatrices[i].reshape(3,1)))))

          Point3D = cv.triangulatePoints(FirstPoint, SecondPoint, X1, X2)
          Points_3D.append(Point3D)

     return Points_3D

def countPositive(Points3d, RotMatrices, TransMatrices):
     I = np.identity(3)
     P_Matrix = np.dot(RotMatrices,np.hstack((I,-TransMatrices.reshape(3,1))))
     P_Matrix = np.vstack((P_Matrix, np.array([0,0,0,1]).reshape(1,4)))
     num_positive = 0

     for i in range(Points3d.shape[1]):
          Xtest = Points3d[:,i]
          Xtest = Xtest.reshape(4,1)
          XC = np.dot(P_Matrix,Xtest)
          XC = XC / XC.item(3)
          Z = XC[2]
          if Z > 0:
               num_positive += 1

     return num_positive

def Determine_R_T(Points3D, RotMatrices, TransMatrices):
     FirstCount = []
     SecondCount = []

     Rot1 = np.identity(3)
     Trans1 = np.zeros((3,1))
     for i in range(len(Points3D)):
          TestPoint = Points3D[i]
          NormTestPoint = TestPoint/TestPoint[3,:]

          FirstCount.append(countPositive(NormTestPoint, Rot1, Trans1))
          SecondCount.append(countPositive(NormTestPoint,RotMatrices[i], TransMatrices[i]))

     FirstCount = np.array(FirstCount)
     SecondCount = np.array(SecondCount)

     Threshold_Count = int(Points3D[0].shape[1]/2)

     index = np.intersect1d(np.where(FirstCount>Threshold_Count), np.where(SecondCount > Threshold_Count))

     True_Rot = RotMatrices[index[0]]
     True_Trans = TransMatrices[index[0]]

     return True_Rot, True_Trans

##----------------------Defining my Rectification Pipeline-----------------------------------##

def GenerateXPoint(Line,Y):
     x = -(Line[1]*Y + Line[2])/Line[0]
     return x

def ResizeImages(Images):
     images = Images.copy()
     sizes = []
     for image in images:
          X,Y,CH = image.shape
          sizes.append([X,Y,CH])

     sizes = np.array(sizes)
     x_goal,y_goal,_ =  np.max(sizes, axis = 0)

     resized_images = []

     for i, image in enumerate(images):
          resized_image = np.zeros((x_goal,y_goal, sizes[i,2]), np.uint8)
          resized_image[0:sizes[i,0], 0:sizes[i,1], 0:sizes[i,2]] = image
          resized_images.append(resized_image)

     return resized_images


def GenerateEpipolarLines(FirstSetPoints, SecondSetPoints, F, FirstImage, SecondImage, rectified = False):
     Lines1 = []
     Lines2 = []

     EpiImage1 = FirstImage.copy()
     EpiImage2 = SecondImage.copy()

     for i in range(FirstSetPoints.shape[0]):
          X1 = np.array([FirstSetPoints[i,0],FirstSetPoints[i,1],1]).reshape(3,1)
          X2 = np.array([SecondSetPoints[i,0],SecondSetPoints[i,1], 1]).reshape(3,1)



          Line2 = np.dot(F, X1)
          Lines2.append(Line2)

          Line1 = np.dot(F.T, X2)
          Lines1.append(Line1)

          if not rectified:
               Y2_Minimum = 0
               Y2_Max = SecondImage.shape[0]
               X2_Minimum = GenerateXPoint(Line2, Y2_Minimum)
               X2_Max = GenerateXPoint(Line2, Y2_Max)

               Y1_Minimum = 0
               Y1_Max = FirstImage.shape[0]
               X1_Minimum = GenerateXPoint(Line1, Y1_Minimum)
               X1_Max = GenerateXPoint(Line1, Y1_Max)

          else:
               X2_Minimum = 0
               X2_Max = SecondImage.shape[1] - 1
               Y2_Minimum = -Line2[2]/Line2[1]
               Y2_Max = -Line2[2]/Line2[1]

               X1_Minimum = 0
               X1_Max = FirstImage.shape[1] - 1
               Y1_Minimum = -Line1[2]/Line1[1]
               Y1_Max = -Line1[2]/Line1[1]


          cv.circle(EpiImage2, (int(SecondSetPoints[i,0]),int(SecondSetPoints[i,1])), 8, (255,0,0), -1)
          EpiImage2 = cv.line(EpiImage2, (int(X2_Minimum), int(Y2_Minimum)), (int(X2_Max), int(Y2_Max)), (255,0,int(i*2.55)),2)

          cv.circle(EpiImage1, (int(FirstSetPoints[i,0]),int(FirstSetPoints[i,1])), 8, (255,0,0), -1)
          EpiImage1 = cv.line(EpiImage1, (int(X1_Minimum), int(Y1_Minimum)), (int(X1_Max), int(Y1_Max)), (255,0,int(i*2.55)),2)

     Image1, Image2 = ResizeImages([EpiImage1, EpiImage2])

     concat = np.concatenate((Image1, Image2), axis = 1)
     concat = cv.resize(concat, (1920,660))

     return Lines1, Lines2, concat

def SolveRectification(Image1, Image2, FeatureSet1, FeatureSet2, F):
     imheight1, imwidth1 = Image1.shape[:2]
     imheight2, imwidth2 = Image2.shape[:2]

     _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(FeatureSet1), np.float32(FeatureSet2), F, imgSize=(imheight1,imwidth1))
     print("\n H1 is: \n", H1)
     print("\n H2 is: \n", H2)

     Image1_Rect = cv.warpPerspective(Image1, H1, (imheight1,imwidth1))
     Image2_Rect = cv.warpPerspective(Image2, H2, (imheight2, imwidth2))

     FeatureSet1_Rect = cv.perspectiveTransform(FeatureSet1.reshape(-1,1,2),H1).reshape(-1,2)
     FeatureSet2_Rect = cv.perspectiveTransform(FeatureSet2.reshape(-1,1,2),H2).reshape(-1,2)

     H2_T_Inverse = np.linalg.inv(H2.T)
     H1_Inverse = np.linalg.inv(H1)

     F_Rect = np.dot(H2_T_Inverse, np.dot(F, H1_Inverse))

     return Image1_Rect, Image2_Rect, FeatureSet1_Rect, FeatureSet2_Rect, F_Rect

##----------------------Defining my Correspondance Pipeline-----------------------------------##
def SolveCorrespondance(RectIM1, RectIM2):
     GrayIM1 = cv.cvtColor(RectIM1, cv.COLOR_BGR2GRAY)
     GrayIM2 = cv.cvtColor(RectIM2, cv.COLOR_BGR2GRAY)
     window_size = 15
     block = 5

     IMHeight, IMWidth = GrayIM1.shape
     disparity_image = np.zeros(shape = (IMHeight, IMWidth))



     start = timeit.default_timer()

     for i in range(block, GrayIM1.shape[0] - block - 1):
          for j in range(block + window_size, GrayIM1.shape[1] - block - 1):
               ssd = np.empty([window_size,1])
               l = GrayIM1[(i-block):(i+block), (j-block):(j+block)]
               for f in range(0,window_size):
                    r = GrayIM2[(i-block):(i+block), (j-f-block):(j-f+block)]
                    ssd[f] = np.sum((l[:,:] - r[:,:])**2)
               disparity_image[i,j] = np.argmin(ssd)
          #print("Solving for SSD. Please be patient. Current Iteration:", i)

     stop = timeit.default_timer()
     print("That took ", stop-start, "seconds to complete.")
     return disparity_image, ssd

##----------------------Defining my Depth Computation Pipeline-----------------------------------##
def GenerateDepthInfo(RectImage1, Disparity_Info):
     GrayIM1 = cv.cvtColor(RectImage1, cv.COLOR_BGR2GRAY)
     depth_info = np.zeros(shape = GrayIM1.shape).astype(float)
     depth_info[Disparity_Info > 0] = (focal_length*baseline) / (Disparity_Info[Disparity_Info > 0])

     img_depth = ((depth_info/depth_info.max())*255).astype(np.uint8)

     return img_depth


##=========================================="Main" Function==========================================##
''' Here is the Image Processing Pipeline and Application of Functions to Solve Stereo Vision'''
##--------------Creating Calibration Matricies from Text File-----------------------##
K_0 = np.array([[1733.74, 0, 792.27],[0, 1733.74, 541.89], [0,0,1]])
K_1 = np.array([[1733.74, 0, 792.27],[0, 1733.74, 541.89], [0,0,1]])
focal_length = K_0[0][0]
baseline = 536.62

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

for match in BestMatches:
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
F_Matrix, BestFeatures = Calculate_F_RANSAC(Corr_Matrix_IM1_IM2)
print("\n The Fundemental Matrix is:\n", F_Matrix)
print("\n The rank of the F Matrix is:", np.linalg.matrix_rank(F_Matrix))
print("\nThe determinent of the F Matrix is:", np.linalg.det(F_Matrix))

##----------------------Calculation of Essential Matrix------------------------##
E_Matrix = Calc_E_Matrix(K_0,K_1,F_Matrix)
print("\n The Essential Matrix is:\n", E_Matrix)
print("\n The rank of the E Matrix is:", np.linalg.matrix_rank(E_Matrix))
print("\nThe determinent of the E Matrix is:", np.linalg.det(E_Matrix))


##----------------------Decomposition of Essential Matrix------------------------##
Rot_Matrices, Trans_Matrices = Decompose_E_Matrix(E_Matrix)

Points_3d = Generate3dPoints(K_0, K_1, BestFeatures, Rot_Matrices, Trans_Matrices)
R,T = Determine_R_T(Points_3d, Rot_Matrices, Trans_Matrices)
print("\n The estimated Rotation Matrix is:\n", R)
print("\n The estimated Translation Matrix is:\n", T)


##----------------------------Drawing Epipolar Lines-----------------------------##

FirstSet = BestFeatures[:,0:2]
SecondSet = BestFeatures[:,2:4]

Lines1, Lines2, unrec = GenerateEpipolarLines(FirstSet, SecondSet, F_Matrix, OG_Image_0, OG_Image_1, False)
plt.imshow(unrec)
plt.show()

Image1_Rect, Image2_Rect, FeatureSet1_Rect, FeatureSet2_Rect, F_Rect = SolveRectification(OG_Image_0, OG_Image_1, FirstSet, SecondSet,F_Matrix)

Lines1_Rect, Lines2_Rect, Rec = GenerateEpipolarLines(FeatureSet1_Rect, FeatureSet2_Rect, F_Rect, Image1_Rect, Image2_Rect, True)
plt.imshow(Rec)
plt.show()

##----------------------------Solving Correspondance and Disparity-----------------------------##

print("Solving for SSD, please be patient. It can take upwards of 5 min.")
Disp_Image, SSD = SolveCorrespondance(Image1_Rect, Image2_Rect)

show_disp = ((Disp_Image/Disp_Image.max())*255).astype(np.uint8)
plt.imshow(cv.cvtColor(show_disp,cv.COLOR_BGR2RGB))
plt.show()

colormap = plt.get_cmap('inferno')
img_heatmap = (colormap(show_disp) * 2**16).astype(np.uint16)[:,:,:3]
img_heatmap = cv.cvtColor(img_heatmap, cv.COLOR_BGR2RGB)
plt.imshow(img_heatmap)
plt.show()

##------------------------------Compute Depth Image------------------------------------------##
DepthImage = GenerateDepthInfo(Image1_Rect,show_disp)
plt.imshow(cv.cvtColor(DepthImage, cv.COLOR_BGR2RGB))
plt.show()

colormap = plt.get_cmap('inferno')
img_heatmap_depth = (colormap(DepthImage) * 2**16).astype(np.uint16)[:,:,:3]
img_heatmap_depth = cv.cvtColor(img_heatmap_depth, cv.COLOR_BGR2RGB)
plt.imshow(img_heatmap_depth)
plt.show()




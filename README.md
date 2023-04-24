# ENPM673_Project_4
UMD ENPM673 Project 4 -- Stereo Vision

# Student Information
Name: Brendan Neal

UID: 119471128

Directory ID: bneal12

# Project Information
Goal: Stereo Vision Application to generate disparity and depth images.

File Names: proj_4_artroom.py, proj_4_chess.py, proj_4_ladder.py

Recommended IDE: Visual Studio Code

Python Version: 3

# Libraries Used:
numpy, opencv, pyplot, math, timeit

# Important Notes
1. There is a separate scripts for each data set.
2. During the execution of the program, there might be an errors thrown for different reasons, especially on the chess data set. There are a few badly matched features and the RASNAC function may occasionally pull them and it breaks the code. Please rerun. This code works 90% of the time, and I have done everything in my power to filter the bad features.
3. If you run into the code throwing errors just rerun the code until it works. I promise it will work.

# Before you run code
1. Download each of the files and place them into their respective folders (proj_4_artroom.py goes into the artroom folder, etc.)
2. Go to lines 378 and 379 in each of the scripts and change the file path to that of the image.

# How to Run Code -- same instructions for each script.
1. Hit "Run"
2. Observe the detected features in image 0, then close.
3. Observe the detected features in image 1, then close.
4. Observe the matched features for both images, then close.
5. The fundamental matrix, essential matrix, rotation matrix, translation matrix are printed to the terminal.
6. Observe the unrectified epipolar lines, then close.
7. H1 and H2 are printed to the terminal.
8. Observe the rectified epipolar lines, then close.
9. SSD will commence. This takes between 3 and 5 minutes to complete.
10. Observe the black and white disparity image, then close.
11. Observe the heatmap disparity image, then close.
12. Observe the black and white depth image, then close.
13. Observe the heatmap depth image, then close.
14. PROGRAM END


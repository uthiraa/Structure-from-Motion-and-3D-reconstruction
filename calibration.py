# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:07:36 2019

@author: uthir
"""

import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
#============================================
# Camera calibration
#============================================
#Define size of chessboard target.
chessboardSize = (5,7)
#Define arrays to save detected points
objPoints = [] #3D points in real world space 
ipoints = [] #3D points in image plane
#Prepare grid and points to display
objp = np.zeros((np.prod(chessboardSize),3),dtype=np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)
objp = objp.reshape(-1,1,3)
#read images
calibration_paths = glob.glob('C:/Me/UIC/Computer Vision 2/cvproject/chessboard/*.jpg')
#Iterate over images to find intrinsic matrix
for image_path in tqdm(calibration_paths):
#Load image
 image = cv2.imread(image_path)
 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 print("Image loaded, Analizying...")
 #find chessboard corners
 ret,corners = cv2.findChessboardCorners(gray_image, chessboardSize, None)
if ret == True:
  print("Chessboard detected!")
  print(image_path)
  #define criteria for subpixel accuracy
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  #refine corner location (to subpixel accuracy) based on criteria.
  cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
  objPoints.append(objp)
  ipoints.append(corners)
  
  #Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, ipoints,gray_image.shape[::-1], None, None)
#Save parameters into numpy file
np.save("C:/Me/UIC/Computer Vision 2/cvproject/ret", ret)
np.save("C:/Me/UIC/Computer Vision 2/cvproject/K", K)
np.save("C:/Me/UIC/Computer Vision 2/cvproject/dist", dist)
np.save("C:/Me/UIC/Computer Vision 2/cvproject/rvecs", rvecs)
np.save("C:/Me/UIC/Computer Vision 2/cvproject/tvecs", tvecs)
#Get exif data in order to get focal length. 
exif_img = PIL.Image.open(calibration_paths[0])
exif_data = {
 PIL.ExifTags.TAGS[k]:v
 for k, v in exif_img._getexif().items()
 if k in PIL.ExifTags.TAGS}
#Get focal length in tuple form
focal_length_exif = exif_data['FocalLength']
#Get focal length in decimal form
focal_length = focal_length_exif[0]/focal_length_exif[1]
np.save("C:/Me/UIC/Computer Vision 2/cvproject/focal", focal_length)

#Calculate projection error. 
mean_error = 0
for i in range(len(objPoints)):
	ipoints2, _ = cv2.projectPoints(objPoints[i],rvecs[i],tvecs[i], K, dist)
	error = cv2.norm(ipoints[i], ipoints2, cv2.NORM_L2)/len(ipoints2)
	mean_error += error

total_error = mean_error/len(objPoints)
print (total_error)
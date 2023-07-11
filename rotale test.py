

import cv2 
import numpy as np 
  
# read image 
image = cv2.imread('results_crop/0450_06376_b.jpg') 
  
# get dimensions of image 
height, width = image.shape[:2] 
  
# rotation matrix 
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1) 
  
# rotate image by 90 degree clockwise  
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height)) 

 # crop the rotated image to get the license plate area only 
cropped_image = rotated_image[0:height, 0:width] 

 # display original and rotated images for comparison   
cv2.imshow("Original Image", image)   
cv2.imshow("Rotated Image", rotated_image)   
cv2.imshow("Cropped Image", cropped_image)   

 # wait for user input before closing the window   

cv2.waitKey(0)   

 # close all windows   

cv2.destroyAllWindows()
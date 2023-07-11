import torch
import cv2 
import easyocr
import numpy as np
import math
import random

import matplotlib.pyplot as plt


def auto_canny(img, sigma=100):
    # apply a Gaussian blur to the image to remove noise
    blurred = cv2.GaussianBlur(img, (7, 7), 0)

    v = np.median(blurred)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)

    # return the edged image
    return edged

def rotale(orig_img):
    # Load the image
    orig_img = cv2.imread(orig_img)

    img = orig_img.copy()

    dim1,dim2, _ = img.shape

    # Calculate the width and height of the image
    img_y = len(img)
    img_x = len(img[0])

    #Split out each channel
    blue, green, red = cv2.split(img)
    mn, mx = 220, 350
    # Run canny edge detection on each channel

    blue_edges = auto_canny(blue)
    # cv2.imshow('',blue_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    green_edges = auto_canny(green)
    # cv2.imshow('',green_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    red_edges = auto_canny(red)
    # cv2.imshow('',red_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Join edges back into image
    edges = blue_edges | green_edges | red_edges

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts=sorted(contours, key = cv2.contourArea, reverse = True)[:20]
    hulls = [cv2.convexHull(cnt) for cnt in cnts]
    perims = [cv2.arcLength(hull, True) for hull in hulls]
    approxes = [cv2.approxPolyDP(hulls[i], 0.02 * perims[i], True) for i in range(len(hulls))]

    approx_cnts = sorted(approxes, key = cv2.contourArea, reverse = True)
    print(approx_cnts)
    lengths = [len(cnt) for cnt in approx_cnts]
    print(lengths)
    print(lengths.index(4))
    for cnt in approx_cnts:       
        if len(cnt) == 4:
           print(cnt)
           x0, y0 = cnt[0][0]
           distances = [math.sqrt((x-x0)**2 + (y-y0)**2) for [[x,y]] in cnt]
           sorted_cnt = [x for _,x in sorted(zip(distances,cnt), reverse=True)]
           print(distances)
           longest_contour = np.asarray((sorted_cnt[1], sorted_cnt[3]))
           print(longest_contour)
           break
    approx = approx_cnts[lengths.index(4)]
    #longest_contour = approx_cnts[0]
    #print(approx)
    #check the ratio of the detected plate area to the bounding box
    if (cv2.contourArea(approx)/(img.shape[0]*img.shape[1]) > .2):
        cv2.drawContours(img, [approx], -1, (0,255,0), 1)
        x,y,w,h = cv2.boundingRect(approx)

        # Crop the image
        crop_img = img[y:y+h, x:x+w]

        # Get the angle of rotation using the minimum area rectangle
        rect = cv2.minAreaRect(longest_contour)
        angle = rect[2]
        angle = angle - 180
        print(angle)

        # Rotate the image
        rows,cols = crop_img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        rotated = cv2.warpAffine(crop_img,M,(cols,rows))

        plt.imshow(img);plt.show()
        plt.imshow(rotated);plt.show()
    return rotated

def main():
    rotale("results_crop/0128_00503_b.jpg")




if __name__ == '__main__':
    main()        
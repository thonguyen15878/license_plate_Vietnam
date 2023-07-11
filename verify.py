import os
import cv2
import torch
import torch
import cv2 
import easyocr
import numpy as np
import math
import random
import re
import matplotlib.pyplot as plt
import pandas as pd
from License_Plate_Recognize import load_model, crop
image_folder = "testload"
detect_folder = "testdetect"
crop_folder = "testcrop"

# load YOLOv7 model

# define paths
input_folder = "testload/"
detect_folder = "testdetect/"
crop_folder = "testcrop/"
txt_file = "detected_images.txt"
rotale_folder = "crop_rotate/"
#model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model = "best.pt", force_reload=True)
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model = "letterbest.pt", force_reload=True) 
reader = easyocr.Reader(['en'])


def tmp():
# process each image in the input folder
    with open(txt_file, "w") as f:
        for file in os.listdir(input_folder):
            # check if file is an image
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                # build path to image
                image_path = os.path.join(input_folder, file)

                # detect objects in the image
                img = cv2.imread(image_path)
                results = model(image_path)
                results.pandas().xyxy[0]
                image_draw = img.copy()

                # process each detected object
                for i in range(len(results.pandas().xyxy[0])):
                    x_min, y_min, x_max, y_max, conf, clas = results.xyxy[0][i].numpy()
                    x_min = x_min 
                    y_min = y_min 
                    width = x_max - x_min 
                    height = y_max - y_min 

                    if conf >=0.8:
                        # draw bounding box on image
                        image_draw = cv2.rectangle(image_draw, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                        path_detect = os.path.join(detect_folder, file)
                        cv2.imwrite(path_detect, image_draw)
                        
                        # crop object and save to file
                        x, y, w, h = int(x_min), int(y_min),  int(width), int(height)
                        crop_img = img[y:y+h, x:x+w]
                        path_crop = os.path.join(crop_folder, file)
                        cv2.imwrite(path_crop, crop_img)

                        # write image name to txt file
                        f.write(file.rsplit(".", 2)[1] + "\n")

    # close txt file
    f.close()
    return None

def auto_canny(img, sigma=100):
    # apply a Gaussian blur to the image to remove noise
    blurred = cv2.GaussianBlur(img, (11, 11), -10)

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
    lengths = [len(cnt) for cnt in approx_cnts]
    for cnt in approx_cnts:       
        if len(cnt) == 4:
           x0, y0 = cnt[0][0]
           distances = [math.sqrt((x-x0)**2 + (y-y0)**2) for [[x,y]] in cnt]
           sorted_cnt = [x for _,x in sorted(zip(distances,cnt), reverse=True)]
           longest_contour = np.asarray((sorted_cnt[1], sorted_cnt[3]))
           break
    approx = approx_cnts[lengths.index(4)]
    #longest_contour = approx_cnts[0]
    #print(approx)
    #check the ratio of the detected plate area to the bounding box
    if (cv2.contourArea(approx)/(img.shape[0]*img.shape[1]) > .2):
        #cv2.drawContours(img, [approx], -1, (0,255,0), 1)
        x,y,w,h = cv2.boundingRect(approx)

        # Crop the image
        crop_img = img[y:y+h, x:x+w]

        # Get the angle of rotation using the minimum area rectangle
        rect = cv2.minAreaRect(longest_contour)
        angle = rect[2]
        angle = angle - 180

        # Rotate the image
        rows,cols = crop_img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        rotated = cv2.warpAffine(crop_img,M,(cols,rows))

        # plt.imshow(img);plt.show()
        # plt.imshow(rotated);plt.show()
    
        return rotated
    
def rotale_crop():
    for file in os.listdir(crop_folder):
        try:
        # check if file is an image
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_path = os.path.join(crop_folder, file)
                result = rotale(image_path)
                path_crop = os.path.join(rotale_folder, file)
                cv2.imwrite(path_crop, result)
        except Exception:
            pass

def load_model_Letter(rotated_plate):
    #model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model = "letterbest.pt", force_reload=True)  
    
    height, width = rotated_plate.shape[:2]

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    class_names = ['0', '1', '2', '3', '4', '5', '55', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', 'm', 'y']
    text_output = []
    point_dic = {}
    outputs = model(rotated_plate)
    results = outputs.pandas().xyxy[0]
    boxes = results[['xmin', 'ymin', 'xmax', 'ymax']].values
    scores = results['confidence'].values
    labels = results['class'].values
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)
        if cy > (height/2):
            cy = int((height/2)+50)
        else:
            cy = int((height/2)-50)
        label = class_names[int(label)]
        color = random.choice(colors)
        cv2.rectangle(rotated_plate, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        point_dic[(cx,cy)] = label
    sorted_dict = dict(sorted(point_dic.items()))
    for y in range(min(sorted_dict, key=lambda x: x[1])[1], max(sorted_dict, key=lambda x: x[1])[1] + 1):
        for x in range(min(sorted_dict, key=lambda x: x[0])[0], max(sorted_dict, key=lambda x: x[0])[0] + 1):
            if (x, y) in sorted_dict:
                text_output.append(sorted_dict[(x, y)])
    return text_output

def Yolov7_char(crop_folder="testcrop/"):
    label_plates = []
    predict_plates_yolo = []
    predict_plates_ocr = []
    for filename in os.listdir(crop_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # đường dẫn đầy đủ tới file ảnh
            img_path = os.path.join(crop_folder, filename)
            label_plate = "".join(char for char in filename if char.isalnum())
            label_plates.append(label_plate[:-3])
            
            # load ảnh và nhận diện ký tự
            img = cv2.imread(img_path)
            
            result_yolo = load_model_Letter(img)
            result_yolo = ''.join(result_yolo)
            predict_plates_yolo.append(result_yolo)
            
            result_ocr = extract_license_plate_numbers(img)
            result_ocr = ''.join(result_ocr)
            print(result_ocr)
            predict_plates_ocr.append(result_ocr)
            # in kết quả nhận diện
            #print(f"Ký tự trên biển số của {filename}: {''.join(result)}")
    df = pd.DataFrame({'Label_plates': label_plates, 'Predict_plates_yolo': predict_plates_yolo, 'Predict_plates_yolo ' : predict_plates_ocr})
    df.to_csv('plate2.csv', index=False)
    
def extract_license_plate_numbers(image):
    detected_chars = ""    
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    result = reader.readtext(image)
    for (bbox, text, prob) in result: 
        detected_chars += text
    return detected_chars

def easyORC(crop_folder="testcrop/"):
    for filename in os.listdir(crop_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # đường dẫn đầy đủ tới file ảnh
            img_path = os.path.join(crop_folder, filename)
            
            # load ảnh và nhận diện ký tự
            img = cv2.imread(img_path)
            result = extract_license_plate_numbers(img)
            result = "".join(char for char in result if char.isalnum())
            # in kết quả nhận diện
            print(f" {''.join(result)}")       



def main():
    #tmp()
    #license_plate_numbers = extract_license_plate_numbers()
    #print(license_plate_numbers)
    #Yolov7_char()   
    #Yolov7_char()
    easyORC()
if __name__ == '__main__': 
    main()    


import torch
import cv2 
import easyocr
import numpy as np
import random

### Detect plate by yolov7 model
def load_model(imagepath):
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model = "best.pt", force_reload=True)
    
    img = cv2.imread(imagepath)

    # Inference
    results = model(imagepath) # in ra 3 phan tu confidence , classification, bounding box
    results.pandas().xyxy[0]
    image_draw = img.copy()
    
    for i in range(len(results.pandas().xyxy[0])):
        x_min, y_min, x_max, y_max, conf, clas = results.xyxy[0][i].numpy()
        width = x_max - x_min 
        height = y_max - y_min 

        if conf >=0.7:
            image_draw = cv2.rectangle(image_draw, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            path_detect = f'results_detect/{imagepath.split("/")[-1]}'
            cv2.imwrite(path_detect, image_draw)
            x, y, w, h = int(x_min), int(y_min), int(width), int(height)

    return imagepath, x, y, w, h

### crop object
def crop(imagepath, x, y, w, h): 
    image = cv2.imread(imagepath)
    crop_img = image[y:y+h, x:x+w]    
    #cv2.imshow("1",crop_img)
    cv2.waitKey(100)
    path_crop = f"results_crop/{imagepath.split('/')[-1]}"
    cv2.imwrite(path_crop, crop_img)
    #cv2.imshow(crop_img)
    #print(crop_img)
    return crop_img


### read charactor by yolov7 model
def load_model_Letter(plate):
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model = "letterbest.pt", force_reload=True)             
    height, width = plate.shape[:2]
    #print(height, width)
    # Inference
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    class_names = ['0', '1', '2', '3', '4', '5', '55', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', 'm', 'y']
    text_output = []
    point_dic = {}
    outputs = model(plate)
    results = outputs.pandas().xyxy[0]
    boxes = results[['xmin', 'ymin', 'xmax', 'ymax']].values
    scores = results['confidence'].values
    labels = results['class'].values
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        #print(xmin, ymin, xmax, ymax)
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)
        #print(cy)
        if cy > (height/2):
            cy = int((height/2)+50)
        else:
            cy = int((height/2)-50)
        label = class_names[int(label)]
        color = random.choice(colors)
        cv2.rectangle(plate, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        #cv2.putText(img, f'{label}', (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.1, color, 2)
        point_dic[(cx,cy)] = label
    sorted_dict = dict(sorted(point_dic.items()))
    #print(sorted_dict)
    for y in range(min(sorted_dict, key=lambda x: x[1])[1], max(sorted_dict, key=lambda x: x[1])[1] + 1):
        for x in range(min(sorted_dict, key=lambda x: x[0])[0], max(sorted_dict, key=lambda x: x[0])[0] + 1):
            if (x, y) in sorted_dict:
                text_output.append(sorted_dict[(x, y)])
    string = "".join(text_output).upper()
    result = "-".join([string[:4], string[4:]])
    #print(result)
    
    return result

#read charactor by easyORC model 
def OCR(path): 
    IMAGE_PATH = path
    reader = easyocr.Reader(['en'])
    detected_chars = ""
    result = reader.readtext(IMAGE_PATH)
    for (bbox, text, prob) in result: 
        detected_chars += text
        results = "".join(char for char in detected_chars if char.isalnum())
        resultt = "-".join([results[:4], results[4:]])    
    return resultt



def draw(org, x ,y ,w ,h, text):
    img = cv2.imread(org)
    img_draw = img.copy()
    image_draw = cv2.rectangle(img_draw, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    text_color = (0, 255, 0)
    text_img = cv2.putText(image_draw, f'{text}', (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 1,text_color, thickness=2)
    cv2.imshow("1", text_img)
    cv2.waitKey(0)
    return

def main(): 
    path_img ="testload/a.jpg" 
    path, x, y, w, h = load_model(path_img)
    croppath = crop(path, x, y, w, h)
    #tmp = rotale(croppath)
    result = load_model_Letter(croppath)
    #result = OCR(croppath)
    draw(path, x, y, w, h, result)
    


if __name__ == '__main__':
    main()    
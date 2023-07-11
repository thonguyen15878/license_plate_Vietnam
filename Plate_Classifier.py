from tensorflow import keras
from PIL import Image, ImageOps 
import numpy as np 
import os 
from License_Plate_Recognize import OCR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2 
cam = cv2.VideoCapture(0)

import warnings 
warnings.filterwarnings('ignore')


# -------------------------------------------

def plate_recognizie(path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress = True)

    # Load the model 
    model = keras.models.load_model('imageclassifier.h5', compile = False)

    # Create the keras model
    data = np.ndarray(shape = (1,256,256,3), dtype = np.float32)

    # Replace this with the path to your image 
    image = Image.open('./results_crop/0520_00060_b.jpg')

    # Resize the image to a 256x256 
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)  

    # Turn the image into a numpy array 
    image_array = np.asarray(image)

    # Display the resized image 
    image.show()

    # Normalize the image 
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Load the image into the array 
    data[0] = normalized_image_array
    prediction = model.predict(data)

    return prediction, normalized_image_array

if __name__ == "__main__": 
    prediction , image = plate_recognizie("testcrop/79-N2 2094.jpg")    
    OCR(image)
    if prediction[0][0] >= 0.5:
        object = 'Domestic_Motor'
        probability = prediction[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + object)
    else:
        object = 'Others'
        probability = 1 - prediction[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + object)
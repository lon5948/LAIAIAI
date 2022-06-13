# face color analysis given eye center position

import sys
import os
import numpy as np
import cv2
import argparse
import time
from mtcnn.mtcnn import MTCNN
import imutils
import pandas as pd
detector = MTCNN()
from skimage import io

# define HSV color ranges for eyes colors
class_name = ("Blue", "Brown", "Black", "Green", "Other")
EyeColor = {
    class_name[0] : ((166, 21, 50), (240, 100, 85)),
    class_name[1] : ((2, 20, 20), (40, 100, 60)),
    class_name[2] : ((0, 10, 5), (40, 40, 25)),
    class_name[3] : ((60, 21, 50), (165, 100, 85))
}

def check_color(hsv, color):
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and \
    hsv[1] <= color[1][1] and (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False

# define eye color category rules in HSV space
def find_class(hsv):
    color_id = 4
    for i in range(len(class_name)-1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            color_id = i

    return color_id

def eye_color(image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))
    
    result = detector.detect_faces(image)
    if result == []:
        print('Warning: Can not detect any face in the input image!')
        return
    print(result)
    bounding_box = result[0]['box']
    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']
    print(left_eye,right_eye)
    eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
    eye_radius = eye_distance/15 # approximate
   
    cv2.circle(imgMask, left_eye, int(eye_radius), (255,255,255), -1)
    cv2.circle(imgMask, right_eye, int(eye_radius), (255,255,255), -1)

    cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (255,155,255),
              2)

    cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

    eye_class = np.zeros(len(class_name), np.float)

    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                eye_class[find_class(imgHSV[y,x])] +=1 

    main_color_index = np.argmax(eye_class[:len(eye_class)-1])
    total_vote = eye_class.sum()

    #print("\n\nDominant Eye Color: ", class_name[main_color_index])
    '''print("\n **Eyes Color Percentage **")
    for i in range(len(class_name)):
        print(class_name[i], ": ", round(eye_class[i]/total_vote*100, 2), "%")'''
    
    label = 'Dominant Eye Color: %s' % class_name[main_color_index]  
    cv2.putText(image, label, (left_eye[0]-10, left_eye[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,255,0))
    cv2.imshow('EYE-COLOR-DETECTION', image)
    return class_name[main_color_index]

if __name__ == '__main__':
    locate = 'C:/Users/chuch/Desktop/LAIAIAI/eye_color'
    df = pd.read_csv(locate+'/person.csv')
    total = 230
    img_paths = df['ID'][:total]
    eyes = df['eyes'][:total]
    correct=0
    # Reading the image 
    for img_path,eye in zip(img_paths,eyes):
        try:
            imageName = img_path+'.jpg'
            image = io.imread(locate+'/dataset/'+imageName)
            #image = cv2.imread(imageInput, cv2.IMREAD_COLOR)
            image = imutils.resize(image, width=1000)
            # detect color percentage
            result = eye_color(image)
            print(result)
            #cv2.imwrite('result.jpg', image)    
            #cv2.waitKey(0)
            if result == eye:
                correct+=1
        except:
            total-=1
            continue
    print('Accuracy = ',float(correct/total))

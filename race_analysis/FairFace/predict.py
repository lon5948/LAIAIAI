#using cnn_face_detector
from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import argparse
import dlib


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def detect_face(image_path,  SAVE_DETECTED_AT, default_max_size=800,size = 300, padding = 0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    img = dlib.load_rgb_image(image_path)
    old_height, old_width, _ = img.shape
    
    img = dlib.load_rgb_image(image_path)

    old_height, old_width, _ = img.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
    img = dlib.resize_image(img, rows=new_height, cols=new_width)

    dets = cnn_face_detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(image_path))
    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        rect = detection.rect
        faces.append(sp(img, rect))
        
    images = dlib.get_face_chips(img, faces, size=size, padding = padding)

    for idx, image in enumerate(images):
        img_name = image_path.split("/")[-1]
        path_sp = img_name.split(".")
        #face_name = os.path.join(SAVE_DETECTED_AT,  path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
        face_name = path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1]
        dlib.save_image(image, face_name)
        #print(face_name)
        return face_name
    


def predidct_age_gender_race(img_name):
    #img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(img_names)
    '''
    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt', map_location=torch.device('cpu')))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()
    '''
    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_4_20190809.pt', map_location=torch.device('cpu')))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    #race_scores_fair = []
    #race_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []


    #face_names.append(img_name)
    image = dlib.load_rgb_image(img_name)
    image = trans(image)
    image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
    image = image.to(device)
    '''
    # fair
    outputs = model_fair_7(image)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.squeeze(outputs)

    race_outputs = outputs[:7]
    race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    race_pred = np.argmax(race_score)
    
    race_scores_fair.append(race_score)
    race_preds_fair.append(race_pred)
    '''

    # fair 4 class
    outputs = model_fair_4(image)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.squeeze(outputs)

    race_outputs = outputs[:4]
    race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    race_pred = np.argmax(race_score)

    race_scores_fair_4.append(race_score)
    race_preds_fair_4.append(race_pred)
    
    if race_pred == 1:
        print('Black')
    elif race_pred == 2:
        print('Asian')
    elif race_pred == 3:
        print('Indian')
    else:
        print('White')              
    

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)




if __name__ == "__main__":
    #Please create a csv with one column 'img_path', contains the full paths of all images to be analyzed.
    #Also please change working directory to this file.
    SAVE_DETECTED_AT = "detected_faces"
    img = "race_Asian.jpg";
    img_name = detect_face(img, SAVE_DETECTED_AT)
    predidct_age_gender_race(img_name)


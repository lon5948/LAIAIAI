from __future__ import print_function
import argparse
import os
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
# we are only going to use 4 attributes
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1
def extract_features(img_path):
    """Exctract 128 dimensional features
    """
    X_img = face_recognition.load_image_file(img_path)
    locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE, model = "cnn")
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    return face_encodings, locs

def predict_one_image(img_path, clf, labels):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path)
    if not face_encodings:
        return None, None
    pred = pd.DataFrame(clf.predict_proba(face_encodings),
                        columns = labels)
    pred = pred.loc[:, COLS]
    return pred, locs
def draw_attributes(img_path, df):
    """Write bounding boxes and predicted face attributes on the image
    """

    for row in df.iterrows():
        top, right, bottom, left = row[1][4:].astype(int)

        race = np.argmax(row[1][1:4])
        text_showed = "{}".format(race)

        if text_showed == '0':
            print('Asian')
        elif text_showed == '2':
            print('Black')
        else:
            print('White')

def main():
    img_path = 'race_White_face0.jpg'
    model_path = "facemodelNewVersion.pickle"

    # load the model    
    with open(model_path, "rb") as f:
        clf, labels = pickle.load(f, encoding="latin1")
    
    pred, locs = predict_one_image(img_path, clf, labels)
    locs = pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
    df = pd.concat([pred, locs], axis=1)
    draw_attributes(img_path, df)

    

if __name__ == "__main__":
    main()

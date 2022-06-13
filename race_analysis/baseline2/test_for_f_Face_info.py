import cv2
import numpy as np
import face_recognition
from race_detection import f_my_race
from my_face_recognition import f_main



race_detector = f_my_race.Race_Model()
rec_face = f_main.rec()
#----------------------------------------------



def get_face_info(im):
    # face detection
    boxes_face = face_recognition.face_locations(im)
    out = []
    if len(boxes_face)!=0:
        for box_face in boxes_face:
            # segmento rostro
            box_face_fc = box_face
            x0,y1,x1,y0 = box_face
            box_face = np.array([y0,x0,y1,x1])
            face_features = {
                "race":[],
                "bbx_frontal_face":box_face#coefficient of face             
            } 

            face_image = im[x0:x1,y0:y1]
            face_features["race"] = race_detector.predict_race(face_image)
            # -------------------------------------- out ---------------------------------------       
            out.append(face_features)
    else:
        face_features = {
            "race":[],
            "bbx_frontal_face":[]             
        }
        out.append(face_features)
    return out



def bounding_box(out,img):
    for data_face in out:
        box = data_face["bbx_frontal_face"]#coefficient of face
        if len(box) == 0:
            continue
        else:
            try:
                print(data_face["race"])
            except:
                pass
            



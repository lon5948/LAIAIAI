# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 00:24:52 2022

@author: Homeuser
"""
import f_Face_info
import cv2
import os
import pandas as pd

a='dataset/dataset'
"""
for filename in os.listdir(r"./"+a):
    frame=cv2.imread(a+"/"+filename)
    out = f_Face_info.get_face_info(frame)
    res_img = f_Face_info.bounding_box(out,frame)
    """
frame = cv2.imread("data_test/dog.jpg")
out = f_Face_info.get_face_info(frame)
res_img = f_Face_info.bounding_box(out,frame)

"""
df=pd.read_csv('dataset\dataset\person.csv')
total=df.shape[0]

img_paths=df['ID'][:200]
sex=df['sex'][:200]

num=0

for img_paths,sex in zip(img_paths,sex):
    img=img_paths+'.jpg'
    img=os.path.join('dataset',img)
    
    frame=cv2.imread(img)
    out = f_Face_info.get_face_info(frame)
    res_img = f_Face_info.bounding_box(out,frame)
    
    if res_img==sex:
        num+=1
        
print('Acuurancy:',float(num/200))
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 00:24:52 2022

@author: Homeuser
"""
import gender.gender3.f_Face_info
import cv2

def gendermain(imagepath):
    frame = cv2.imread(imagepath)
    out = f_Face_info.get_face_info(frame)
    res_img = f_Face_info.bounding_box(out,frame)
    return res_img
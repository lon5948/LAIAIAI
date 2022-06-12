# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 00:24:52 2022

@author: Homeuser
"""
import f_Face_info
import cv2


frame = cv2.imread("data_test/3.jpg")
out = f_Face_info.get_face_info(frame)
res_img = f_Face_info.bounding_box(out,frame)
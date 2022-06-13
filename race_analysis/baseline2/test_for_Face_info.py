import test_for_f_Face_info
import cv2


frame = cv2.imread("race_Asian_face0.jpg")
out = test_for_f_Face_info.get_face_info(frame)
res_img = test_for_f_Face_info.bounding_box(out,frame)

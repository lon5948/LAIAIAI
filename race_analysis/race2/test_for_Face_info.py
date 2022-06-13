<<<<<<< HEAD
import test_for_f_Face_info
import cv2


frame = cv2.imread("race_Asian_face0.jpg")
out = test_for_f_Face_info.get_face_info(frame)
res_img = test_for_f_Face_info.bounding_box(out,frame)
=======
from race_analysis.race2.test_for_f_Face_info import get_face_info,bounding_box
import cv2

def color(imagepath):
    frame = cv2.imread(imagepath)
    out = get_face_info(frame)
    skincolor = bounding_box(out,frame)
    return skincolor
>>>>>>> 8a8112f87734651ff1b98b4acf9eaa0eb66724ed

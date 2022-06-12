from race_analysis.race2.test_for_f_Face_info import get_face_info,bounding_box
import cv2

def color(imagepath):
    frame = cv2.imread(imagepath)
    out = get_face_info(frame)
    skincolor = bounding_box(out,frame)
    return skincolor
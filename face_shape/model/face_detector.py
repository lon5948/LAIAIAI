import cv2  
import dlib  
from imutils import face_utils
import joblib

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('face_shape/shape_predictor_68_face_landmarks.dat')
clsfr=joblib.load('face_shape/model/KNN_model.sav')

def predict_face_type(points):
    label_dict={0:'diamond',1:'oblong',2:'oval',3:'round',4:'square',5:'triangle'}
    my_points=points[2:9,0]
    D1=my_points[6]-my_points[0]
    D2=my_points[6]-my_points[1]
    D3=my_points[6]-my_points[2]
    D4=my_points[6]-my_points[3]
    D5=my_points[6]-my_points[4]
    D6=my_points[6]-my_points[5]

    d1=D2/float(D1)*100
    d2=D3/float(D1)*100
    d3=D4/float(D1)*100
    d4=D5/float(D1)*100
    d5=D6/float(D1)*100

    result=clsfr.predict([[d1,d2,d3,d4,d5]])
    label=result[0]
    return label_dict[label]

def shapemain(imagepath):
    img = cv2.imread(imagepath)
    #img is a single frame (RGB) captured by the camera
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img will be converted into a gray image
    rect=face_detector(gray)
    (height,width)=img.shape[0:2]

    try:
        points=landmark_detector(gray,rect[0])
        #getting the 68 points of the face
        points=face_utils.shape_to_np(points)
        #converting the points into a numpy array
        count=1
        for p in points:
            cen = (p[0],p[1])
            cv2.circle(img,cen,2,(0,255,0),-1)
            cv2.putText(img,str(count),cen,cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255),1)
            count = count+1
            face = predict_face_type(points)
        cv2.imwrite('output.jpg',img)
        return face
        
    except Exception as e:
        pass

#shapemain('./test1.jpg')


import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

model = load_model('detectfacemodel.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

labels_dict = { 
    0: 'man',
    1 : 'woman',
}
'''
a='data_test'

for filename in os.listdir(r"./"+a):
    img=cv2.imread(a+"/"+filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img,scaleFactor=1.1,
    		minNeighbors=5,
    		minSize=(30, 30)
            )
    print(len(faces))
    for i in range(len(faces)) :
        (x,y,w,h) = faces[i]
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # roi_gray = gray[y:y+h, x:x+w]
        resized = img[y:y+w, x:x+h]
        resized = cv2.resize(resized,(128,128))
        test_now = np.expand_dims(resized,axis=0)
        test_now = test_now/255
        #result = model.predict_classes(test_now) 
        result = (model.predict(test_now)>0.5).astype("int32")
        #predict_x=model.predict(test_now)
        #result=np.argmax(predict_x,axis=1)
        cv2.putText(
              img, labels_dict[result[0][0]], 
              (x, y-10),
              cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)  
        print(labels_dict[result[0][0]])
   '''
img = cv2.imread("data_test/dog.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img,scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
        )
print(len(faces))
for i in range(len(faces)) :
    (x,y,w,h) = faces[i]
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # roi_gray = gray[y:y+h, x:x+w]
    resized = img[y:y+w, x:x+h]
    resized = cv2.resize(resized,(128,128))
    test_now = np.expand_dims(resized,axis=0)
    test_now = test_now/255
    #result = model.predict_classes(test_now) 
    result = (model.predict(test_now)>0.5).astype("int32")
    #predict_x=model.predict(test_now)
    #result=np.argmax(predict_x,axis=1)
    cv2.putText(
          img, labels_dict[result[0][0]], 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)  
    print(labels_dict[result[0][0]])

'''
img= cv2.resize(img, (960,540))
cv2.imshow('Gender Detector',img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
'''
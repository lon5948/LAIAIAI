from PIL import Image, ImageDraw
import os
import face_recognition
import math

def shape(chin):
    a = math.sqrt(abs(chin[6][0]-chin[6][0])**2+abs(chin[6][1]-chin[8][1])**2)
    b = math.sqrt(abs(chin[8][0]-chin[10][0])**2+abs(chin[8][1]-chin[10][1])**2)
    c = math.sqrt(abs(chin[6][0]-chin[10][0])**2+abs(chin[6][1]-chin[10][1])**2)
    cos = (a**2+b**2-c**2)/(2*a*b)
    angle1 = math.acos(cos)*180/math.pi
    if angle1 > 150:
        print('sqrare')
        return('square')
    a = math.sqrt(abs(chin[4][0]-chin[8][0])**2+abs(chin[4][1]-chin[8][1])**2)
    b = math.sqrt(abs(chin[8][0]-chin[12][0])**2+abs(chin[8][1]-chin[12][1])**2)
    c = math.sqrt(abs(chin[4][0]-chin[12][0])**2+abs(chin[4][1]-chin[12][1])**2)
    cos = (a**2+b**2-c**2)/(2*a*b)
    angle2 = math.acos(cos)*180/math.pi
    print(angle2)
    if angle1 < 100:
        print('triangle')
        return('triangle')
    else:
        print('round')
        return('round')

image = face_recognition.load_image_file('./test2.jpeg')
face_landmarks = face_recognition.face_landmarks(image)[0]

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)
d.line(face_landmarks['chin'], width=10)
pil_image.save('output2.jpg')

face = shape(face_landmarks['chin'])
print(face)




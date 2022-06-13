from PIL import Image, ImageDraw
import os
import face_recognition
import math

def shape(chin):
    a = math.sqrt(abs(chin[7][0]-chin[8][0])**2+abs(chin[7][1]-chin[8][1])**2)
    b = math.sqrt(abs(chin[8][0]-chin[9][0])**2+abs(chin[8][1]-chin[9][1])**2)
    c = math.sqrt(abs(chin[7][0]-chin[9][0])**2+abs(chin[7][1]-chin[9][1])**2)
    cos = (a**2+b**2-c**2)/(2*a*b)
    angle1 = math.acos(cos)*180/math.pi
    if angle1 > 163:
        return('square')
    a = math.sqrt(abs(chin[4][0]-chin[8][0])**2+abs(chin[4][1]-chin[8][1])**2)
    b = math.sqrt(abs(chin[8][0]-chin[12][0])**2+abs(chin[8][1]-chin[12][1])**2)
    c = math.sqrt(abs(chin[4][0]-chin[12][0])**2+abs(chin[4][1]-chin[12][1])**2)
    cos = (a**2+b**2-c**2)/(2*a*b)
    angle2 = math.acos(cos)*180/math.pi
    if angle2 < 105:
        return('triangle')
    else:
        return('round')

hit = 0
index = 0
length = len(os.listdir('C:/Users/user/LAIAIAI/face_shape/datasets/round'))
for filename in os.listdir('C:/Users/user/LAIAIAI/face_shape/datasets/round'):
    index += 1
    if index == 10:
        break
    image = face_recognition.load_image_file(os.path.join('C:/Users/user/LAIAIAI/face_shape/datasets/round',filename))
    face_landmarks = face_recognition.face_landmarks(image)[0]
    '''
    # draw the line of the chin
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    d.line(face_landmarks['chin'], width=10)
    pil_image.save('output2.jpg')
    '''
    face = shape(face_landmarks['chin'])
    if face == 'round':
        hit += 1

print('length:',length)
print('accuracy: ',hit/length)




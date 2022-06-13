from face_shape.model.face_detector import shapemain
from gender.gender3.model_main import gendermain
from race_analysis.MLP_face_classification.pred import racemain
from eye_color.model.test import eye

#imagepath = input("Enter your image path")
imagepath = './test2.jpeg'
gender = gendermain(imagepath)
faceshape = shapemain(imagepath)
race = racemain(imagepath)
eyecolor = eye(imagepath)

print('------------')
print('gender: ',gender)
print('shape:  ',faceshape)
print('race:   ',race)
print('eyecolor',eyecolor)
print('------------')
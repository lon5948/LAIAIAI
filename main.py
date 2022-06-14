from face_shape.model.face_detector import shapemain
from gender.model.model_main import gendermain
from race_analysis.model.pred import racemain
from eye_color.model.test import eye

imagepath = input("\n\n\nEnter your image path: ")
gender = gendermain(imagepath)
faceshape = shapemain(imagepath)
race = racemain(imagepath)
eyecolor = eye(imagepath)
print('\n\n\n-------HERE IS THE OUTPUT-------\n')
print('     gender:   ',gender)
print('     shape:    ',faceshape)
print('     race:     ',race)
print('     eyecolor: ',eyecolor)
print('\n--------------------------------\n\n\n')
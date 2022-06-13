from face_shape.model.face_detector import shapemain
#from gender.model.model_main import gendermain
from race_analysis.model.pred import racemain
from eye_color.model.test import eye

#imagepath = input("Enter your image path")
imagepath = './test2.jpeg'
#gender = gendermain(imagepath)
#faceshape = shapemain(imagepath)
#race = racemain(imagepath)
eyecolor = eye(imagepath)

print('-------HERE IS THE OUTPUT-----')
#print('gender:   ',gender)
#print('shape:    ',faceshape)
#print('race:     ',race)
print('eyecolor: ',eyecolor)
print('------------------------------')
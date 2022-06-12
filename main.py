from face_shape.model.face_detector import shape
from gender_model import gender
from face_shape.model.face_detector import color
from face_shape.model.face_detector import eye

#imagepath = input("Enter your image path")
imagepath = './test2.jpeg'
gen = gender(imagepath)
faceshape = shape(imagepath)
skincolor = color(imagepath)
eyecolor = eye(imagepath)

print('gender: ',gen)
print('shape: ',faceshape)
print('shape: ',skincolor)
print('eyecolor',eyecolor)
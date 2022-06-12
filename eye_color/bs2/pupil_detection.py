''' Pupil Detection '''
#--------------------------------
# Date : 23-06-2020
# Project : Pupil Detection
# Category : Detection
# Company : weblineindia
# Department : AI/ML
#--------------------------------
import os
import numpy as np
import cv2
from skimage import io
import imutils
import argparse
import dlib



def pixelReader(img,startHorizontal,startVertical,height):
	'''
	Used to read specific pixel of given image.
	img = image 
	startHorizontal =  horizontal starting pixel value
	startVertical = vertical starting value
	height = maximum limit for pixel reading
	'''
	# list for satisfied pixels
	blackColour = []

	# for loops for traversing pixels
	for j in range(-int(height*1.5), 0):
		for i in range(startVertical- height,startVertical+int(height*1.5)):
			# setting lower bound for pixels, termination 
			blackLowerRange = [80, 50, 50]
			pixel = startHorizontal + j
			colorCI = img[int(pixel), i]
			if ((colorCI[0] <= blackLowerRange[0] and colorCI[1] <= blackLowerRange[1] and colorCI[2] <= blackLowerRange[2])):
				blackColour.append([int(pixel), i])
	return blackColour

def getFaceAttributeVector(image):
	'''
	Used to get the 68 facial attributes of face image.
	image: it is an image.
	'''
	# dlib shape predictor 
	predictorPath = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictorPath)
	dets = detector(image)

	# if dlib is able to detect the faces 
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	for k, d in enumerate(dets):
		shape = predictor(image, d)

	# to get the 68 facial points detect from dlib into a list
	faceCoord = np.empty([68, 2], dtype = int)

	for b in range(68):
		faceCoord[b][0] = shape.part(b).x
		faceCoord[b][1] = shape.part(b).y

	return faceCoord

def getEyeCoordinates(image, faceCoord):
	'''
	Used to crop the eyes from image
	image: it is an image.
	faceCoord: array of facial landmarks coordinates
	'''
	# using facial points to crop the part of the eyes from image
	leftEye = image[int(faceCoord[19][1]):int(faceCoord[42][1]),int(faceCoord[36][0]):int(faceCoord[39][0])]
	rightEye = image[int(faceCoord[19][1]):int(faceCoord[42][1]),int(faceCoord[42][0]):int(faceCoord[45][0])]

	eyeLCoordinate = [int(faceCoord[37][0]+int((faceCoord[38][0]-faceCoord[37][0])/2)), int(faceCoord[38][1]+int((faceCoord[40][1]-faceCoord[38][1])/2))]
	eyeRCoordinate = [int(faceCoord[43][0]+int((faceCoord[44][0]-faceCoord[43][0])/2)), int(faceCoord[43][1]+int((faceCoord[47][1]-faceCoord[43][1])/2))]

	leftBlackPixel = pixelReader(image,eyeLCoordinate[1],eyeLCoordinate[0],int((faceCoord[38][0]-faceCoord[37][0])/2))
	rightBlackPixel = pixelReader(image, eyeRCoordinate[1], eyeRCoordinate[0], int((faceCoord[44][0]-faceCoord[43][0])/2))
	return leftEye, rightEye, leftBlackPixel, rightBlackPixel

def getPupilPoint(img, blackCoordinates, eyeTopPointX, eyeBottomPointY):
	'''
	Used to get the coordinates of the pupil 
	image: cropped eye image
	blackCoordinates: It is the array of the black color pixels inside the eyes
	eyeTopPointX: eye's starting coordinates horizontally
	eyeBottomPointY: eye's starting coordinates vertically
	'''
	# after getting the eyes pixels applying cv2 method to detect circle pixels which we can do using houghcircles 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT, dp = 1,minDist = 5,
		param1=250,param2=10,minRadius=1,maxRadius=-1)
	
	# check if houghcircles has detect any circle or not
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:1]:
			pupilPoint = [int(eyeTopPointX[0])+ i[0],int(eyeBottomPointY[1]) + i[1]]
	else:
		# if HoughCircles is unable to detect the circle than using eyes points to get pupil points 
		a = 0
		for j,k in blackCoordinates:
			if a == int(len(blackCoordinates)/2):
				pupilPoint = [k,j]
			a += 1
	return pupilPoint

# define HSV color ranges for eyes colors
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0] : ((166, 21, 50), (240, 100, 85)),
    class_name[1] : ((166, 2, 25), (300, 20, 75)),
    class_name[2] : ((2, 20, 20), (40, 100, 60)),
    class_name[3] : ((20, 3, 30), (65, 60, 60)),
    class_name[4] : ((0, 10, 5), (40, 40, 25)),
    class_name[5] : ((60, 21, 50), (165, 100, 85)),
    class_name[6] : ((60, 2, 25), (165, 20, 65))
}

def check_color(hsv, color):
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and \
    hsv[1] <= color[1][1] and (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False

# define eye color category rules in HSV space
def find_class(hsv):
    color_id = 7
    for i in range(len(class_name)-1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            color_id = i

    return color_id

def eye_color(image,left_eye,right_eye):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))
    


    eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
    eye_radius = eye_distance/15 # approximate
   
    cv2.circle(imgMask, left_eye, int(eye_radius), (255,255,255), -1)
    cv2.circle(imgMask, right_eye, int(eye_radius), (255,255,255), -1)

    cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

    eye_class = np.zeros(len(class_name), np.float)

    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                eye_class[find_class(imgHSV[y,x])] +=1 

    main_color_index = np.argmax(eye_class[:len(eye_class)-1])
    total_vote = eye_class.sum()

    #print("\n\nDominant Eye Color: ", class_name[main_color_index])
    '''print("\n **Eyes Color Percentage **")
    for i in range(len(class_name)):
        print(class_name[i], ": ", round(eye_class[i]/total_vote*100, 2), "%")'''
    
    label = 'Dominant Eye Color: %s' % class_name[main_color_index]  
    cv2.putText(image, label, (left_eye[0]-10, left_eye[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,255,0))
    cv2.imshow('EYE-COLOR-DETECTION', image)
    return class_name[main_color_index]

# Reading the image 
#for imageName in os.listdir(args["image"]):
image = io.imread("test2.jpeg")
image = imutils.resize(image, width=1000)
# getting the faceattribute vector from dlib
faceVector = getFaceAttributeVector(image)
# getting the eye points
leftEye, rightEye, eyeLeftBlackPixels, eyeRightBlackPixels = getEyeCoordinates(image, faceVector)
# getting pupilpoint for left eye
# leftEye is the cropped part of face image
leftEyeCoord, eyeBrowCoord = faceVector[36], faceVector[19]
leftPupilPoint = getPupilPoint(leftEye, eyeLeftBlackPixels, leftEyeCoord, eyeBrowCoord)
# getting pupilpoint for right eye
# rightEye is the cropped part of face image
rightEyeCoord = faceVector[42]
rightPupilPoint = getPupilPoint(rightEye, eyeRightBlackPixels, rightEyeCoord, eyeBrowCoord)


# drawing pupil points on image
cv2.circle(image, tuple(leftPupilPoint), 5, (255, 0, 0), -1)
cv2.circle(image, tuple(rightPupilPoint), 5, (255, 0, 0), -1)
finalImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

print(eye_color(finalImage,leftPupilPoint,rightPupilPoint))
cv2.imwrite("result.jpg",finalImage)
cv2.waitKey(0)
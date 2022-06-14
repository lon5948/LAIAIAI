## Introduction

## Input/Output
#### Input      
> imagepath   

#### Output    
> **gender：** Man / Woman    
> **face shape：** diamond / oblong / oval / round / square / triangle     
> **race：** Asian / Indian / Black / White / Middle eastern / Latino hispanic    
> **eye color：** Black / Brown / Green / Blue  

## Implementation

#### Gender
> bs1： 
> When you run the script for the first time, it will download the pre-trained model from this [link](https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model) and place it under pre-trained
> directory in the current path.

> gender3：   
> Download the pretrained models from [here](https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk) and save it in the same folder as where f_my_gender.py 
> is located.

#### Face Shape

| test image  | baseline 1 | baseline 2 | KNN model |
| ------------- | ------------- | ------------- | ------------- |
| <img src="https://github.com/lon5948/LAIAIAI/blob/main/face_shape/baseline1/test1.jpg" width="250" height="280">  | <img src="https://github.com/lon5948/LAIAIAI/blob/main/face_shape/baseline1/output.jpg" width="250" height="280">  | <img src="https://github.com/lon5948/LAIAIAI/blob/main/face_shape/baseline2/output.jpg" width="250" height="280"> | <img src="https://github.com/lon5948/LAIAIAI/blob/main/face_shape/model/output.jpg" width="250" height="280">
 

#### Eye Color
| test image  | baseline 1 | baseline 2 | Iris extraction |
| ------------- | ------------- | ------------- | ------------- |
| <img src="https://github.com/lon5948/LAIAIAI/blob/main/eye_color/baseline1/test2.jpeg" width="250" height="280">  | <img src="https://github.com/lon5948/LAIAIAI/blob/main/eye_color/baseline1/result.jpg" width="250" height="280">  | <img src="https://github.com/lon5948/LAIAIAI/blob/main/eye_color/bs2/result.jpg" width="250" height="280"> | <img src="https://github.com/lon5948/LAIAIAI/blob/main/eye_color/model/results/test2.jpeg" width="250" height="280">

#### Race Analysis
> FairFace：      
> Download the pretrained models from [here](https://drive.google.com/file/d/1n7L6mZjf9JeZqDiUL8SvdqY_kJeefhzO/view?usp=sharing) and save it in the same folder as 
> where predict.py is located. Two models are included, race_4 model predicts race as White, Black, Asian and Indian.

> race 2：      
> Download the pretrained models from [here](https://drive.google.com/file/d/1aJYpSF34_G-Hybrq6HRKDQ6FVjn2ZGzq/view?usp=sharing) and save it in the same folder as 
> where test_for_Face_info.py is located. 

## Accuracy
> check if output = label or not
#### Gender
#### Face Shape   
|   | baseline 1 | baseline 2 | model |
| ------------- | ------------- | ------------- | ------------- |
| square | 47.3 % | 47 % | 93 % |
| oblong | X | 56 % | 91 % |
|  oval | X | 58 % | 81 % |
| round | 2.51 % | 69 % | 88 % |
| diamond | X | 35 % | 98 % |
| triangle | 45 % | 30 % | 30 % |     
#### Eye Color
#### Race Analysis

## Installation
```
pip install tensorflow>=1.12.1
pip install mtcnn
pip install numpy
pip install imutils
pip install scikit-image
pip install argparse
pip install opencv-python
pip install opencv-contrib-python
pip install dlib
pip install pandas
pip install face_recognition
pip install pickle
pip install sklearn
pip install torch
pip install torchvision
pip install os
pip install deepface
pip install natsort
pip install face_alignment
```

## Run
```
  $  cd LAIAIAI
```
```
  $  python main.py
```
```
  $  [enter your image path]
```

## Datasets

1. Datasets for testing：[download](https://drive.google.com/file/d/1NcbXiu5LVI8T_QC9_SduTqQVMMae1oZ0/view)    
  
2. Dataset of Face Shape：[download](https://drive.google.com/file/d/1K5MkBs9EVuNA8isQR_3fJr_84TIFJOT7/view)

## Reference
###### gender
* [baseline1](https://github.com/arunponnusamy/gender-detection-keras)
* [baseline2](https://github.com/nman7/Facial-Gender-classification)
* [Model](https://github.com/juan-csv/Face_info)

###### face shape
* [baseline1](https://github.com/ageitgey/face_recognition)
* [baseline2](https://github.com/BhagyaPerera/face-shape-recognition-system-)
* [KNN model](https://github.com/AakashBelide/gesture_control_and_mood_detection)
* [paper](https://www.researchgate.net/publication/337386649_Face_shape_classification_using_Inception_v3)   
###### race
* [baseline1](https://github.com/dchen236/FairFace)
* [baseline2-1](https://github.com/juan-csv/Face_info)
* [baseline2-2](https://github.com/serengil/deepface)
* [Model](https://github.com/wondonghyeon/face-classification)
 
###### eye color
* [baseline1](https://github.com/ghimiredhikura/Eye-Color-Detection)
* [baseline2](https://github.com/weblineindia/AIML-Pupil-Detection)
* [Model](https://github.com/vanquish630/ExtractIris)

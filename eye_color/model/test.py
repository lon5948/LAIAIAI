import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import pickle
from pylab import *
import time
import eye_color.model.utils.segment
from eye_color.model.utils.logger import setup_logger
from eye_color.model.utils import util
from eye_color.model.utils.iris import histMatchIris,makeIris,combineIris
import imutils
import pandas as pd
from skimage import io



def parse_args():
  """Parses arguments."""
  
  parser = argparse.ArgumentParser()

  parser.add_argument('--test_dir', type=str, default = './test_data',
                      help='directory of images to invert.')

 
  parser.add_argument('-o', '--output_dir', type=str, default='./results',
                      help='Directory to save the results. If not specified, '
                           '`./results/'
                           'will be used by default.')

  parser.add_argument('--pretrained_dir', type=str, default = './pretrained_models',
                      help='Directory tof pretraied models. If not specified, '
                           '`./pretrained_models/'
                           'will be used by default.')
  
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  
  return parser.parse_args()



def test():
     args = parse_args()
     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



     output_dir = args.output_dir
     if not os.path.exists(output_dir):
          os.makedirs(output_dir)
     
     iris_green,iris_brown,iris_blue,iris_black,iris_mask = util.returnIrisTemplate()
     #images = os.listdir(args.test_dir)
     locate = 'eye_color'
     df = pd.read_csv(locate+'/person.csv')
     total = 230
     img_paths = df['ID'][:total]
     eyes = df['eyes'][:total]
     correct=0
     # Reading the image 
     for img_path,eye in zip(img_paths,eyes):
          try:
               imageName = img_path+'.jpg'
               #image = io.imread(locate+'/dataset/'+imageName)

               image = Image.open(os.path.join(locate+'/dataset',imageName))
               eyeWholeL, eyeCenterL,predIrisL = histMatchIris(image,eye_left_right=5)

               if eyeCenterL is None:
                    irisL = None
                    resultL='None'
               else:
                    coloursL = util.majorColors(eyeCenterL)
                    #print(coloursL)
               #  print(coloursL[1].shape)
               #  print(predIrisL.shape)
                    irisL,result = makeIris(coloursL[1],predIrisL, iris_brown , iris_blue, iris_green,iris_black)
                    resultL=result
               eyeWholeR, eyeCenterR,predIrisR = histMatchIris(image,eye_left_right=4)

               if eyeCenterR is None:
                    irisR = None
                    resultR='None'
               else:
                    coloursR = util.majorColors(eyeCenterR)
                    #print(coloursR)
                    irisR,result = makeIris(coloursR[1],predIrisR, iris_brown , iris_blue, iris_green,iris_black)
                    resultR=result
               
               if(resultL=='None' and resultR=='None'):
                    total-=1
                    continue
               elif resultL!='None':
                    result = resultL
               else:
                    result = resultR
               print(result)
               if result == eye:
                    correct+=1
               '''iris = combineIris(irisL,irisR)
               if iris is None:
                    print("No iris found in image.")
               else:
                    #print(iris.shape)
                    plt.imsave(os.path.join('results',img),iris)'''
          except:
               total-=1
               continue
     print('Accuracy = ',float(correct/total))
               #print(irisL,irisR)

               
               
          #print(resultL)
          #print(resultR


def eye(imageName):
     image = Image.open(imageName)
     eyeWholeL, eyeCenterL,predIrisL = histMatchIris(image,eye_left_right=5)
     iris_green,iris_brown,iris_blue,iris_black,iris_mask = util.returnIrisTemplate()
     if eyeCenterL is None:
          irisL = None
          resultL='None'
     else:
          coloursL = util.majorColors(eyeCenterL)
          irisL,result = makeIris(coloursL[1],predIrisL, iris_brown , iris_blue, iris_green,iris_black)
          resultL=result
     eyeWholeR, eyeCenterR,predIrisR = histMatchIris(image,eye_left_right=4)

     if eyeCenterR is None:
               irisR = None
               resultR='None'
     else:
          coloursR = util.majorColors(eyeCenterR)
          #print(coloursR)
          irisR,result = makeIris(coloursR[1],predIrisR, iris_brown , iris_blue, iris_green,iris_black)
          resultR=result
     if(resultL=='None' and resultR=='None'):
          total-=1
          print('Not fouond')
     elif resultL!='None':
          result = resultL
     else:
          result = resultR
     return result













    



  
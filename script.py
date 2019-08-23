
import cv2
import os
import numpy as np
#from PIL import Image
#from keras_vggface.vggface import VGGFace
#import random
#from tqdm import tqdm
#from matplotlib import pyplot as plt
#from math import floor
#import seaborn as sns
#import random
#from scipy import ndarray
#import skimage as sk
#from skimage import transform
#from skimage import util
#import warnings
#warnings.filterwarnings('ignore')
#from keras.preprocessing.image import ImageDataGenerator
#from tqdm import tqdm
from keras.layers import Input,Conv2D,Dense, Dropout, BatchNormalization, MaxPooling2D, Activation, Flatten, AvgPool2D
from keras.layers import  BatchNormalization as btn
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
#from keras.callbacks import LearningRateScheduler
#from IPython.display import HTML
#import base64
#from scipy.ndimage.interpolation import shift
from keras.optimizers import Adam
#from xgboost import XGBClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #### FOR FACE DETECTION
def detect(gray, frame):
    """
    
    THIS FUNCTION DETECTS THE FACE IN GREY IMAGE AND CROP IT THEN PREDICT ITS  FACE FEATURE
    
    PAPAMETER:
    GREY:  np array; THE GREY SCALE OF IMAGE
    FRAME: np array; ACTUAL IMAGE
    
    RETURNS:
    FACE_FEATURE: np.array; PREDICTED ARRAY OF SIZE (1,2048)
    FRAME_CROP: np array; CROPED IMAGE
    
    """
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    try:
        x, y, w, h = faces[0]
        frame_crop = frame[x:x+w,y:y+h]
        
        
        return frame_crop
    except:
        return None

     
def fetch_image():
    path = "Images/"
    index = -1
    photos = os.listdir("Images/")
    
    
    img = cv2.imread(path + photos[index])
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    frame = cv2.resize(detect(grey,img), (224,224))
    return frame
frame = fetch_image()
ras_model = ResNet50(include_top=False,weights= 'resnet.h5',input_shape=(224,224,3))
ras_model.trainable = False
f1 = Flatten()(ras_model.output)
f4 = Dense(512)(f1)
f4 = Activation('relu')(f4)
f2 = Dense(1)(f4)
out = Activation('sigmoid')(f2)

model1 = Model(inputs = ras_model.input,outputs = out)
model2 = Model(inputs = ras_model.input,outputs = out)
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.load_weights("model1.h5")
model2.load_weights("model2.h5")

v = model2.predict(frame.reshape(1,224,224,3))
k = model1.predict(frame.reshape(1,224,224,3))

t = str(v)[2]

if t == 1:
    word = "vishal has come to meet you. Say Hi to Him"
else:
    word = "Krishiv has come to meet you. Say Hi to Him"

with open("result/pick.txt","w") as f:
	f.write(word)

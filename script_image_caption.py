import numpy as np
import os
import pandas as pd
#import pickle
#import matplotlib.pyplot as plt
#import random
#from tqdm import tqdm
#from image_processing import processing

from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation,BatchNormalization, RepeatVector,Concatenate
from keras.models import Sequential, Model
#from keras.utils import np_utils
from keras.preprocessing import image, sequence

embedding_size = 128
max_len = 40



image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(BatchNormalization())
image_model.add(RepeatVector(max_len))

#image_model.summary()

language_model = Sequential()

language_model.add(Embedding(input_dim=8254, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(275, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

#language_model.summary()

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(150, return_sequences=True)(conca)
x = LSTM(550, return_sequences=False)(x)
x = Dense(8254)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.load_weights("model/model_weights2.h5")
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
from PIL import Image as pilimage


import json
with open('model/word_in.json') as json_data:
    word_in = json.load(json_data)

with open('model/in_word.json') as json_data:
    in_word = json.load(json_data)

def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word_in[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = in_word[str(np.argmax(preds[0]))]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])
images_path='Images/'
input_image=  os.listdir(images_path)[-1]
filename=images_path+input_image
ras_model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
def processing(image_name,model = None,get_histogram = False,size = None,convert_BGR = True,image_path = 'flickr30k-images/', mean_of_data = [104,117,124], return_arr = False):
       
        
    '''
    Subtracting the dataset mean serves to "center" the data. Additionally, you ideally would like to divide by the sttdev of that
    feature or pixel as well if you want to normalize each feature value to a z-score. 
    
    PARAMETER:
    image_name --> name of the image
    model --> want to extract feature then give the model
    get_histogram --> Boolean getting histogram
    size --> If resize give to tuple in form of (width,height)
    convert_BGR--> convert to BGR Boolen type
    mean)of_data --> to normalize the data in order of the channel
    
    RETURN:
    pred_value if model given.A coloumn array.
    im_arr image array
    '''
    if model == None:
        
        base_model = VGG16(weights='resnet.h5', include_top=True, input_shape=(224,224,3))
        model = Model(base_model.input, base_model.layers[-2].output)
        
    path = image_path + str(image_name)
    with pilimage.open(path) as image:
        if not(size == None):
            image = image.resize(size)
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
        im_arr = np.expand_dims(im_arr, axis=0)
    
    
    
    if get_histogram == True:
        print(image.histogram())

    if convert_BGR == True:
        im_arr = im_arr[ :,:, :, ::-1]
    if not(mean_of_data == None):
        im_arr[ :,:, :, 0] -= mean_of_data[0]
        im_arr[ :,:, :, 1] -= mean_of_data[1]
        im_arr[ :,: ,:, 2] -= mean_of_data[2]
        
    if not(model == None):    
        pred = model.predict(im_arr)
        pred = np.reshape(pred, pred.shape[1])
        if return_arr == False:
            return pred
        if return_arr == True:
            return pred, im_arr
    
    return im_arr
 
pred_test = processing(input_image,size = (224,224),convert_BGR=False,model = ras_model
                      ,mean_of_data=[0,0,0],return_arr=False,image_path=images_path) # local module
Argmax_Search = predict_captions(pred_test)

#Argmax_Search = predict_captions(pred_test)
with open("result/pick.txt","w") as f:
	f.write(Argmax_Search)

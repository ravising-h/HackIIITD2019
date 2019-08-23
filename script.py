import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
#from image_processing import processing

from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation,BatchNormalization, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence

embedding_size = 128
max_len = 40



image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(BatchNormalization())
image_model.add(RepeatVector(max_len))

image_model.summary()

language_model = Sequential()

language_model.add(Embedding(input_dim=8254, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(275, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

language_model.summary()

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(150, return_sequences=True)(conca)
x = LSTM(550, return_sequences=False)(x)
x = Dense(8254)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.load_weights("model/model_weights2.h5")
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()

model.load_weights('model_weights1.h5')
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

#Argmax_Search = predict_captions(pred_test)
with open("result/pick.txt","w")

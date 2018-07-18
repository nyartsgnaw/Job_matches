from keras.layers.convolutional import Conv2DTranspose , Conv1D, Conv2D,Convolution3D, MaxPooling2D,UpSampling1D,UpSampling2D,UpSampling3D
from keras.layers import Input,LSTM,Embedding, Dense, Dropout, Activation, Flatten,   Reshape, Flatten, Lambda
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
import numpy as np 
import pandas as pd
import os

def create_LSTM(input_dim=200,output_dim=200):
    #vocab_size = len(tokenizer.word_index) + 1
    vocab_size = 20444
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=input_dim))
    model.add(LSTM(200))
    model.add(BatchNormalization())
#    model.add(GaussianDropout(0.25))  #https://arxiv.org/pdf/1611.07004v1.pdf
#    model.add(GaussianNoise(0.05))

    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model

if __name__ == '__main__':
    model_id = 'LSTM_1'
    model = create_LSTM()
    print(model.summary())

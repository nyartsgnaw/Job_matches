from keras.layers.convolutional import Conv2DTranspose , Conv1D, Conv2D,Convolution3D, MaxPooling2D,UpSampling1D,UpSampling2D,UpSampling3D
from keras.layers import Input,LSTM,Bidirectional,TimeDistributed,Embedding, Dense, Dropout, Activation, Flatten,   Reshape, Flatten, Lambda
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
import numpy as np 
import pandas as pd
import os

def create_LSTM(input_dim,output_dim,embedding_matrix=[]):
    model = Sequential()
    if embedding_matrix != []:
        embedding_layer = Embedding(embedding_matrix.shape[0],
                                    embedding_matrix.shape[1],
                                    weights=[embedding_matrix],
                                    #input_shape=(input_dim,),
                                    input_length=input_dim,
                                    trainable=False)
        model.add(embedding_layer)
        
        model.add(LSTM(150))
        model.add(Bidirectional(LSTM(150, return_sequences=True)))
    else:
        model.add(LSTM(150,input_shape=(None,input_dim)))
        model.add(Bidirectional(LSTM(150, return_sequences=True, input_shape=(None,input_dim))))

#    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    
    model.add(BatchNormalization())
#    model.add(GaussianDropout(0.25))  #https://arxiv.org/pdf/1611.07004v1.pdf
#    model.add(GaussianNoise(0.05))

    model.add(Activation('relu'))
#    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(output_dim))
#    model.add(TimeDistributed(Dense(output_dim)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    print(model.summary())

    return model

if __name__ == '__main__':
    model_id = 'LSTM_1'
    model = create_LSTM(input_dim=200,output_dim=200,embedding_matrix=[])

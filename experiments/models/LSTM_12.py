from keras.layers.convolutional import Conv2DTranspose , Conv1D, Conv2D,Convolution3D, MaxPooling2D,UpSampling1D,UpSampling2D,UpSampling3D
from keras.layers import merge,Input,LSTM,Bidirectional,TimeDistributed,Embedding, Dense, Dropout, Activation, Flatten,   Reshape, Flatten, Lambda
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Permute,RepeatVector
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model,K
from keras.layers.advanced_activations import LeakyReLU
import numpy as np 
import pandas as pd
import os


def attention_3d_block(inputs,input_dim,is_single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    feature_length = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
#    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if is_single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(feature_length)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def create_LSTM(input_dim,output_dim,time_steps=1,embedding_matrix=[]):
    inputs = Input(shape=(input_dim,time_steps, 1,))
    if embedding_matrix != []:
        embedding_layer = Embedding(embedding_matrix.shape[0],
                                    embedding_matrix.shape[1],
                                    weights=[embedding_matrix],
                                    input_shape=(input_dim,),
                                    trainable=False)
        x = embedding_layer(inputs)
        x = Reshape([input_dim,embedding_matrix.shape[1]])(x)
    else:
        x = Reshape([input_dim,1,])(inputs)
    
#    x = LSTM(200,return_sequences=True)(x)
#    x = Bidirectional(LSTM(128, return_sequences=True))(x)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)

    x = LSTM(200)(x)
#    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    output = Activation('relu')(x)

#    x = attention_3d_block(x,input_dim=input_dim)
#    x = Flatten()(x)
#    x = BatchNormalization()(x)
    x = Dense(256, activation='tanh')(x)
    output = Dense(output_dim, activation='tanh')(x)
    model = Model(input=[inputs], output=output)
    print(model.summary())
    return model


if __name__ == '__main__':
    model_id = 'LSTM_1'
    model = create_LSTM(input_dim=200,output_dim=200,time_steps=1,embedding_matrix=[])

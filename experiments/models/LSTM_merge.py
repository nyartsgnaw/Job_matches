#the file that describe generator,discriminator and tester structures
#generally speaking, a shared tester structure is recommended
from keras.layers.convolutional import Conv2DTranspose , Conv1D, Conv2D,Convolution3D, MaxPooling2D,UpSampling1D,UpSampling2D,UpSampling3D
from keras.layers import Input,Embedding, Dense, Dropout, Activation, Flatten,   Reshape, Flatten, Lambda
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
import numpy as np 
import pandas as pd
import os

def create_LSTM(input_dim=100,output_dim=200):
    #vocab_size = len(tokenizer.word_index) + 1

    embedding_vecor_length = 32
    #description
    model1 = Sequential()
    model1.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_length))
    model1.add(LSTM(100))

    #qualification
    model2 = Sequential()
    model2.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_length))
    model2.add(LSTM(100))

    #responsibility
    model3 = Sequential()
    model3.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_length))
    model3.add(LSTM(100))

    #experience
    model4 = Model(inputs=Input(shape=(1,)), outputs=a)
    #skills

    education = []
    for i in range(len(JD_ls)):
        JD = JD_ls[i]
        education.append(JD['education'])

    edu_types = list(set(sum(education,[])))
    to_categorical(edu_types)
    import pandas as pd
    s = pd.Series(edu_types)
    pd.get_dummies(s)
    #int

    #education level
    #onehot


    model = Sequential()
    model.add(Merge([model1, model2,model3,model4], mode='concat'))
    model.add(Dense(vector_dim))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model

if __name__ == '__main__':
	model_id = 'LSTM_1'
	model = create_LSTM()



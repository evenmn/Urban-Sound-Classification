# Import packages
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 
from keras.utils import np_utils

from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.layers import Dense, Dropout, Activation         # FNN
from keras.layers import GRU, LSTM, Embedding               # RNN
from keras.layers import Conv2D, Flatten, MaxPooling2D      # CNN
from keras.optimizers import Adam


def load_mfcc():
    # Load data set
    X = np.loadtxt('../data/mfcc_40.txt')
    text_file = open('../data/t.txt', 'r')
    t = []
    for line in text_file:
        t.append(line)
        
    # One hot encode
    lb = LabelEncoder()
    t = np_utils.to_categorical(lb.fit_transform(t))

    # Split into training and test set
    N = len(X)
    N_train = int(N*0.8)
    X_train = X[:N_train]
    t_train = t[:N_train]
    X_val = X[N_train:]
    t_val = t[N_train:]
    return X_train, t_train, X_val, t_val
    
    
    
def load_spectrogram():
    # Load data set
    X = np.loadtxt('../data/spectrogram_40.txt')
    text_file = open('../data/t.txt', 'r')
    t = []
    for line in text_file:
        t.append(line)
        
    X = np.reshape(X, (5433, 40, 173, 1))       #CNN needs 4D array as input
        
    # One hot encode
    lb = LabelEncoder()
    t = np_utils.to_categorical(lb.fit_transform(t))

    # Split into training and test set
    N = len(X)
    N_train = int(N*0.8)
    X_train = X[:N_train]
    t_train = t[:N_train]
    X_val = X[N_train:]
    t_val = t[N_train:]
    return X_train, t_train, X_val, t_val
    


# Run a deep learning model and get results

def Logistic():
    X_train, t_train, X_val, t_val = load_mfcc()
    
    num_labels = t_train.shape[1]

    model = Sequential()

    model.add(Dense(num_labels, input_shape=(40,), W_regularizer=l2(1.0)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(X_train, t_train, batch_size=32, epochs=50, validation_data=(X_val, t_val))
    

def FNN(N=1):
    X_train, t_train, X_val, t_val = load_mfcc()

    num_labels = t_train.shape[1]

    model = Sequential()

    model.add(Dense(1024, input_shape=(40,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    for i in range(N-1):
        model.add(Dense(1024))
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(X_train, t_train, batch_size=32, epochs=10000, validation_data=(X_val, t_val))


def Convolutional():
    X_train, t_train, X_val, t_val = load_spectrogram()
    
    num_labels = t_train.shape[1]

    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(40,173,1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.15))
    
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.20))
    model.add(Flatten())
    
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(X_train, t_train, batch_size=64, epochs=1000, validation_data=(X_val, t_val))
    
    
    
def Long_short():
    X_train, t_train, X_val, t_val = load_mfcc()
    
    X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
    X_val = np.reshape(X_val, (len(X_val), len(X_val[0]), 1))
    
    num_labels = t_train.shape[1]

    model = Sequential()
    
    #model.add(Embedding(1000, 512, input_length = X_train.shape[1]))
    
    model.add(LSTM(256,input_shape=(40,1),return_sequences=False))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    #model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, t_train, batch_size=32, epochs=1000, validation_data=(X_val, t_val))
    
    
    
def Gated():
    X_train, t_train, X_val, t_val = load_mfcc()
    
    X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
    X_val = np.reshape(X_val, (len(X_val), len(X_val[0]), 1))
    
    num_labels = t_train.shape[1]

    model = Sequential()
    
    model.add(GRU(256, activation='relu', recurrent_activation='hard_sigmoid'))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    #model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, t_train, batch_size=32, epochs=1000, validation_data=(X_val, t_val))
    

    
if __name__ == '__main__':
    #Logistic()
    #FNN(4)
    #Convolutional()
    #Long_short()
    Gated()

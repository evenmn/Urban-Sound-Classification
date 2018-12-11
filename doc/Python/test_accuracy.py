# Import packages
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation         # FNN
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

    return X, t
    

def FNN(N=1):
    X, t = load_mfcc()

    num_labels = t.shape[1]

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
    model.fit(X, t, batch_size=32, epochs=500)
    
    # Examine test dataset
    X_test = np.loadtxt('../data/mfcc_test_40.txt')
    y = model.predict(X_test)
    
    categories = lb.inverse_transform(np.argmax(y, axis=1))
    for cat in categories:
        print(cat)
        
        
if __name__ == '__main__':
    FNN(4)

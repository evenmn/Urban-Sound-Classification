# Import packages
import numpy as np
import csv

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

    return X, t, lb
    

def FNN(N=1):
    X, t, lb = load_mfcc()

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
    model.fit(X, t, batch_size=32, epochs=1000)
    
    # Examine test dataset
    X_test = np.loadtxt('../data/mfcc_test_40.txt')
    y = model.predict(X_test)
    
    # One hot decoder
    categories = lb.inverse_transform(np.argmax(y, axis=1))
    for i in range(len(categories)):
        categories[i] = categories[i].strip()       # Remove '\n'
    
    with open('../data/test.csv') as f:
        reader = csv.reader(f, delimiter=',')
        
        IDs = []
        
        line = 0
        for row in reader:
            if line != 0:
                IDs.append(int(row[0]))
            line += 1
        IDs = np.array(IDs)
    f.close()

    with open('../data/test.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['ID', 'Class'])
        j = 0
        for i in range(len(IDs)):
            if IDs[i]!=1201 and IDs[i]!=2893 and IDs[i]!=4020 and IDs[i]!=5469 and IDs[i]!=5501 and IDs[i]!=5993 and IDs[i]!=5998:
                writer.writerow([str(IDs[i]), categories[j]])
                j += 1
            else:
                writer.writerow([str(IDs[i]), 'damaged'])
    f.close()
        
        
if __name__ == '__main__':
    FNN(4)

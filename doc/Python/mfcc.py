# Load audio files and extract features
import numpy as np
import librosa
import pandas as pd
import os
import csv

def parser(ID, n_mfcc, filename):
   # function to load files and extract features
   file_name = os.path.join(filename, str(ID) + '.wav')

   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
      
   except:
      print("Error encountered while parsing file: ", file_name)
      print(ID)
      return None
 
   return mfccs

n_mfcc = 40
'''
# Training data
train = pd.read_csv("../data/train.csv")

X = np.zeros((len(train), n_mfcc))
#f = open("../data/t.txt", "w")

for i in range(len(train)):
    feature, label = parser(train.ID[i], train.Class[i], n_mfcc, "../data/Train/")
    X[i] = feature
    #f.write(str(label)+'\n')

#np.savetxt("../data/X_40.txt", X)
#f.close()
'''
# Test data
test = pd.read_csv("../data/test.csv")

X = np.zeros((len(test), n_mfcc))

for i in range(len(test)):
    feature = parser(test.ID[i], n_mfcc, '../data/Test/')
    X[i] = feature

#np.savetxt("../data/X_test_40.txt", X)


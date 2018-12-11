# Load audio files and extract features
import numpy as np
import librosa
import pandas as pd
import os
from tqdm import tqdm

def Spec(ID, bands, filename):
   # function to load files and extract features
   file_name = os.path.join(filename, str(ID) + '.wav')

   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      
      melspec = librosa.feature.melspectrogram(X, n_mels = bands)
      logspec = librosa.amplitude_to_db(melspec)
      
      
   except:
      print("Error encountered while parsing file: ", file_name)
      logspec = np.zeros(bands*173)
 
   return logspec.flatten()


if __name__ == '__main__':
    bands = 40
    
    
    # Training data
    train = pd.read_csv("../data/train.csv")

    X = np.zeros((len(train)-2, bands*173))

    j = 0
    for i in tqdm(range(len(train))):
        if i != 1986 and i != 5312:
            feature = Spec(train.ID[i], bands, "../data/Train/")
            X[j,:len(feature)] = feature       # Zero padding
            j += 1

    np.savetxt("../data/spectrogram_%d.txt"%bands, X)
    
    
    
    # Test data
    test = pd.read_csv("../data/test.csv")

    X = np.zeros((len(test)-6, bands*173))

    j = 0
    for i in tqdm(range(len(test))):
        if i!=437 and i!=1100 and i!=1506 and i!=2087 and i!=2100 and i!=2291: 
            feature = Spec(test.ID[i], bands, "../data/Test/")
            X[j,:len(feature)] = feature       # Zero padding
            j += 1

    np.savetxt("../data/spectrogram_%d_test.txt"%bands, X)
    

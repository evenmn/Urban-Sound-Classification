import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd

train = pd.read_csv("../data/train.csv")

i = np.random.choice(train.index)
x, sr = librosa.load('../data/Train/' + str(train.ID[i]) + '.wav')

t = np.linspace(0,int(len(x)/sr),len(x))

# === Plot ===
# Time domain
FS = 14

plt.subplot(3,1,1)
plt.plot(t,x)
plt.title("%s"%train.Class[i], fontsize=FS)
plt.xlabel("Time [s]", fontsize=FS)
plt.ylabel("Amplitude", fontsize=FS)

'''
# Frequency domain
FFT = np.fft.fft(x)

plt.subplot(2,1,2)
plt.plot(FFT[:int(len(FFT)/2)])
plt.xlabel("Frequency [Hz]", fontsize=FS)
plt.ylabel("Amplitude", fontsize=FS)
plt.show()
'''

# Spectrogram
melspec = librosa.feature.melspectrogram(x, n_mels=40)
logspec = librosa.amplitude_to_db(melspec)

plt.subplot(3,1,2)
plt.imshow(logspec)

# MFCC
mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T,axis=0)

plt.subplot(3,1,3)
plt.plot(np.linspace(0,int(len(x)/sr),40),mfccs)
plt.xlabel("Time [s]", fontsize=FS)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def f(t, f1=1, f2=2):
    return np.sin(2*np.pi*f1*t)*np.sin(2*np.pi*f2*t+1)

x = np.linspace(0,1,1000)
p = np.linspace(0,1,40)
y = f(x)

label_size = {'size':'10'}
plt.plot(x, y, 'r')
plt.stem(p, f(p))
plt.xlabel('Time',**label_size)
plt.ylabel('Pressure',**label_size)

plt.show()

def f(t, f=1):
    return np.sin(2*np.pi*f*t)

y1 = f(x)
y2 = f(x, f=2)
y3 = f(x, f=4)

FFT1 = np.fft.fft(y1)
FFT2 = np.fft.fft(y2)
FFT3 = np.fft.fft(y3)

plt.subplot(2,1,1)
plt.plot(x, y1, color='r')
plt.plot(x, y2, color='b')
plt.plot(x, y3, color='g')
plt.ylabel("Pressure", **label_size)
plt.title("Time [s]", **label_size)
plt.subplot(2,1,2)
plt.stem(FFT1[:10], 'r', markerfmt=" ", label='$\sin(2\pi t)$')
plt.stem(FFT2[:10], 'b', markerfmt=" ", label='$\sin(4\pi t)$')
plt.stem(FFT3[:10], 'g', markerfmt=" ", label='$\sin(8\pi t)$')
plt.legend(loc='best', fontsize=10)
plt.xlabel("Frequency [Hz]", **label_size)
plt.ylabel("Magnitude", **label_size)
plt.show()

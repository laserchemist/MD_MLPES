import pywt
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import chirp
wavelist = pywt.wavelist(kind='continuous')
print("Available CWT: ",wavelist)
# Define signal
fs = 1000.0
sampling_period = 1 / fs
t = np.linspace(0, 2, 2 * fs)
x1 = chirp(t, 60, 2, 40, 'quadratic')
x=np.sin(2*np.pi*60*t)

# Calculate continuous wavelet transform
coef, freqs = pywt.cwt(x, np.arange(1, 100), 'morl',
                       sampling_period=sampling_period)

# Show w.r.t. time and frequency
plt.figure(figsize=(5, 2))
plt.pcolor(t, freqs, coef)

# Set yscale, ylim and labels
#plt.yscale('log')
#plt.ylim([1, 100])
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
icoef=coef.sum(axis=1) # sum over rows which contain coefficients 
integralcoef=np.cumsum(icoef[:-10])
plt.figure(figsize=(5, 2))
plt.xlim((5,500))
plt.plot(freqs,icoef,'g',lw=1)
plt.plot(freqs[:-10],integralcoef,'r-',lw=1)
filename="wavelet_data.csv"
wlfile=open(filename,"w+")
for i, coef in enumerate(icoef):
    wlfile.write(str(i)+", "+str(freqs[i])+", "+str(coef)+"\n")
wlfile.close()
#plt.savefig('egg_sine.png', dpi=150)
plt.figure(figsize=(5, 2))
coef, freqs = pywt.cwt(x1, np.arange(1, 50), 'morl',
                       sampling_period=sampling_period)
plt.pcolor(t, freqs, coef)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.show()
#plt.savefig('egg.png', dpi=150)
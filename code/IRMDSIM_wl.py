#!/usr/bin/env python
# JMS 10 May 2019  IRMDSIM
#     Split off version that solely processes dipole data from programs like CP2K or Gaussian
#        after stripped from output or originally created
# JMS 04 December 2018
# 1.4 JMS 25 January 2019
#     Python script to read GAMESS MD trajectory from output file and save as xmol
#     and process data. Beta version
#     Need to add retrieval of solvent molecules
# 1.5 JMS 30 January
#     Updated to read and process version dependent output
# 1.6 JMS 18 March
#     Adding code to detect Gaussian output and parse with new class
# Some debug printing remains, treat as beta
# IRMDSIM_wl
#   Wavelet addition
import re
import numpy as np
#import scipy
from scipy import signal
import matplotlib.pyplot as plt
import math
from scipy import fftpack
import os
import sys
import pywt # wavelet tool


# ++++++++++++++++++ DEFINE FUNCTIONS +++++++++++++
# Need function to return array of COORDINATES

def calc_derivative(array_1D, delta_t):
   # Use finite difference
    dy = np.gradient(array_1D)
    return np.divide(dy, delta_t)
def autocorr(x):
    yunbiased = x - np.mean(x, axis=0)
    ynorm = np.sum(np.power(yunbiased,2), axis=0)  # Just appears to normalize largest peak to 1.00
    result = np.correlate(x, x, mode='full')/ynorm
    return result[int(result.size/2):]
def zero_padding(sample_data):
    """ A series of Zeros will be padded to the end of the dipole moment
    array (before FFT performed), in order to obtain a array with the
    length which is the "next power of two" of numbers.

    #### Next power of two is calculated as: 2**math.ceil(math.log(x,2))
    #### or Nfft = 2**int(math.log(len(data_array)*2-1, 2))
    """
    return int(2 ** math.ceil(math.log(len(sample_data), 2)))
def choose_window(data, name,width):
    kind = name
    print("KIND:",kind)
    if kind == "Gaussian":
        sigma = 2 * math.sqrt(2 * math.log(2))
        std = float(width)
        window_function = signal.gaussian(len(data), std/sigma, sym=False)

    elif kind == "Blackman-Harris":
        window_function = signal.blackmanharris(len(data), sym=False)

    elif kind == "Hamming":
        window_function = signal.hamming(len(data), sym=False)

    elif kind == "Hann":
        window_function = signal.hann(len(data), sym=False)

    print(window_function)
    return window_function
def calc_FFT(array_1D, window):
    """
    This function is for calculating the "intensity" of the ACF at each frequency
    by using the discrete fast Fourier transform.
    """
####
#### http://stackoverflow.com/questions/20165193/fft-normalization
####
    #window = choose_window(array_1D, "Gaussian")
    # swindow=np.sum(window)
    # WE = swindow /array_1D[0].shape
    # print("Window shape: ",WE.shape)
    #wf=np.true_divide(window, WE)

    WE = np.sum(window) / len(array_1D)
    wf = window / WE
    # convolve the blackman-harris or other window function.
    sig = array_1D * wf
    # A series of number of zeros will be padded to the end of the \
    # VACF array before FFT.
    N = zero_padding(sig)
    # Tried using Numpy FFT but fftpack works better for this application
    #yfft = np.fft.fft(sig, N, axis=0) / len(sig)
    # yfft = np.fft.fft(sig, N, axis=0)/len(sig) # 
    # Try this... Works better, somehow above shifts spectrum to much higher cm-1
    yfft=fftpack.fft(sig)/len(sig)
    print("shape of yfft {:}".format(np.shape(yfft)))
    #return np.square(np.absolute(yfft))
    return np.abs(yfft)
#+++++++++++++++++++++++++++++++++
# Wavelet Plotting
def plot_wavelet(time, signal, scales, 
                 waveletname = 'cmor', 
                 cmap = plt.cm.seismic, 
                 title = 'Wavelet Transform (Power Spectrum) of signal', 
                 ylabel = 'Period (years)', 
                 xlabel = 'Time'):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    #plt.show()

#++++++++++++++++++++++++++++++++++++++++++++
# MAIN PROGRAM
#++++++++++++++++++++++++++++++++++++++++++++
# Command Line:
# 1. filename prefix (input csv file with dipole components)
# 2. path to file
# 3. file ending [.traj]
# 4. wnwidth (width of Gaussian spectral features)
# 
if len(sys.argv)>1:
    file=sys.argv[1]
    filename=file
else:
    # Sample local files
    file='output65'
    file='imidazole_30w_output65'
    file='HCl_output77'
    file='imidazole_water_1fs_output78'
    file='acetylene_300K_MD_output84'
    file='acetylene_MD_output84'
    file='acetylene_MD_300K_long_output84'
    file='Acetyleneoutput91'
    file='acetylene_output91'
    path='/Users/jms/local/molecules/imidazole/'
    file='output65'
    file='imidazole_30w_output65'
    file='HCl_output77'
    file='imidazole_water_1fs_output78'
    file='acetylene_300K_MD_output84'
    file='acetylene_MD_output84'
    file='acetylene_MD_300K_long_output84'
    file='Acetyleneoutput91'
    file='acetylene_output91'
    file='acetylene_300K_MD_output84'
    file="dipolemet.traj"
    print("No command line argument using default file: ", file)
if len(sys.argv)>2:
    path=sys.argv[2]
else:
    path='/Users/jms/local/molecules/imidazole/'
    print("No command line argument using default path: ", path)
if len(sys.argv)>3:
    fend=sys.argv[3]
else:
    fend=".traj"
if len(sys.argv)>4:
    wnwidth=float(sys.argv[4])
else:
    wnwidth=float(10)
filepath=path+file+fend
filepathdat=path+file+".dat"
print("\n\nIRMDSIM.py")
print("++++++++++++++++++")
print("ver. 0.1 JMS 10 May 2019\n\n")
print("File: ",filepath)
# Gaussian(R) in Gaussian program
# GAMESS VERSION in Gamess program
# Determine Output file type
gxx="Gaussian"
ga="GAMESS"
# DIPOLE [Non Periodic](Debye)|                    6.289877  -2.485134  -0.000479
# DIPOLE [Non Periodic] DERIVATIVE(A.U.)|         -0.000236  -0.000517  -0.000091
cp2k="DERIVATIVE"
cp2k_dipole="(Debye)"
try:
    with open(filepath, "r") as source:
            for line in source:
                    ltest=line
                    print("Line: ",ltest)
                    if ltest.find(ga) > 0:
                        code = "GAMESS"
                        #calc1=gamess(filepath)
                        break
                    elif ltest.find(gxx) > 0:
                        code = "GAUSSIAN"
                        #calc1=g16(filepath)
                        break
                    elif ltest.find(cp2k)>0:
                        code="CP2K"
                        print("CP2K")
                        break
                    elif ltest.find(cp2k_dipole)>0:
                        code="CP2K-dipole"
                        print("CP2K dipole")
                        break
                   # else:
                   #     print("Computational code for log file unidentified...\n----> retry entry")
                   #     quit()
except:
    print("No file named ", file,fend," found\n ---> QUITTING.")
    quit()
print("Computational code: ",code)
outlines=[]
numline=[]
substr='fsec'
# Need time computation
# Should be based on file length


#Now retrieve dipole moment as a function of time from trajectory
version=["R3",3,2019]
print(version,version[2])
if code=="CP2K":
    time_step=0.2 # Default in ASE md_sgdml_psi4_ase.py
    #dipderiv=[]
    with open(filepath, "r") as f:
        n=0
        for line in f:
            #  DIPOLE [Non Periodic](Debye)|                    6.289877  -2.485134  -0.000479
            #  DIPOLE [Non Periodic] DERIVATIVE(A.U.)|         -0.000236  -0.000517  -0.000091
            
            match = re.search(r"DERIVATIVE\(A\.U\.\)\|\s+([-\s]\d+\.\d+)\s+([-\s]\d+\.\d+)\s+([-\s]\d+\.\d+)", line)
            if match:
                n+=1
                ddlist=[match.group(1),match.group(2),match.group(3)]
                print(ddlist)
                if n==1:
                    dipderiv=np.hstack([float(i) for i in ddlist])
                else:
                    nprow=np.array([float(i) for i in ddlist])
                    #print("shape: ",dipderiv.shape,nprow.shape)
                    dipderiv=np.vstack([dipderiv,nprow])
    endtime=n*time_step
    print("CP2K",endtime)
    np_times=np.linspace(0.0, endtime, num=n)
    delta_t=np_times[-1]-np_times[-2]
    ndderiv=np.array(dipderiv[0:-1],dtype=float) # ValueError: setting an array element with a sequence.
    print("Shape of numpy array: ",ndderiv.shape)
    #ndderiv=calc_derivative(ndderiv,delta_t)
    print("Shape of numpy derivative array: ",ndderiv.shape)
    #ndderiv=np.array(ndderiv,dtype=float)[0,:,:] #Goes from 3D to 2D
    print("Shape of numpy derivative array: ",ndderiv.shape)
    #print("dipderive",dipderiv)
elif code=="CP2K-dipole":
    print("processing CP2K dipole...")
    time_step=0.2 # fsec step
    dipderiv=[]
    with open(filepath, "r") as f:
        n=0
        for line in f:
            #  DIPOLE [Non Periodic](Debye)|                    6.289877  -2.485134  -0.000479
            #  DIPOLE [Non Periodic] DERIVATIVE(A.U.)|         -0.000236  -0.000517  -0.000091
            #   X=   -0.15546699 Y=   -0.03735786 Z=   -2.68081527     Total=      2.68557929
            matchold = re.search(r"X=\s?([-\s]\d+\.\d+)\sY=\s?([-\s]\d+\.\d+)\sZ=\s?([-\s]\d+\.\d+)\s+Total=\s?([-\s]\d+\.\d+)", line)
            match = re.findall(r"[+-]?\d+\.\d+",line) # First number is energy
            print(line)
            print(match)
            if match:
                #print("Match")
                n+=1
                ddlist=match[1:4]
                #ddlist=[match.group(1),match.group(2),match.group(3),match.group(4)]
                #print("ddlist:",ddlist)
                if n==1:
                    dipderiv=np.hstack([float(i) for i in ddlist])
                else:
                    nprow=np.array([float(i) for i in ddlist])
                    #print("shape: ",dipderiv.shape,nprow.shape)
                    dipderiv=np.vstack([dipderiv,nprow])
    endtime=n*time_step

    print("CP2K_dipole",endtime)
    print("Dipderiv: ",dipderiv)
    print("total: ",dipderiv[:][3])
    np_times=np.linspace(0.0, endtime, num=n)
    delta_t=np_times[-1]-np_times[-2]
    # calc_derivative(array_1D, delta_t):
    ndderiv=np.array(dipderiv[0:-1],dtype=float)  # ValueError: setting an array element with a sequence.
    #ndderiv=np.square(np.array(dipderiv[0:-1],dtype=float)) # ValueError: setting an array element with a sequence.
    print("Shape of numpy array: ",ndderiv.shape)
    #ndderiv=calc_derivative(ndderiv,delta_t)
    print("Shape of numpy derivative array: ",ndderiv.shape)
    #ndderiv=np.array(ndderiv,dtype=float)[0,:,:] #Goes from 3D to 2D
    print("Shape of numpy derivative array: ",ndderiv.shape)
elif float(version[2])<2018:
        #Dipole
        dipole=calc1.dipole()
        print("dipole length ",len(dipole))
        dipderiv=calc1.dipderiv()
        print("dipole derivative length ",len(dipderiv))
        print("dipole derivative: ", dipderiv[0],"++++++++++") # Works
        print("Lengths ", len(dipderiv[0][:]),len(dipderiv[:]),len(dipderiv[:][:]),len(dipderiv[3][:]))
        ndderiv=np.array(dipderiv[0:-1],dtype=float) # ValueError: setting an array element with a sequence.
        print("Shape of numpy array: ",ndderiv.shape)
        nddipole=np.array(ndderiv,dtype=float)[:,0,:] #Goes from 3D to 2D
        print("Shape of nddipole numpy array: ",nddipole.shape)
else:
        print("Newer version ") # 2018 version puts dipole this in .dat file
        # Capture this text:  DIPOLE      -0.028421  0.034413  0.047308
        dipole=calcdat.dipnew()
        print("dipole length ",len(dipole))
        nddipole=np.array(dipole[0:-1],dtype=float) # ValueError: setting an array element with a sequence.
        print("Shape of numpy array: ",nddipole.shape)
        nddipole=np.array(nddipole,dtype=float)[1:,0,:] #Goes from 3D to 2D REMOVE first row to match dimensions
        print("Shape of nddipole numpy array: ",nddipole.shape)
        time_step=float(time[-2][1])-float(time[-3][1]) # 1.0 # Time step in FSEC [22 January]
        ddipolez=calc_derivative(nddipole[:,2], time_step)
#time_step=float(time[-1][0])-float(time[-2][0]) # 1.0 # Time step in FSEC [22 January]
time=np_times[-1]
print("Last Time: ", time)

print("Time step [fsec]: ",time_step)
print("Time step retrieved: ",time_step)
print("Shape of np_times",np_times.shape)
t0=500 # Initial time index point to start data analysis
dipoletotal=np.sqrt(np.square(ndderiv[t0:,0])+np.square(ndderiv[t0:,1])+np.square(ndderiv[t0:,2]))
print("Shape of total dipole array: ",dipoletotal.shape)
#ddipoletot=calc_derivative(dipoletotal, time_step)
#testcorr=autocorr(ndderiv[t0:,2])
#testcorr=autocorr(dipoletotal)
ddipoley=ndderiv[t0:,1] #calc_derivative(nddipole[t0:,1], time_step)
ddipolex=ndderiv[t0:,0] #calc_derivative(nddipole[t0:,0], time_step)
ddipolez=ndderiv[t0:,2]  #calc_derivative(nddipole[t0:,2], time_step)
ddipoletot=ddipolez+ddipoley+ddipolex
print(dipoletotal.shape)
autodipole=autocorr(ddipoletot)
zderiv=ddipolez
az=autocorr(ddipolez)
ay=autocorr(ddipoley)
ax=autocorr(ddipolex)
# Select dipole component to plot
sum=az # Using SUM of autocorrelated dipole derivatives
sum=autocorr(ddipoletot) # looks more correct from DFT results which have total dipole tabulated
#sum=az 
psum=zero_padding(sum)
print("+ Zero padding for FFT: ",psum)
sigma = 2 * math.sqrt(2 * math.log(2))
standard=4000
std = float(standard)
wf_name="Blackman-Harris"
# "The larger the number, the narrower the line-shape.
# "(e.g. 4000 yields FWHM around 10 cm^-1 for a 20-ps traj.)",
width=10000
timelength=float(np_times[-1])
width=67*wnwidth/900*timelength
window = choose_window(sum, wf_name,width)
N = zero_padding(sum)
#yfft = np.fft.fft(sig, N, axis=0) / len(sig)
testfft = np.fft.fft(sum, N, axis=0) # no window func
plt.figure(figsize=(6, 5))
plt.plot(testfft)
plt.title('Test FFT')
dpfft=calc_FFT(sum, window)
plt.figure(figsize=(6, 5))
plt.plot(dpfft)
plt.title('Test FFT with window')
# TRY fftpack
#multiply by 27 THz signal
time_step=float((float(np_times[-1])-float(np_times[t0]))/len(sum)) # 1.0 # Time step in FSEC
print("time_step",time_step)
end_time=float(np_times[-2])+time_step
start_time=float(np_times[t0])
time_vec = np.arange(start_time,end_time, time_step) # Time step vector in FSEC
f=1/time_step
f=0.038 # 1260 cm-1 vibration
# 1 femtosecond = 1000 THz
sigsim=(np.sin(f * 2 * np.pi * time_vec)+1.0)
f=0.0007
sigsim2=(np.sin(f * 2 * np.pi * time_vec)+1.0)
# 1 femtosecond = 1000 THz
plt.figure(figsize=(6, 5))
plt.plot(time_vec, sum, label='Original signal')
#+sigsim #*sigsim*sigsim2
#plt.plot(time_vec, sum, label='Multiplied signal')
plt.legend()
sig_fft = fftpack.fft(sum)
# And the power (sig_fft is of complex dtype)
# FT ACF
# +++++++++++++++++++++++++++++++++++++++++++
power = np.abs(sig_fft)[0:int(sum.size / 2)]
wf_name="Blackman-Harris"
window = choose_window(sum, wf_name,width)
powerBH=calc_FFT(sum, window)[0:int(sum.size / 2)]
wf_name="Hann"
window = choose_window(sum, wf_name,width)
powerHann=calc_FFT(sum, window)[0:int(sum.size / 2)]
wf_name="Hamming"
window = choose_window(sum, wf_name,width)
powerHam=calc_FFT(sum, window)[0:int(sum.size / 2)]
wf_name="Gaussian"
width=13000/wnwidth
#width=1300 #temporary
window = choose_window(sum, wf_name,width)
powerGauss=calc_FFT(sum, window)[0:int(sum.size / 2)]
power2=dpfft[0:int(sum.size / 2)]
# The corresponding frequencies
c = 2.9979245899e-5 # speed of light in vacuum in [cm/FSEC]
sample_freq = fftpack.fftfreq(sum.size, d=time_step/1000)[0:int(sum.size / 2)]
sample_omega=np.fft.fftfreq(sum.size, time_step*2 * np.pi/1000)[0:int(sum.size / 2)]
wavenumber = fftpack.fftfreq(sum.size, time_step * c)[0:int(sum.size / 2)]
wavenumber1 = np.fft.fftfreq(sum.size, time_step * c)[0:int(sum.size / 2)]
kb=1.38064E-23
T=298
hbar=1.0545718E-34
# Compute Infrared intensity 
IR_intensity=power2*sample_omega*np.tanh(hbar*1.0E15*sample_omega/(kb*T))
IR_intensityBH=powerBH*sample_omega*np.tanh(hbar*1.0E15*sample_omega/(kb*T))
IR_intensityHann=powerHann*sample_omega*np.tanh(hbar*1.0E15*sample_omega/(kb*T))
IR_intensityHam=powerHam*sample_omega*np.tanh(hbar*1.0E15*sample_omega/(kb*T))
IR_intensityGauss=powerGauss*sample_omega*np.tanh(hbar*1.0E15*sample_omega/(kb*T))
# Create a series of diagnostic plots
# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [THz]')
plt.ylabel('power')
plt.figure(figsize=(6, 5))
plt.plot(sample_omega, power)
plt.xlabel('Frequency [omega]')
plt.ylabel('power')
plt.figure(figsize=(6, 5))
plt.plot(wavenumber, power)
plt.grid(True)
plt.xlim(100, 5000)
plt.xlabel('Frequency [cm^-1]')
plt.ylabel('power')
plt.figure(figsize=(6, 5))
plt.plot(wavenumber, IR_intensityBH, '-b', label="Blackman-Harris")
plt.plot(wavenumber,IR_intensityHann,'-r', label="Hann")
plt.plot(wavenumber,IR_intensityHam,'-g', label="Hamming")
plt.plot(wavenumber,IR_intensityGauss,'-b', label="Gaussian")
plt.xlabel('Frequency [cm^-1]')
plt.legend()
plt.ylabel('IR_intensity')
plt.grid(True)
plt.xlim(100, 7000)
plt.savefig(filename+'_IRNW'+'.png')
plt.figure(figsize=(6, 5))
plt.plot(wavenumber, IR_intensity)
plt.xlabel('Frequency [cm^-1]')
plt.ylabel('IR_intensity')
label="Window width: "+str(width)
plt.text(3000,1,label)
plt.grid(True)
plt.xlim(100, 5000)
plt.ylim(0,.5)
plt.savefig(filename+'_IR'+'.png')
filename=filepath+"_spec.csv"
print("Writing spectrum file to:",filename)
# Write out Spectra and MD data
# Quick way to write out 2D array
# https://www.python-course.eu/numpy_reading_writing.php
spectrum=np.vstack((wavenumber,IR_intensity,IR_intensityBH,IR_intensityHann,IR_intensityHam,IR_intensityGauss)).T
np.savetxt(filename, spectrum, delimiter=',', fmt='%15.11f')

# These arrays get mishaped, have to fix ad hoc
end_time=float(np_times[-2])+time_step
start_time=float(np_times[0])
time_vec_tot = np.arange(start_time,end_time, time_step) # Total Time step vector in FSEC
print("Time_vec shapes: ",time_vec_tot.shape)
print(time_vec.shape)

# Wavelet plot
scales = np.arange(1,256)
# Formic acid C=O stretch: 1712 cm-1 = 19.48274 fsec = 51.32748 THz
timetry=np.arange(0,1000,0.1)
COstretch=np.cos(2*np.pi*timetry*190) # Used 0.0513
#plot_wavelet(time_vec, sum, scales,title='Spectrum')
plt.figure(figsize=(6, 5))

plt.plot(time_vec[:1000],COstretch[:1000],'r',label="COStretch")
plt.plot(time_vec[:1000],az[:1000],'g',label="signal")
plt.legend()
sampling_period = 1 / (time_step)
sa=0.1
# Calculate continuous wavelet transform

#plot_wavelet(time_vec, COstretch, scales,title='COStretch')
#plt.show()
plt.savefig('signal.png', dpi=150)
# Stop here to check what is happening
# TypeError: Dimensions of C (2047, 6000) are incompatible with X (4990) and
# /or Y (2047); see help(pcolormesh)
exit()
plt.figure(figsize=(6, 5))
coef, freqs = pywt.cwt(COstretch[0:6000], np.arange(1, 2048), 'morl',
                       sampling_period=sa)
plt.pcolormesh(time_vec[0:6000], freqs, coef) # Time intensive https://matplotlib.org/2.0.2/examples/pylab_examples/pcolor_demo.html
plt.ylabel('Frequency (1/fsec)')
plt.yscale('log')
#plt.ylim([1, 100])
plt.xlabel('Time (fsec)')
plt.savefig('wavelet.png', dpi=150)
icoef=np.square(coef.sum(axis=1)) # sum over rows which contain coefficients 
print("icoef",icoef.shape)
plt.xlabel('Frequency (1/fsec)')
plt.yscale('linear')
plt.xscale('log')
plt.plot(freqs,icoef,'g')
plt.savefig('sum_of_coeff_CO.png', dpi=150)
icoef=coef.sum(axis=1) # sum over rows which contain coefficients 

plt.figure(figsize=(5, 2))
plt.xlim((5,500))
plt.plot(freqs,icoef,'g',lw=1)
print("icoef",icoef.shape)
plt.xlabel('Frequency (1/fsec)')

plt.savefig('sum_of_coeff_sim.png', dpi=150)
filename="wavelet_data_sim.csv"
wlfile=open(filename,"w+")
for i, coef in enumerate(icoef):
    wlfile.write(str(i)+", "+str(freqs[i])+", "+str(coef)+"\n")
plt.figure(figsize=(6, 5))
coef, freqs = pywt.cwt(az, np.arange(1, 1024), 'morl',
                       sampling_period=sampling_period)
#f = pywt.scale2frequency(np.arange(1, 128), 'morl')/(sampling_period/4)
plt.pcolormesh(time_vec, freqs, coef)
print("Freqs:",freqs.shape)
#print(freqs)
print("Coeffs:",coef.shape)
print(coef[0])
print(coef[0].shape)
plt.figure(figsize=(6, 5))
plt.ylabel('Frequency (1/fsec)')
plt.xlabel('Time (fsec)')
#plt.yscale('log')
plt.savefig('signal_wavelet.png', dpi=150)
icoef=coef.sum(axis=1) # sum over rows which contain coefficients 

plt.figure(figsize=(5, 2))
plt.xlim((5,500))
plt.plot(freqs,icoef,'g',lw=1)
print("icoef",icoef.shape)
plt.xlabel('Frequency (1/fsec)')

plt.savefig('sum_of_coeff.png', dpi=150)
filename="wavelet_data.csv"
wlfile=open(filename,"w+")
for i, coef in enumerate(icoef):
    wlfile.write(str(i)+", "+str(freqs[i])+", "+str(coef)+"\n")
plt.figure(figsize=(6, 5))
#plt.surf(time_vec[1000:3000],freqs,abs(coef));shading('interp');
#plt.pcolormesh(time_vec, freqs, abs(coef))
plt.savefig('signal_wavelet_surf.png', dpi=150)
# Show w.r.t. time and frequency

# Finish

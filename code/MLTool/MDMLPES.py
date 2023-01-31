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

#!/Users/jms/.conda/envs/MDQUANT/bin/python
#########
# FFT_plot.py
# -----------
# 04.12.20 Add FFT option
#    JMS Oslo
#
import re
import numpy as np
#import scipy
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import math
from scipy import fftpack
import os
import sys
import pywt # wavelet tool
import pandas as pd
from ase import units
color=['r','b','c','m','y']
#######################
###    FUNCTIONS    ###
#######################
def correct_limit(ax, x, y,sc):   
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
    # ax: axes object handle
    #  x: data for entire x-axes
    #  y: data for entire y-axes
    # sc: scale max
    # assumption: you have already set the x-limit as desired
        lims = ax.get_xlim()
        xmin=lims[0]
        xmax=lims[1]
        print(f'xmin: {xmin:.6f} xmax: {xmax:.6f}')
        xtic=float((xmax-xmin)/10)
        i = np.where( (x > lims[0]) &  (x < lims[1]) )[0]
        # print("i: \n",i) Returns a list i of the index for given x values
        ax.grid(b=True, which='major', color='#666666', linestyle='-')
        ax.set_xlabel('Frequency [cm$\mathregular{^{-1}}$]')
        y_min = y[i].min()-0.1
        y_max = y[i].max()
        print(f'ymin: {ymin:.6f} ymax: {ymax:.6f}')
        ax.set_ylim( y_min, y_max*sc ) 
        ax.xaxis.set_ticks(np.arange(xmin,xmax,xtic))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4, color='m') 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
### FFT FUNCTIONS    ###
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
### FINISH FUNCTIONS ###
print("\n\nFFT_plot.py")
print("++++++++++++++++++")
print("ver. 0.1 JMS Oslo 04 December 2020\n\n")
# Read input file from command line argument
if len(sys.argv)<1:
    print("Command line should have file name(s)...")
    exit()
col1 = int(input("Enter the first, x column index [0]: ") or "0")
col2str = str(input("Enter the first, y column index [1 or range to average 1-3]: ") or "1")
if col2str.find("-") == 1:
    print("Multi row average")
    colrange=col2str.split('-')
    cr1=int(colrange[0])
    cr2=int(colrange[1])+1
    print("range: ",cr1,"-",cr2)
    avgcol=1
else:
    col2 = int(col2str)
    avgcol=0
xmin =float(input("Enter the x column lower limit[500]: ") or "500")
xmax = float(input("Enter the x column upper limit [4500]: ") or "4500")
xtic=float((xmax-xmin)/8)
print(xmin,xmax)
xtitle=str(input("Enter the x axis units [wavenumber (cm$^{-1}$)]: ") or "wavenumber (cm$^{-1}$)")
title=str(input("Enter plot title [filename]: ") or sys.argv[1])
yaxis =  int(input("Label y-axis [0/1]: ") or "0")
if yaxis:
    ytitle=str(input("Enter the y axis units [arbitrary]: ") or "arbitary")
fft =  int(input("FFT data [0/1]: ") or "0")
#row2 = data.iloc [:, [1, 2]] 
# Instead of =df['intensity'] =df.iloc[:,col2]
# Instead of =df['wavenumber'] =df.iloc[:,col1]

#headers=["wavenumber","intensity","I2"]
if fft:
    print('Performing FFT...\n')
    fig,axs = plt.subplots(2, 1, tight_layout=True)
    names=[]
    fn = sys.argv[1]
    print("Reading ",fn," will perform FFT")
    df=pd.read_csv(fn,skiprows=5,comment='#',sep='[, ]+', engine = "python", header=None)
    cols = df.columns
    for col in cols:
        df[col] = df[col].astype(float)
    print(df.iloc[:,col1].describe())
    if avgcol:
        df['avg'] = df.iloc[:,cr1:cr2].sum(axis=1)/3
        print("first two rows : ",df.iloc[1:4,cr1:cr2])
        print(df.iloc[:,cr1].describe())
        print(df['avg'].describe())
        col2=-1 # Trick is that the new column is now last!
        print(df.iloc[:,col2].describe())
    else:    
        print(df.iloc[:,col2].describe())
    ymin=df.iloc[:,col2].min()
    df.iloc[:,col2] -=ymin
    ymax=df.iloc[:,col2].max()
    df.iloc[:,col2] /=ymax
    print("file scaled: ",fn)
    print(df.iloc[:,col1].describe())
    print(df.iloc[:,col2].describe())
    axs[0].set_xlim(xmin,xmax)
    axs[0].grid(True)
    x=df.iloc[:,col1]
    y=df.iloc[:,col2]
    i = np.where( (x > xmin) &  (x < xmax) )[0]
    print("y.max: ",y[i].max(),y[i].min())
    xmax = x[i].max()
    xmin = x[i].min()
    print("x.max: ",y[i].max(),y[i].min()," fsec (assumed)")

    y=y/y[i].max() 
    print(y[i].max())
    axs[0].plot(x,y,color[0],label=fn)
    correct_limit(axs[0], x,y,1.1)
    axs[0].set_xlabel(xtitle)
    if yaxis:
            # If displaying, need to rescale back
            ymin_now=df.iloc[:,col2].min()
            ymax_now=df.iloc[:,col2].max()
            df.iloc[:,col2] *=(ymax)
            axs[0].spines['left'].set_visible(True)
            axs[0].axes.get_yaxis().set_visible(True)
            axs[0].set_ylabel(ytitle)
    axs[0].legend(fontsize=5)
    ###############
    ### Now FFT ###
    ###############
    tstep = x[1]-x[0]
    print("Time step computed..: ",tstep)
    timelength=float(xmax-xmin)
    print("Number of data points: ",i.max())
    start=int(i.max()/10)
    print("start: ",start)
    dacf=autocorr(y) 
    # Prepare for FFT
    zp=zero_padding(dacf)
    timelength=float(xmax-xmin)
    wnwidth=7.0 # waveneumber width
    width=67*wnwidth/900*timelength
    wf_name="Hann"
    # Change to match MLPES_MD_IR_MLDIP.py
    width=len(dacf)/4 # /4 works since the data width will be ~2 sigma 
    wf_name="Hann"
    wf_name="Gaussian"
    window = choose_window(dacf, wf_name,width)
    powerHann=calc_FFT(dacf, window)[0:int(dacf.size / 2)]
    end_time=timelength
    start_time=float(xmin)
    c = 2.9979245899e-5 # speed of light in vacuum in [cm/FSEC]
    hbar=units._hplanck/(2*np.pi)
    kb=units._k
    print("Constants used (c, hbar, kb): ",c,hbar, kb)
    wavenumber = fftpack.fftfreq(dacf.size, tstep * c)[0:int(dacf.size / 2)]
    sample_omega=np.fft.fftfreq(dacf.size, tstep*2 * np.pi/1000)[0:int(dacf.size / 2)]
    temp = 300 # Temperature for computing intensity
    IR_intensityHann=powerHann*sample_omega*np.tanh(hbar*1.0E15*sample_omega/(kb*temp))
    ### Now plot FFT ###
    xmin=500
    xmax=5000
    axs[1].set_xlim(xmin,xmax)
    axs[1].grid(True)
    #correct_limit(axs[1], wavenumber, IR_intensityHann ,1.1)
    xtic=float((xmax-xmin)/10)
    x = wavenumber
    i = np.where( (x > xmin) &  (x < xmax) )[0]
    y = IR_intensityHann
    axs[1].set_ylim( y[i].min(), y[i].max() ) 
    axs[1].plot(wavenumber,IR_intensityHann,color[1],label="FFT "+fn)
    spectrum=np.vstack((wavenumber,IR_intensityHann,powerHann)).T
    np.savetxt(fn+'_FFT.csv', spectrum, delimiter=',', fmt='%15.11f')
## Case without FFT
else:
    fig,axs = plt.subplots(1, 1, tight_layout=True)
    names=[]
    numfile = len(sys.argv[1:])
    for count,fn in enumerate(sys.argv[1:]):
        print("file: ",fn)
        names.append(fn)
        df=pd.read_csv(fn,skiprows=5,comment='#',sep='[, ]+', engine = "python", header=None)
        cols = df.columns
        for col in cols:
            df[col] = df[col].astype(float)
        print(df.iloc[:,col1].describe())
        print(df.iloc[:,col2].describe())
        ymin=df.iloc[:,col2].min()
        df.iloc[:,col2] -= ymin
        ymin=df.iloc[:,col2].min()
        ymax=df.iloc[:,col2].max()
        df.iloc[:,col2] /= ymax
        df.iloc[:,col2] += count/10
        print("file scaled: ",fn)
        print(df.iloc[:,col1].describe())
        print(df.iloc[:,col2].describe())
        axs.set_xlim(xmin,xmax)
        axs.grid(True)
        x=df.iloc[:,col1]
        y=df.iloc[:,col2]
        i = np.where( (x > xmin) &  (x < xmax) )[0]
        ymin = y[i].min()
        ymax = y[i].max()
        print("ymin: ",ymin," ymax: ",ymax)
        y -= ymin
        ymax = y[i].max()
        y=y/ymax+count/10 
        print(y[i].max())
        ytop = 1 + 0.11*numfile
        axs.set_ylim(0,ytop)
        axs.plot(x,y,color[count],label=fn)
        if count==0:
            correct_limit(axs, x,y,ytop)
        axs.set_xlabel(xtitle)
        if yaxis:
            # If displaying, need to rescale back
            ymin_now=df.iloc[:,col2].min()
            ymax_now=df.iloc[:,col2].max()
            df.iloc[:,col2] *=(ymax)
            axs.spines['left'].set_visible(True)
            axs.axes.get_yaxis().set_visible(True)
            axs.set_ylabel(ytitle)
        axs.legend(fontsize=5)
# Now finish off figure and save as pdf
fig.suptitle(title, fontsize=11)
fig.show()
fname=sys.argv[1]+"_fig.png"
fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=True, bbox_inches=None, pad_inches=0.1,
        metadata=None)



   



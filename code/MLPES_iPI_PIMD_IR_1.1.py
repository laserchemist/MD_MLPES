#!/Users/jms/.conda/envs/MDQUANT/bin/python
# MLPES_iPI_PIMD_IR.py
# 13 December 2019 JMS Oslo
# 	Grab centroid output 
# 	from i-PI and compute dipole
# 	at each step.
# 16 November 2020 JMS Oslo
#       Incorporate ML dipole surface option from
#       MLPES_MD_IR_MLDIP.py
#
################################
# IMPORTS ######################
################################
## General
print("Starting to load libraries...\n")
from math import sqrt
import re
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math
from scipy import fftpack
import os
import sys
from datetime import date
import time
print("Loaded initial libraries...\n")
## Specific
sys.path.append('/Users/jms/local/python/code/')
# New tools for normal mode sampling
from MLPES_Tool import NMSample, Heigen, projection, plotspec
from sgdml.intf.ase_calc import SGDMLCalculator
from ase.io.extxyz import read_xyz
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase.vibrations import Vibrations
from ase import units
import pywt # wavelet tool
# Import PSI4 http://www.psicode.org
import psi4

#########################################
## Functions and Classes ################
#########################################
def set_psi4(atoms=None):
 # Generate the atomic input
        result = ''
        for atom in atoms:
            temp = '{}\t{:.15f}\t{:.15f}\t{:.15f}\n'.format(atom.symbol,
                                                            atom.position[0],
                                                            atom.position[1],
                                                            atom.position[2])
            result += temp
        #print("Geometry: ", result)
        molecule = psi4.geometry(result)
def compute_E_dipole(mol):
        methodpsi4='HF/6-31G(d)'  #'BLYP-D3BJ/6-31G'
        E=psi4.energy(methodpsi4)* psi4.constants.hartree2kcalmol
        dipx=psi4.variable('SCF DIPOLE X')
        print( methodpsi4+" Energy: ",E," kcal/mole")
        dipy=psi4.variable('SCF DIPOLE Y')
        dipz=psi4.variable('SCF DIPOLE Z')
        diptot=sqrt(dipx**2+dipy**2+dipz**2)
        print('dipole components: x= %.4f y= %.4f z= %.4f total= %.4f' %(dipx,dipy,dipz,diptot))
        return E,dipx,dipy,dipz
## Function to put atoms in standard order
def sortperm(atoms):
    import collections
    atom_labels=list(atoms.symbols)
    sorted_atom_counts=collections.Counter(atom_labels).most_common()
    print("Sorted: ",sorted_atom_counts)
    # Get alphabetic sort
    sorted_atom_counts = sorted(sorted_atom_counts, key = lambda x: (-x[1], x[0]))
    print("Sorted: ",sorted_atom_counts)
    std_order_atoms = []
    for tup in sorted_atom_counts:
        for atom in atoms:
                if atom.symbol == tup[0]:
                    std_order_atoms.append(atom)
    perm=[]
    for atom in std_order_atoms:
        perm.append(atom.index)
    print(perm)
    return perm
## Function to scale plot
def correct_limit(ax, x, y):
   from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
   # ax: axes object handle
   #  x: data for entire x-axes
   #  y: data for entire y-axes
   # assumption: you have already set the x-limit as desired
   lims = ax.get_xlim()
   xmin=lims[0]
   xmax=lims[1]
   xtic=float((xmax-xmin)/8)
   i = np.where( (x > lims[0]) &  (x < lims[1]) )[0]
   # print("i: \n",i) Returns a list i of the index for given x values
   ax.set_ylim( y[i].min(), y[i].max() ) 
   ax.xaxis.set_ticks(np.arange(xmin,xmax,xtic))
   ax.xaxis.set_minor_locator(AutoMinorLocator())
   ax.tick_params(which='both', width=1)
   ax.tick_params(which='major', length=7)
   ax.tick_params(which='minor', length=4, color='m') 
   ax.spines['right'].set_visible(False)
   ax.spines['top'].set_visible(False)
   ax.spines['left'].set_visible(False)
   ax.axes.get_yaxis().set_visible(False)
##########################################
## Functions for dipole autocorrelation ##
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
################################
## END Functions ###############
################################
## CODE ########################
################################
# Move to input file approach
print("\n\nMLPES_iPI_PIMD_IR.py")
print("++++++++++++++++++")
print("ver. 1.1 JMS 16 November 2020\n\n")
# Read input file from command line argument
if len(sys.argv)>1:
    inpfile=sys.argv[1]
else:
    # Default local file
    inpfile='mlpes.inp'
# Read parameters
#
#  INPUT FILE FORMAT (LINES)
# --------------------------
# 1. i-PI trajectory centroid file name (.xyz)
# 2. TITLE
# 3. Path to ML Dipole surface or "Dipole=False"
# 4. Output filename prefix
# 5. Plot xmin, xmax
#
try:
    with open(inpfile,'r') as inp:
        # i-PI trajectory file read (.npz)
        print("started reading file: ", inpfile)
        trajfile=inp.readline().rstrip()
        print("i-PI trajectory filename: ",trajfile)
        # Read title
        title=inp.readline().rstrip()
        print("Title: ",title)
        # Path to ML dipole surface through PES learn or "Dipole=False"
        MLdipole=inp.readline().rstrip()
        if "Dipole=False" in MLdipole:
            MLDIP=False # Set switch for dipole evaluation
            print("No machine learning dipole surface, use PSI4")
        else:
            print("Using ",MLdipole," ML dipole surface")
            sys.path.append(MLdipole)
            print("adding to path")
            from compute_energy import pes
            MLDIP=True
        # Read temperature or energy
        temp=float(inp.readline().rstrip())
        print("Temperature: ",temp,'K')
        # Read time step (fsec)
        tstep=float(inp.readline().rstrip())
        print("Time step: ",tstep,' fsec, Nyquist@4500 cm-1 ~ 0.6 fsec/step')
        # Read output filename
        outfile=inp.readline().rstrip()
        print("Output file name: ",outfile)
        # Read plotting parameters
        xtrema=inp.readline().rstrip().split(",") # Get plot xmin/xmax
        print(xtrema)
        xmin=float(xtrema[0])
        xmax=float(xtrema[1])
        print(f"x axis range:{xmin}:{xmax}")
        inp.seek(0) # Reset file pointer to beginning
        inputfiletext=inp.read()
except:
    print(sys.exc_info()[0],"\nNo file named ", inpfile," found\n ---> QUITTING.")
    quit()
lf=open(outfile+".log","w+")
lf.write("\n\nMLPES_iPI_PIMD_IR.py\n")
lf.write("++++++++++++++++++\n")
lf.write("ver. 1.1 JMS 16 November 2020\n\n")
lf.write("Date: %s\n"  %date.today())
timenow=time.strftime(" %H:%M:%S")
lf.write(" %s\n"  %timenow)
lf.write("\nINPUT FILE\n++++"+inputfiletext+"\n++++\n")
# Open file
#########################################
### i-PI Trajectory File format #########
#########################################
# 4
# CELL(abcABC):  377.94522   377.94522   377.94522    90.00000    90.00000    90.00000  Step:           0  Bead:       0 x_centroid{angstrom}  cell{atomic_unit}
#       C  3.50528e-01 -2.65116e-01 -4.51212e-01
#       C -3.23149e-01  1.47546e-01  4.52444e-01
#       H -9.21339e-01  5.13965e-01  1.25484e+00
#       H  9.48718e-01 -6.31542e-01 -1.25361e+00
psi4.core.set_output_file('output.dat', False)
psi4.set_memory('500 MB')
fdipole=open('dipole_'+trajfile+'.traj','w')
traj = read(trajfile,index=':') #ASE reads entire .xyz trajectory
mol = read(trajfile,index='0') 
print("Molecule file: ",mol)
lf.write("Molecule file: "+str(mol)+"\n\n")
print("Cartesians: "+str(mol.get_positions()))
print("Atomic numbers: ",mol.get_atomic_numbers())
lf.write("Atomic numbers: "+str(mol.get_atomic_numbers())+"\n")
cartpos=np.array(mol.get_positions())
numatm=np.shape(cartpos)[0]
print("number of atoms: ",numatm)
lf.write("Number of atoms: "+str(numatm))
print("Distances: ",mol.get_all_distances())
imagefile=outfile+"_initial_geo.png"
write(imagefile,mol)
for i, Atoms in enumerate(traj):
    print('Step : ',i,'\n',Atoms.get_positions())
    result=''
    if MLDIP:
                #print("loading ML dipole surface")
                perm=sortperm(Atoms) # Permute for dipole surface
                cartpos=np.array(Atoms.get_positions())
                cartpos2=cartpos[perm]
                cartlist=np.reshape(cartpos2,3*numatm)
                #print("Coordinates: ",cartpos)
                print("Distances: ",Atoms.get_all_distances())
                diptot=pes(cartlist,cartesian=True)
                print("Dipole total: ",diptot)
                dipx=dipy=dipz=0
                E = 0
    else:
        for atom in Atoms:
            temp = '{}\t{:.15f}\t{:.15f}\t{:.15f}\n'.format(atom.symbol,atom.position[0],atom.position[1],atom.position[2])
            result += temp
            molecule = psi4.geometry(result)
            E,dipx,dipy,dipz=compute_E_dipole(molecule)
            diptot=sqrt(dipx**2+dipy**2+dipz**2)
    ddlist=[dipx,dipy,dipz,diptot]
    if i==0:
        dipderiv=np.hstack([float(j) for j in ddlist])
    else:
        nprow=np.array([float(i) for i in ddlist])
        dipderiv=np.vstack([dipderiv,nprow])
    fdipole.write('%.7f DIPOLE [Non Periodic](Debye)| X=  %.6f Y=  %.6f Z=   %.6f Total=  %.6f \n' % (E,dipx,dipy,dipz,diptot))
    # Now write Cartesians
    fdipole.write(result)
steps=int(i)
### Trajectory Finished ###
endtime=tstep*steps
np_times=np.linspace(0.0,endtime,num=steps)
t0=int(endtime/10*tstep)
print("Time zero (after equilibration): ",t0)
ddipolex=dipderiv[t0:,0] #calc_derivative(nddipole[t0:,0], time_step)
ddipoley=dipderiv[t0:,1] #calc_derivative(nddipole[t0:,1], time_step)
ddipolez=dipderiv[t0:,2]  #calc_derivative(nddipole[t0:,2], time_step)
ddipoletot=dipderiv[t0:,3]  # Use total already computed
#ddipoletot=calc_derivative(dipderiv[t0:,3], tstep)

dacf=autocorr(ddipoletot) 
# Prepare for FFT
zp=zero_padding(dacf)
timelength=float(np_times[-1])
wnwidth=7.0 # waveneumber width
width=67*wnwidth/900*timelength
wf_name="Hann"
# Change to match MLPES_MD_IR_MLDIP.py
width=len(dacf)/8 # /4 works since the data width will be ~2 sigma 
wf_name="Hann"
wf_name="Gaussian"
window = choose_window(dacf, wf_name,width)
powerHann=calc_FFT(dacf, window)[0:int(dacf.size / 2)]
end_time=float(np_times[-2])+tstep
start_time=float(np_times[t0])
c = 2.9979245899e-5 # speed of light in vacuum in [cm/FSEC]
hbar=units._hplanck/(2*np.pi)
kb=units._k
print("Constants used (c, hbar, kb): ",c,hbar, kb)
wavenumber = fftpack.fftfreq(dacf.size, tstep * c)[0:int(dacf.size / 2)]
sample_omega=np.fft.fftfreq(dacf.size, tstep*2 * np.pi/1000)[0:int(dacf.size / 2)]
IR_intensityHann=powerHann*sample_omega*np.tanh(hbar*1.0E15*sample_omega/(kb*temp))
## Plot figure and save figure and data
fig,axs = plt.subplots(1, 1, tight_layout=True)
axs.plot(wavenumber,IR_intensityHann,'-r', label="Hann")
axs.grid(True)
axs.set_xlim(xmin, xmax)
correct_limit(axs, wavenumber,IR_intensityHann)
axs.set_xlabel('Frequency [cm^-1]')
axs.legend(fontsize=5)
axs.set_ylabel('IR_intensity')
fig.savefig(trajfile+'_IRNW'+'.png')
spectrum=np.vstack((wavenumber,IR_intensityHann,powerHann)).T
np.savetxt(trajfile+'_spec.csv', spectrum, delimiter=',', fmt='%15.11f')
fdipole.close()
print("+++COMPLETED+++\nPlot saved as: ",trajfile+'_IRNW'+'.png',"\nData saved as: ",trajfile+'_spec.csv',"\nTrajectory data and Cartesians saved as: ",'dipole_'+trajfile+'.traj')
lf.close()

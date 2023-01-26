#!/Users/jms/.conda/envs/MDQUANT/bin/python
# MLPES_MD_IR.py
# 11 December 2019 JMS Oslo
# 23 March 2020 JMS Oslo
#   Add option to use ML dipole surface
# 8  May 2020 JMS Oslo
#   Add new normal mode sampling option
# 13 May 2020 JMS Oslo
#   Add external plotting with overlay
# 19 May 2020 JMS Oslo
#   Add log, Use Boltzmann normal mode sampling
#   Plot projection on normal coordinates
################################
# IMPORTS ######################
################################
## General
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
## Specific
sys.path.append('/Users/jms/local/python/code/') 
from MLPES_Tool import NMSample, Heigen, projection, plotspec # New tools for normal mode sampling
from sgdml.intf.ase_calc import SGDMLCalculator
from ase.io.extxyz import read_xyz
from ase.io import read, write
from ase.optimize import QuasiNewton
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.vibrations import Vibrations
from ase.calculators.psi4 import Psi4
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
        #print("Geeometry: ", result)
        molecule = psi4.geometry(result)
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
def compute_E_dipole(mol):
        # function to print the potential, kinetic and total energy
        epot = mol.get_potential_energy() / len(mol)
        ekin = mol.get_kinetic_energy() / len(mol)
        #mol.center(vacuum=2.0)
        geometry=set_psi4(atoms=mol)
        psi4.set_options({'maxiter': 500})
        methodpsi4='MP2/aug-cc-pVTZ'#'HF/6-31G(d)'#'BLYP-D3BJ/6-31G''HF/aug-cc-pVTZ'#
        E=psi4.energy(methodpsi4)* psi4.constants.hartree2kcalmol
        dipx=psi4.variable('SCF DIPOLE X')
        print( methodpsi4+" Energy: ",E," kcal/mole")
        dipy=psi4.variable('SCF DIPOLE Y')
        dipz=psi4.variable('SCF DIPOLE Z')
        diptot=sqrt(dipx**2+dipy**2+dipz**2)
        print('dipole components: x= %.4f y= %.4f z= %.4f total= %.4f' %(dipx,dipy,dipz,diptot))
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
                'Etot = %.3feV E = %.3f kcal/mol ' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, E))
        return E,dipx,dipy,dipz
# Subroutine to get xyz file output from PSI4 molecule
def xyzgeo(Atoms):
    geo=''
    for atom in Atoms:
         temp = '{}\t{:.15f}\t{:.15f}\t{:.15f}\n'.format(atom.symbol,atom.position[0],atom.position[1],atom.position[2])
         geo += temp
    return geo
# use ASE to get indices for particular symbols
def get_indices(atoms, symbol):
    indices = []
    for atom in atoms:
        if atom.symbol == symbol:
            indices.append(atom.index)
    return indices
## Functions for dipole autocorrelation
def calc_derivative(array_1D, delta_t):
   # Use finite difference
    dy = np.gradient(array_1D)
    return np.divide(dy, delta_t)
def autocorr(x):
    yunbiased = x - np.mean(x, axis=0)
    ynorm = np.sum(np.power(yunbiased,2), axis=0)  # normalize largest peak
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
        window_function = signal.gaussian(len(data), std=std, sym=False)

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
#### FFT with window
####
    WE = np.sum(window) / len(array_1D)
    wf = window / WE
    # convolve the blackman-harris or other window function.
    sig = array_1D * wf
    # A series of number of zeros will be padded to the end of the \
    # VACF array before FFT.
    N = zero_padding(sig)
    yfft=fftpack.fft(sig)/len(sig)
    print("shape of yfft {:}".format(np.shape(yfft)))
    #return np.square(np.absolute(yfft))
    return np.abs(yfft)

################################
## END Functions ###############
################################
## CODE ########################
################################
# Read input file from command line argument
if len(sys.argv)>1:
    inpfile=sys.argv[1]
else:
    # Default local file
    inpfile='mlpes.inp'
# Read parameters
print("\n\nMLPES_MD_IR.py")
print("++++++++++++++++++")
print("ver. 1.01 JMS 19 May 2020\n\n")
### Grab input file to direct calculation
try:
    with open(inpfile,'r') as inp:
        # Compressed ML filename (.npz)
        print("started reading file: ", inpfile)
        MLfile=inp.readline().rstrip()
        print("Compressed ML filename: ",MLfile)
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
        # Read number of trajectories
        ntraj=int(inp.readline().rstrip())
        print("Number of trajectories: ",ntraj)
        # Read steps/trajectory
        steps=int(inp.readline().rstrip())
        print("Steps: ",steps)
        # Read time step (fsec)
        tstep=float(inp.readline().rstrip())
        print("Time step: ",tstep,' fsec, Nyquist@4500 cm-1 ~ 0.6 fsec/step')
        # Read starting .xyz geometry
        xyzfile=inp.readline().rstrip()
        print("Geometry file (xyz): ",xyzfile)
        # Read output filename
        outfile=inp.readline().rstrip()
        print("Output file name: ",outfile)
        # Read mode sampling 
        modenums=inp.readline().rstrip()
        modenums.replace("modes=","")
        if not modenums:
            modesamp=False
        else:
            modesamp=True # Bool for mode sampling
        # Read if molecule is non-linear or linear
        linear=inp.readline().rstrip()
        if "non-linear" in linear or "nonlinear" in linear:
            linearmol=False
        else:
            linearmol=True
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
lf.write("\n\nMLPES_MD_IR.py\n")
lf.write("++++++++++++++++++\n")
lf.write("ver. 1.01 JMS 19 May 2020\n\n")
lf.write("Date: %s\n"  %date.today())
timenow=time.strftime(" %H:%M:%S")
lf.write(" %s\n"  %timenow)
lf.write("\nINPUT FILE\n++++"+inputfiletext+"\n++++\n")
# Set up sGDML ASE calculator
calc = SGDMLCalculator(MLfile)
mol = read(xyzfile)    # [Consider randomized starting point]
print("Molecule file: ",mol)
print("Cartesians: ",mol.get_positions())
print("Atomic numbers: ",mol.get_atomic_numbers())
print("Distances: ",mol.get_all_distances())
imagefile=outfile+"_initial_geo.png"
write(imagefile,mol)
cartpos=np.array(mol.get_positions())
print(cartpos)
print(np.shape(cartpos))
print("Distances: ",mol.get_all_distances())
numatm=np.shape(cartpos)[0]
print("number of atoms: ",numatm)
new_indices = []
print("Atomic numbers: ",mol.get_atomic_numbers())
perm=sortperm(mol) # Permutations for dipole surface
if MLDIP:
    cartpos=cartpos[perm]
    cartpos=np.reshape(cartpos,3*numatm)
    print(cartpos)
    diptot=pes(cartpos,cartesian=True)
    print("Total dipole: ",diptot)
calc = Psi4(atoms = mol,method = 'b3lyp',memory = '500MB',basis = '6-311g_d_p_')
#mol.set_calculator(calc)
#qn = QuasiNewton(mol)  #From ASE
#qn.run(4e-3, 700)      #Optimize initial structure 
#vib = Vibrations(mol)
#vib.run()
print("Distances after optimization: ",mol.get_all_distances())

#vib.summary() # print a summary of the vibrational frequencies
imagefile=outfile+"_opt_geo.png"
write(imagefile,mol)
write(outfile+"_opt_geo.xyz",mol)
calc = SGDMLCalculator(MLfile)
mol.set_calculator(calc)
qn = QuasiNewton(mol)  #From ASE
qn.run(4e-3, 700)      #Optimize initial structure 
vib = Vibrations(mol)
vib.clean()
vib.run()
sdist=np.array_str(mol.get_all_distances())
print("Distances after 2nd optimization:\n "+sdist+"\n")
smasses= np.array_str(mol.get_masses())
lf.write("\nMasses:\n"+smasses+"\n")
lf.write("\nIntramolecular distances:\n"+sdist+"\n")
imagefile=outfile+"_sGDML_geo.png"
write(imagefile,mol)
write(outfile+"_sgdml_opt_geo.xyz",mol)
vib.summary() # print a summary of the 
print("++++\n")
s = 0.01 * units._e / units._c / units._hplanck # Conversion factor
lf.write("\n++++++\nVibrational Analysis (ASE):\n")
lf.write('---------------------\n')
lf.write('  #      cm^-1\n')
lf.write('---------------------\n')
for n, e in enumerate(vib.hnu):
    if e.imag != 0:
        c = 'i'
        e = e.imag
    else:
        c = ' '
        e = e.real
    lf.write('%3d  %7.1f%s\n' % (n, s * e, c))
lf.write('---------------------\n')
lf.write('Zero-point energy: %.3f eV\n++++++\n' % 
        vib.get_zero_point_energy())
# END writing vibrations to log file
###################################
# Now get ready for the iterations
# Send PSI4 output to file 
psi4.core.set_output_file(outfile+'_output.dat', False)
# Run Langevin NVT to warm up
# Room temperature simulation
#MaxwellBoltzmannDistribution(mol, temp * units.kB) 
Stationary(mol) # zero linear momentum
ZeroRotation(mol) # zero angular momentum
# If we want Langevin first...
#dyn = Langevin(mol, tstep/2 * units.fs, units.kB * temp, 0.002)
#dyn.run(300)
lf.write("\n%d Trajectories" %ntraj)
## ITERATE through trajectories
for tr in range(ntraj):
    lf.write("\nTrajectory %d\n" %tr)
    # Set the distribution
    dtemp=temp/5
    mol=read(outfile+"_opt_geo.xyz") # Reset the geometry for next traj
    mol.set_calculator(calc)
    MaxwellBoltzmannDistribution(mol, dtemp * units.kB) 
    Stationary(mol) # zero linear momentum
    ZeroRotation(mol) # zero angular momentum
    # NVT first
    dyn = Langevin(mol, tstep/2 * units.fs, units.kB * dtemp, 0.02)
    dyn.run(1000)
    Stationary(mol) # zero linear momentum
    #ZeroRotation(mol) # zero angular momentum
    # Normal mode sampling
    vib = Vibrations(mol)
    vib.clean()
    vib.run()
    vib.summary() 
    masses=mol.get_masses()
    num=len(masses)
    # non-linear modes= 3*num-6
    nmodes=3*num-6
    if linearmol:
        print("Linear molecule specified...")
        nmodes=3*num-5
    vmodes=list(range(-1,-(nmodes+1),-1))
    print(nmodes," vibrational modes:\n",vmodes)
    vib.write_mode()
    pe0=mol.get_potential_energy().item(0)
    ke0=mol.get_kinetic_energy().item(0)
    for n in range(-1,-(nmodes+1),-1):
        quanta=0
        chanceupper=math.exp(-(vib.hnu[n])/(units.kB*temp))
        chanceupper2=math.exp(-(2*vib.hnu[n])/(units.kB*temp))
        print("Chance upper: %5.3f" %chanceupper)
        rand=np.random.random() 
        if rand < chanceupper:
            quanta=1
        elif rand < chanceupper2:
            quanta=2
        else:
            quanta=0
        #quanta=np.random.randint(0,2) #Does not include endpoint
        print("Mode: ",n,"\n",vib.get_mode((3*num+n)))
        ##if n==-9: # To set modes arbitrarily
        ##    quanta=1
        ##else:
        ##    quanta=0 "
        print("Normal mode %d sampling %d quanta; frequency %5.3f cm-1" %(n,quanta,vib.hnu[n]/units.invcm))
        lf.write("Normal mode %d sampling %d quanta; frequency %5.3f cm-1\n" %(n,quanta,vib.hnu[n]/units.invcm))
        lf.write("Probability of 1 quanta: %5.3f; 2 Quanta: %5.3f; rand: %5.3f\n" %(chanceupper,chanceupper2,rand))
        omega, modes=Heigen(vib.H,masses)
        if vib.hnu[n]/units.invcm >100.0: # Skip lowest en mode(large disp)
            v,d=NMSample(modes,n,omega,quanta,np.random.random(),masses)
            mol.positions += d
            vold=mol.get_velocities()
            vnew=vold+v
            mol.set_velocities(vnew)
    print("Normal mode sampling completed")  
    pe1=mol.get_potential_energy().item(0) # Not sure why value comes as np
    ke1=mol.get_kinetic_energy().item(0)
    deltaE=pe1-pe0
    print(pe0,pe1)
    print(f"Potential Energy {pe0:.3f} {pe1:.3f} difference after displacement {(pe1-pe0):.3f} eV")
    print(f"Kinetic energy: {ke0:.3f} {ke1:.3f}")
    print(f"Total energy (PE+KE) = {(ke1-ke0+pe1-pe0):.3f} eV  {(ke1-ke0+pe1-pe0)*8065:.3f} cm-1")
    lf.write(f"Potential Energy (before,after): {pe0:.3f} {pe1:.3f}\n...difference after displacement {deltaE*8065:.3f} cm-1\n")
    lf.write(f"Kinetic energy (before,after): {ke0:.3f} {ke1:.3f} eV\n")
    lf.write(f"Total energy injected(PE+KE) = {(ke1-ke0+pe1-pe0):.3f} eV\n...  {(ke1-ke0+pe1-pe0)*8065:.3f} cm-1\n")
    # Now NVE
    dyn = VelocityVerlet(mol, tstep * units.fs,trajectory='md_'+outfile+str(tr)+'.traj',logfile=outfile+'md.log') 
    fdipole=open('dipole_'+outfile+str(tr)+'.traj','w')
    print("Mode sampling: ",modesamp)
    ### START TRAJECTORY ###
    for i in range(steps):
        dyn.run(1)
        print("Step : ",tr,":",i)
        try: #If SCF fails try new step
            if MLDIP:
                #print("loading ML dipole surface")
                perm=sortperm(mol) # Permute for dipole surface
                cartpos=np.array(mol.get_positions())
                cartpos2=cartpos[perm]
                cartlist=np.reshape(cartpos2,3*numatm)
                #print("Coordinates: ",cartpos)
                print("Distances: ",mol.get_all_distances())
                diptot=pes(cartlist,cartesian=True)
                print("Dipole total: ",diptot)
                dipx=dipy=dipz=0
                PE=mol.get_potential_energy()
                KE=mol.get_kinetic_energy()
                if modesamp:
                    print("Mode sampling...")
                    velocity=mol.get_velocities()
                    pm=[]
                    for k, n in enumerate(vmodes):
                        pm.append(projection(velocity,modes,n,omega,masses))
            else:
                PE,dipx,dipy,dipz=compute_E_dipole(mol)
                KE=mol.get_kinetic_energy()
                diptot=sqrt(dipx**2+dipy**2+dipz**2)
                if modesamp:
                    velocity=mol.get_velocities()
                    pm=[]
                    for k, n in enumerate(vmodes):
                        pm.append(projection(velocity,modes,n,omega, masses))

        except:
            print("Oops!", sys.exc_info()[0], "occured.")
            dyn.run(1)
            try:
                PE,dipx,dipy,dipz=compute_E_dipole(mol)
                KE=mol.get_kinetic_energy()
                diptot=sqrt(dipx**2+dipy**2+dipz**2)
            except:
                dyn.run(1)
                PE,dipx,dipy,dipz=compute_E_dipole(mol)
                KE=mol.get_kinetic_energy()
                diptot=sqrt(dipx**2+dipy**2+dipz**2)
                if modesamp:
                    velocity=mol.get_velocities()
                    pm=[]
                    for k, n in enumerate(vmodes):
                        pm.append(projection(velocity,modes,n,omega,masses))
        ddlist=[dipx,dipy,dipz,diptot]
        if i==0:
            energy=PE
            Kenergy=KE
            distance=mol.get_all_distances()
            dipderiv=np.hstack([float(i) for i in ddlist])
            if modesamp:
                pm=np.array(pm)
                smode=np.hstack([pm])
        else:
            energy=np.vstack([energy,PE])
            Kenergy=np.vstack([Kenergy,KE])
            distance=np.vstack([distance,mol.get_all_distances()])
            nprow=np.array([float(i) for i in ddlist])
            dipderiv=np.vstack([dipderiv,nprow])
            if modesamp:
                pm=np.array(pm)
                smode=np.vstack([smode,pm])
        fdipole.write('%.7f DIPOLE [Non Periodic](Debye)| X=  %.6f Y=  %.6f Z=   %.6f Total=  %.6f \n' % (PE,dipx,dipy,dipz,diptot))
        geo=xyzgeo(mol)
        fdipole.write(geo)
    ###     NEXT STEP       ###
    ### Trajectory Finished ###
    endtime=tstep*steps
    np_times=np.linspace(0.0,endtime,num=steps)
    t0=int(endtime/10*tstep)
    print("Time zero (after equilibration): ",t0)
    imagefile=outfile+"_traj"+str(tr)+"_geo.png"
    write(imagefile,mol)
    ddipolex=dipderiv[t0:,0] #calc_derivative(nddipole[t0:,0], time_step)
    ddipoley=dipderiv[t0:,1] #calc_derivative(nddipole[t0:,1], time_step)
    ddipolez=dipderiv[t0:,2]  #calc_derivative(nddipole[t0:,2], time_step)
    ddipoletot=dipderiv[t0:,3]  #
#    ddipoletot=calc_derivative(dipderiv[t0:,3], tstep) # Try alternative
#    ddipoletot=ddipolez+ddipoley+ddipolex
    dacf=autocorr(ddipoletot) 
    # Prepare for FFT
    zp=zero_padding(dacf)
    timelength=float(np_times[-1])
    width=len(dacf)/8 # /4 works since the data width will be ~2 sigma 
    wf_name="Hann"
    wf_name="Gaussian"
    window = choose_window(dacf, wf_name,width)
    print("dacf shape: ",dacf.shape,window.shape)
    # Quick plot of window and time domain signal
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
    fig,axs=plt.subplots(4,1, tight_layout=True)
    x=np.arange(0,len(dacf)*tstep,tstep)
    # Plot current dipole autocorrelation function
    axs[0].plot(x,dacf)
    axs[0].plot(x,window)
    axs[0].set_xlabel('Time [fsec]')
    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].tick_params(which='both', width=1)
    axs[0].tick_params(which='major', length=7)
    axs[0].tick_params(which='minor', length=4, color='m') 
    # Plot current dipole autocorrelation function * window
    axs[1].plot(x,window*dacf)
    axs[1].set_xlabel('Time [fsec]')
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1].tick_params(which='both', width=1)
    axs[1].tick_params(which='major', length=7)
    axs[1].tick_params(which='minor', length=4, color='m') 
     # Plot current pootential and kinetic energy
    energy=energy[:x.shape[0]] # Trim
    Kenergy=Kenergy[:x.shape[0]]
    axs[2].plot(x,energy, label='Potential Energy')
    axs[2].legend()
    axs[2].set_xlabel('Time [fsec]')
    axs[2].xaxis.set_minor_locator(AutoMinorLocator())
    axs[2].tick_params(which='both', width=1)
    axs[2].tick_params(which='major', length=7)
    axs[2].tick_params(which='minor', length=4, color='m')
    axs[3].plot(x,Kenergy, label='Kinetic Energy')
    axs[3].legend()
    axs[3].set_xlabel('Time [fsec]')
    axs[3].xaxis.set_minor_locator(AutoMinorLocator())
    axs[3].tick_params(which='both', width=1)
    axs[3].tick_params(which='major', length=7)
    axs[3].tick_params(which='minor', length=4, color='m') 
    fig.savefig(outfile+"_MLPES_timedomain.png")
    # Now take FFT
    powerHann=calc_FFT(dacf, window)[0:int(dacf.size / 2)]
    smode=smode.reshape((-1,nmodes))
    nstp = min(dacf.shape[0],  window.shape[0])
    smode=smode[:nstp,:]
    window = choose_window(smode, wf_name,width)
    projectFFT=[]
    for k, n in enumerate(vmodes):
        projectFFT.append(calc_FFT(smode[:,n],window)[0:int(dacf.size / 2)]) # Mode project
    projectFFT=np.array(projectFFT)
    end_time=float(np_times[-2])+tstep
    start_time=float(np_times[t0])
    c = 2.9979245899e-5 # speed of light in vacuum in [cm/FSEC]
    hbar=units._hplanck/(2*np.pi)
    kb=units._k
    print("Constants used (c, hbar, kb): ",c,hbar, kb)
    wavenumber = fftpack.fftfreq(dacf.size, tstep * c)[0:int(dacf.size / 2)]
    sample_omega=np.fft.fftfreq(dacf.size, tstep*2 * np.pi/1000)[0:int(dacf.size / 2)]
    IR_intensityHann=powerHann*sample_omega*np.tanh(hbar*1.0E15*sample_omega/(kb*temp))
    fig,axs = plt.subplots(1, 1, tight_layout=True)
    axs.plot(wavenumber,IR_intensityHann,'-r', label=outfile+str(tr))
    axs.grid(True)
    axs.set_xlim(xmin, xmax)
    correct_limit(axs, wavenumber,IR_intensityHann)
    axs.set_xlabel('Frequency [cm^-1]')
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    axs.legend()
    axs.set_ylabel('IR_intensity')
    fig.savefig(outfile+str(tr)+'_IRNW'+'.png')
    # Now plot projections
    fighandle=plotspec(wavenumber,projectFFT,[xmin,xmax],title='projection on normal modes')
    fighandle.savefig(outfile+str(tr)+'_mode_plotspec'+'.pdf')
    mspectrum=np.vstack((wavenumber,projectFFT)).T
    spectrum=np.vstack((wavenumber,IR_intensityHann,powerHann)).T
    #print(spectrum.shape)
    if tr==0:
        specsum=spectrum
        mspecsum=mspectrum
    else:
        specsum=np.add(specsum,spectrum)
        mspecsum=np.add(mspecsum,mspectrum)
        #print(specsum.shape)
    np.savez(outfile+str(tr)+'_.npz',energy=energy,distance=distance)
    np.savetxt(outfile+'_'+str(tr)+'_spec.csv', spectrum, delimiter=',', fmt='%15.11f')
    fdipole.close()
    write('md_'+outfile+str(tr)+'.xyz', read('md_'+outfile+str(tr)+'.traj',index=':'))
## End set of MD trajectories
specsum=specsum/(tr+1)
mspecsum=mspecsum/(tr+1)
np.savetxt(outfile+'_avg_spec.csv', specsum, delimiter=',', fmt='%15.11f')
np.savetxt(outfile+'_avg_mspec.csv', mspecsum, delimiter=',', fmt='%15.11f')
## Plot averaged spectrum
fig,axs = plt.subplots(1, 1, tight_layout=True)
axs.plot(wavenumber,specsum[:,1],'-b', label=outfile+" Average "+str(tr))
axs.grid(True)
axs.set_xlim(xmin,xmax)
correct_limit(axs, wavenumber,specsum[:,1])
axs.set_xlabel('Frequency [cm^-1]')
fig.legend(fontsize=7)
axs.set_ylabel('IR_intensity')
fig.savefig(outfile+'_avg_IRNW'+'.png')
fighandle=plotspec(wavenumber,mspecsum.T[1:,:],[250,4500],title='Projection on normal modes avg')
fighandle.savefig(outfile+'_mode_plotspec_avg'+'.pdf')

print("+++COMPLETED+++\nPlot saved as: ",outfile+'_IRNW'+'.png',"\nData saved as: ",outfile+'_avg_spec.csv')
timenow=time.strftime(" %H:%M:%S")
lf.write("\n\nFINISHED!\n %s\n"  %timenow)
lf.close()




















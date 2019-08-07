import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
from scipy import interpolate as intp
import sys

# Fundamental constant [cgs units]
h  = 6.6260755e-27    # planck constants
kB = 1.380658e-16     # Boltzmann constant
c  = 2.99792458e10    # speed of light
me = 9.10938291e-28   # electron mass
e  = 4.80320425e-10   # electron charge
mH = 1.67e-24         # proton mass
sigma0 = 0.0263       # Cross section [cm^2/sec] ??

# Units Conversion
cm_km = 1.e-5  # Convert cm to km
km_cm = 1.e5   # Convert km to cm
ang_cm = 1.e-8 # Convert Angstrom to cm

class Transition:
    def __init__(self,name,wave,osc_f,gamma,mass):
        self.name  = name
        self.wave  = wave
        self.osc_f = osc_f
        self.gamma = gamma
        self.mass  = mass


def ReadTransitionData():
	"""
	e.g., transition_dict['CIVa']
	"""
	data_file = './data/atomic_data.dat'
	amu = 1.66053892*1e-24   # 1 atomic mass in grams
	
	name  = np.loadtxt(data_file, dtype=str, usecols=[0])
	wave  = np.loadtxt(data_file, usecols=[1])
	osc_f = np.loadtxt(data_file, usecols=[2])
	gamma = np.loadtxt(data_file, usecols=[3])
	mass  = np.loadtxt(data_file, usecols=[4]) * amu

	transition_dict = {}
	for i in xrange(len(name)):
		transition_dict[str(name[i])] = Transition(name[i],wave[i],osc_f[i],gamma[i],mass[i])
	return transition_dict

if __name__ == '__main__':

	transition_dict = ReadTransitionData()
	print transition_dict['CIVa'].wave

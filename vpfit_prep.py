
"""
Todo: 

"""
import sys
import os
import numpy as np
#import pyfits as pf
import matplotlib.pyplot as pl
from scipy.interpolate import BarycentricInterpolator,interp1d

from atomic_data import ReadTransitionData

import Utilities

# My default matplotlibrc setting uses tex. tex makes plotting slow
pl.rc('font', family='Bitstream Vera Sans')
pl.rc('text', usetex=False)

#For interactive key_press_event
pl.switch_backend('TkAgg')


def Write_NormSpec_ascii(wave,flux,error,ion_name):
	fname = output_path + '/' + ion_name + '.dat'
	f = open(fname,'w')
	f.write('# wave flux error\n')
	for i in xrange(len(wave)):
		f.write('%f\t%f\t%f\n' % (wave[i],flux[i],error[i]))

	f.close()



def Write_NormSpec_Fits(spec,fname,LSF_header):
	wave,flux,sig = spec
	pf.writeto(output_path + '/' +fname +'.fits', flux,clobber=True)
	pf.writeto(output_path + '/' +fname +'.wav.fits', wave,clobber=True)
	pf.writeto(output_path + '/' +fname +'.sig.fits', sig,clobber=True)

	# Add a header to flux fits file
	flux_hdu = pf.open(output_path + '/' +fname +'.fits')
	flux_header = flux_hdu[0].header
	flux_header.set('RESFILE',LSF_header)
	os.remove(output_path + '/' +fname +'.fits')
	pf.writeto(output_path + '/' +fname +'.fits', flux,flux_header, clobber=True)

def read_spec(path,transition,redshift):
	"""
	Read continuum-normalized spectrum created by the IDL 
	routine. 
	"""
	flux_file = path + '/' + 'Norm-flux.fits'
	wave_file = path + '/' + 'wave.fits'
	sig_file  = path + '/' + 'Norm-df.fits'
	flux_hdu = pf.open(flux_file); 
	flux = flux_hdu[0].data
	wave = pf.open(wave_file)[0].data
	sig = pf.open(sig_file)[0].data
	spec = wave,flux,sig

	flux_header = Select_LSF(transition,redshift)
	LSF_header = flux_hdu[0].header
	LSF_header.set('RESFILE',flux_header)
	
	return spec,LSF_header

def read_spec_normal(input_path, qso_name, transition,redshift):

	fname = input_path + '/norm_' + qso_name + '.spec'
	wave,flux,error = np.loadtxt(fname,unpack=True,usecols=[0,1,2])
	inds = np.where(~np.isnan(flux))[0]
	wave = wave[inds]; 
	flux = flux[inds];
	error = error[inds];
	error = error / np.median(flux)
	flux = flux / np.median(flux)

	spec = wave,flux,error
	flux_header = Select_LSF(transition,redshift)
	return spec,flux_header


def Select_LSF(transition,redshift):
	"""
	Create a header name based on the observed wavelength 
	of the transition in interest. 
	"""
	transition_dict = ReadTransitionData()
	obs_wave = transition_dict[transition].wave * (1+redshift)
	LSF_range = str(int(round(obs_wave/50.,0)*50)) # round to nearest 50; convert to string
	
	LSF_file = 'COS_res' + LSF_range + '.dat'
	return LSF_file

def SelectDataRange(spec,transition,redshift,dwave = 20):
	"""
	Select spectrum given the transition and the redshift. 
	"""
	wave,flux,sig = spec
	transition_dict = ReadTransitionData()
	obs_wave = transition_dict[transition].wave * (1+redshift)
	inds = np.where((wave > obs_wave-dwave) & (wave < obs_wave+dwave))[0]

	return wave[inds],flux[inds],sig[inds]

	
def plot_spec(spec,transition_name, line_region, LSF_header, ion_name, 
			  dwave = 10, dflux_window_up = 0.0, dflux_window_down=0.0):
	"""
	Usage/Precedures: 

	1. Press 'a' to select two points for data to be considered
	2. Press 'C' local continuum fit - and write out new spectra
		-- wave,flux,error with name in vpfit style. 
	3. Press 'G' to a gaussian to get estimate b, and z. and print out to screen
	4. Press 'W' to write out fort.13 file for vpfit based on the gaussian fit
		-- also write out the continuum-normalized ion.fits with COS LSF header
	5. Press 'w' to append (N,b,z) parameters to fort.13 
	"""

	wave,flux,error = spec

	line_region = np.median(wave)

	fig1 = pl.figure(figsize=(16,8))
	ax1 = fig1.add_subplot(111)
	ax1.clear()
	ax1.set_xlabel(r'$\lambda$ ($\AA$)')
	ax1.set_ylabel(r'$\rm Flux$')

	# Rest central wavelength vertical line
	obs_central_wave = transition_dict[transition_name].wave * (1+redshift)
	pl.axvline(obs_central_wave,lw=2,ls='--',color = 'b')

	# Original data. 
	pl.step(wave,flux,color = 'k')
	pl.step(wave,error,color = 'r')
	pl.axhline(1,ls='--',lw=1.5)
	pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])

	# New data: spec after local continuum fitting. 
	normwave = []; normflux = []; normdf = []; 
	norm_flux_line, = pl.step(normwave,normflux,color = 'b')
	norm_dflux_line, = pl.step(normwave,normdf,color = 'g')

	# Two points setting boundary of data for calculation 
	x_record = []; y_record = []
	points_to_fit, = pl.plot(x_record,y_record,'ro',ms = 8)

	# Linear line estimate for local continuum 
	cont_wave = []; cont_flux = []
	new_continuum, = pl.plot(cont_wave,cont_flux,'b',lw = 1.5)

	# Gaussian Lines
	gauss_wave = []; gauss_flux = []
	gauss_fit, = pl.plot(gauss_wave,gauss_flux,'g',lw = 2.)

	dwave = 10
	big_shift = 0.5; small_shift = 0.1
	big_zoom  = 0.5; zoom        = 0.1
	dflux_window_up = 0.0; dflux_window_down = 0.0
	flux_zoom = 0.01; 		Big_flux_zoom   = 0.05; 
	

	def shift_spec(event):
		#global transition_name
		global line_region
		global dwave
		global dflux_window_up
		global dflux_window_down		
		ix, iy = event.xdata, event.ydata

		##########################################################
		##################### WINDOW CONTROL #####################
		##########################################################
		if event.key == '}':
			line_region += big_shift
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
		elif event.key == '{':
			line_region -= big_shift
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
		elif event.key == ']':
			line_region += small_shift
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
		elif event.key == '[':
			line_region -= small_shift
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
		elif event.key == '-':
			dwave += zoom
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
		elif event.key == '=':
			dwave -= zoom
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
		elif event.key == '_':
			# shift -
			dwave += big_zoom
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
		elif event.key == '+':
			dwave -= big_zoom
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
		elif event.key =='b':
			dflux_window_up += flux_zoom
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])
		elif event.key =='B':
			dflux_window_up -= flux_zoom
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])
		elif event.key =='t':
			dflux_window_down -= flux_zoom
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])			
		elif event.key =='T':
			dflux_window_down += flux_zoom
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])
		elif event.key =='m':
			dflux_window_up += Big_flux_zoom
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])
		elif event.key =='M':
			dflux_window_up -= Big_flux_zoom
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])
		elif event.key =='u':
			dflux_window_down -= Big_flux_zoom
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])			
		elif event.key =='U':
			dflux_window_down += Big_flux_zoom
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])
		elif event.key =='r':
			dwave = 10
			dflux_window_up = 0.0
			dflux_window_down = 0.0
			line_region = np.median(wave)
			pl.xlim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[0])
			pl.ylim(Utilities.zoom_region(line_region,dwave,dflux_window_up,dflux_window_down)[1])
		elif event.key == 'k':
			print '\n'
			Utilities.printLine()
			print 'k	Show keys map (What is shown here)'

			Utilities.printLine()
			print 'WINDOW CONTROL KEYS:'
			print '}		shift to right with 0.5 Angstrom'
			print '{		shift to left with 0.5 Angstrom'
			print ']		shift to right with 0.1 Angstrom'
			print '[		shift to left with 0.1 Angstrom'
			print 'shift +/-	Zoom in/out by 0.5 Angstrom'
			print '+/-		Zoom in/out by 0.1 Angstrom'
			print 'T/t		Zoom top by 1e-15'
			print 'B/b		Zoom bottom by 1e-15'
			print 'U/u		Zoom top by 5e-15'
			print 'M/m		Zoom bottom by 5e-15'
			print 'r		replot'

			Utilities.printLine()
			print 'FITTING SPEC KEYS:'
			print 'a		Add points'
			print 'shift+g		Fit Gaussian'
			print 'shift+c		Fit Continuum'
			print 'shift+w		Write fits file and fort.13'
			print 'w		Append guess (N,b,z) to fort.13'
			Utilities.printLine()		

		##########################################################
		##################### Fitting Control ####################
		##########################################################
		elif event.key == 'a':
			'''Add 2 Points setting boundary for fitting data''' 
			if len(x_record) < 10:
				x_record.append(ix)
				y_record.append(iy)
			else:
				del x_record[:]; del y_record[:]
				x_record.append(ix)
				y_record.append(iy)
		elif event.key == 'C':
			"""
			spline interpolation of the points to get 
			the continuum 
			"""

			spl_cont = BarycentricInterpolator(x_record,y_record)

			x1 = np.min(x_record); x2 = np.max(x_record)	
			edges = [x1,x2]
			temp_wave,temp_flux,temp_error = Utilities.Select_Data(spec,edges)
			cont_flux = spl_cont(temp_wave)

			new_continuum.set_xdata(temp_wave)
			new_continuum.set_ydata(cont_flux)

			normwave = temp_wave
			normflux = temp_flux / cont_flux
			#normdf = temp_error * (temp_flux / cont_flux)
			normdf = temp_error / (cont_flux)

			norm_flux_line.set_xdata(normwave)
			norm_flux_line.set_ydata(normflux)
			norm_dflux_line.set_ydata(normdf)
			norm_dflux_line.set_xdata(normwave)

		elif event.key == 'G':
			"""
			Fit a gaussian for inspection
			"""
			if not x_record: 
				print 'No data selected to fit.'
				pass
			else:
				print 'transition_name = ', transition_name
				p1,p2 = np.transpose(np.array([x_record,y_record]))

				x1,y1 = p1; x2,y2 = p2
				temp_spec = Utilities.Select_Data(spec,[x1,x2])
				estimated_cont_level = np.mean((y1,y2))

				gauss_params = Utilities.Fit_Gaussian(temp_spec,estimated_cont_level)


				if gauss_params:
					amplitude,centroid_wave,sigma_width = gauss_params

					# Apparent column density 
					logN,dlogN = Utilities.ComputeAppColumn(temp_spec,transition_name)

					# Print out results of gaussian fit 
					Utilities.Print_LineInfo(gauss_params,logN,transition_name)

					# Make the plot to show goodness of fit
					gauss_flux = Utilities.Gaussian_function(temp_spec[0],amplitude,centroid_wave,sigma_width)
					gauss_wave = temp_spec[0];
					gauss_fit.set_xdata(gauss_wave)
					gauss_fit.set_ydata(gauss_flux + estimated_cont_level)
		
		elif event.key == 'W':
			"""
			Fit and write continuum-normalized ion.fits with header
			Also write the filename and range of wavelength for fort.13
			"""

			spl_cont = BarycentricInterpolator(x_record,y_record)
			 
			x1 = np.min(x_record); x2 = np.max(x_record)	
			edges = [x1,x2]
			temp_wave,temp_flux,temp_error = Utilities.Select_Data(spec,edges)
			cont_flux = spl_cont(temp_wave)

			new_continuum.set_xdata(temp_wave)
			new_continuum.set_ydata(cont_flux)

			normwave = temp_wave
			normflux = temp_flux / cont_flux
			normdf = temp_error  / cont_flux
			
			norm_flux_line.set_xdata(normwave)
			norm_flux_line.set_ydata(normflux)
			norm_dflux_line.set_ydata(normdf)
			norm_dflux_line.set_xdata(normwave)

			Write_NormSpec_ascii(normwave, normflux, normdf,transition_name)

		elif event.key == 'E':
			if not x_record: 
				print 'No data selected for calculation.'
				pass
			else:
				p1,p2 = np.transpose(np.array([x_record,y_record]))
				x1,y1 = p1; x2,y2 = p2
				edges = [x1,x2]
				temp_spec = Utilities.Select_Data(spec,edges)
				W0 = Utilities.ComputeEquivWdith(p1,p2,temp_spec)

				print 'W0 = %s mA' % W0

		points_to_fit.set_xdata(x_record)
		points_to_fit.set_ydata(y_record)
		pl.draw() # needed for instant response. 

	civ = fig1.canvas.mpl_connect('key_press_event', shift_spec)
	pl.show()

	return 


if __name__ == '__main__':

	#qso_path = '/Users/CameronLiang/data/uvqso_archive/MRK106'
	if len(sys.argv) != 3: 
		print 'python vpfit_prep.py input_path qso_name'
	else:
		input_path = sys.argv[1] 
		qso_name = sys.argv[2]

		#qso_name = sys.argv[1]
		#input_path = '/Users/cameronliang/research/projects/metallicity/data/disk_qsos/'
		#input_path = input_path + qso_name + '/reduced/'
		
		#qso_path = '/Users/CameronLiang/data/uvqso_archive/' + qso_name
		dwave = 10; dflux_window_up = 0.0; dflux_window_down = 0.0; 
		transition_dict = ReadTransitionData()

		output_path = input_path

		Utilities.PrintIonsNames()
		transition_name, ion_name,redshift,line_region = Utilities.Vpfit_UserInputs()

		spec,LSF_header = read_spec_normal(input_path,qso_name,transition_name,redshift)
		spec = SelectDataRange(spec,transition_name,redshift)
		
		plot_spec(spec,transition_name,line_region, LSF_header, ion_name, dwave = 10)





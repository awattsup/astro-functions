import numpy as np
import numpy.polynomial as npPoly
import math 
from math import gcd
import matplotlib.path as pltPath
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, Wcsprm
from astropy import units as u
try:
	from spectral_cube import SpectralCube
except:
	print('no spectral cube installed')
try:
	import requests
	from io import BytesIO
	from PIL import Image
except:
	print('cant get SDSS images')

import copy
from scipy.stats import ks_2samp,anderson_ksamp
import scipy.stats as sps
from scipy.stats import distributions
import os

import astropy.constants as ac
import astropy.units as au
from astropy.stats import sigma_clip



rng = np.random.default_rng()

#some useful things

emlines = { 'HI':{      'lambda':[21.106114], 'nu':[1420.405751], 'ratio':[1]},
            'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
            'Halpha':{	'lambda':[6562.819],			'ratio':[1]},

            'Pa9':{		'lambda':[9229.014],			'ratio':[1]},
            'Pa10':{	'lambda':[9014.909],			'ratio':[1]},
            'Pa11':{	'lambda':[8862.782],			'ratio':[1]},
            'Pa12':{	'lambda':[8750.472],			'ratio':[1]},
            'Pa13':{	'lambda':[8665.019],			'ratio':[1]},			#overlaps with CaT line!!
            'Pa14':{	'lambda':[8598.392],			'ratio':[1]},
            'Pa15':{	'lambda':[8545.383],			'ratio':[1]},			#overlaps with CaT line!!
            'Pa16':{	'lambda':[8502.483],			'ratio':[1]},			#overlaps with CaT line!!
            'Pa17':{	'lambda':[8467.254],			'ratio':[1]},
            'Pa18':{	'lambda':[8437.956],			'ratio':[1]},
            'Pa19':{	'lambda':[8413.318],			'ratio':[1]},
            'Pa20':{	'lambda':[8392.397],			'ratio':[1]},



            'OI':{		'lambda':[6300.304,6363.78],	'ratio':[1,0.33]},					#low ionisation 14.53
            'OI8446': {	'lambda':[8446.359],			'ratio':[1]},
            'OII':{		'lambda':[7319.990, 7330.730],	'ratio':[1,1]}, #?? check 			#low ionisation 13.62
            'OIII':{	'lambda':[4958.911, 5006.843],	'ratio':[0.35,1]}, 					#high ionisation 35.12

            'NI':{ 		'lambda':[5200.257],			'ratio':[1]},
            'NII':{		'lambda':[6548.050,6583.460],	'ratio':[0.34,1]},					#low ionisation 15.43


            'Fe4993':{	'lambda':[4993.358],			'ratio':[1]},
            'Fe5018':{	'lambda':[5018.440],			'ratio':[1]},

            'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
            'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
            'HeI7065':{'lambda':[7065.196], 			'ratio':[1]},
            'HeII4685':{'lambda':[4685.710],			'ratio':[1]}, 

            'SII6716':{	'lambda':[6716.440],			'ratio':[1]},						#low ionisation 10.36
            'SII6730':{	'lambda':[6730.810],			'ratio':[1]},						#^^
            'SIII9069':{'lambda':[9068.6],				'ratio':[1]},						#high ionisation 23.33

            'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 						#high ionisation 27.63 #doublet**
            'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},						#high ionisation 27.63
            'ArIV':{	'lambda':[4711.260],			'ratio':[1]}, 						#high ionisation 40.74 #doublet**
            'ArIV':{	'lambda':[4740.120],			'ratio':[1]},						#high ionisation 40.74
            }

emlines_indiv = {'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
				'Halpha':{	'lambda':[6562.819],			'ratio':[1]},
				
				'Pa9':{		'lambda':[9229.014],			'ratio':[1]},
				'Pa10':{	'lambda':[9014.909],			'ratio':[1]},
				'Pa11':{	'lambda':[8862.782],			'ratio':[1]},
				'Pa12':{	'lambda':[8750.472],			'ratio':[1]},
				'Pa13':{	'lambda':[8665.019],			'ratio':[1]},			#overlaps with CaT line!!
				'Pa14':{	'lambda':[8598.392],			'ratio':[1]},
				'Pa15':{	'lambda':[8545.383],			'ratio':[1]},			#overlaps with CaT line!!
				'Pa16':{	'lambda':[8502.483],			'ratio':[1]},			#overlaps with CaT line!!
				# 'Pa17':{	'lambda':[8467.254],			'ratio':[1]},
				'Pa18':{	'lambda':[8437.956],			'ratio':[1]},
				# 'Pa19':{	'lambda':[8413.318],			'ratio':[1]},
				# 'Pa20':{	'lambda':[8392.397],			'ratio':[1]},

				'OI6300':{	'lambda':[6300.304],	'ratio':[1]},					#low ionisation 14.53
				'OI6364':{	'lambda':[6363.78],		'ratio':[1]},					#low ionisation 14.53
				'OI8446': {	'lambda':[8446.359],			'ratio':[1]},
				'OII7312':{	'lambda':[7319.990],	'ratio':[1]}, #?? check 			#low ionisation 13.62
				'OII7330':{	'lambda':[7330.730],	'ratio':[1]}, #?? check 			#low ionisation 13.62
				'OIII4959':{'lambda':[4958.911],	'ratio':[1]}, 					#high ionisation 35.12
				'OIII5006':{'lambda':[5006.843],	'ratio':[1]}, 					#high ionisation 35.12

				'NI':{ 		'lambda':[5200.257],			'ratio':[1]},
			 	'NII6548':{		'lambda':[6548.050],	'ratio':[1]},					#low ionisation 15.43
			 	'NII6583':{		'lambda':[6583.460],	'ratio':[1]},					#low ionisation 15.43

			 	'Fe4993':{	'lambda':[4993.358],			'ratio':[1]},
			 	'Fe5018':{	'lambda':[5018.440],			'ratio':[1]},
			 	
				'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
				'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
				'HeI7065':{'lambda':[7065.196], 			'ratio':[1]},
				'HeII4685':{'lambda':[4685.710],			'ratio':[1]}, 

				'SII6716':{	'lambda':[6716.440],			'ratio':[1]},						#low ionisation 10.36
				'SII6730':{	'lambda':[6730.810],			'ratio':[1]},						#^^
				'SIII9069':{'lambda':[9068.6],				'ratio':[1]},						#high ionisation 23.33
				
				'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 						#high ionisation 27.63 #doublet**
				'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},						#high ionisation 27.63
				'ArIV4711':{	'lambda':[4711.260],			'ratio':[1]}, 						#high ionisation 40.74 #doublet**
				'ArIV4740':{	'lambda':[4740.120],			'ratio':[1]},						#high ionisation 40.74
				}


def inside_polygon(coordinates, poly_x, poly_y):
	#inspired by /copied from  the IDL-coyote routine 'inside'
	Npoly = len(poly_x)
	Ncoords = len(coordinates[:,0])

	poly_x = np.append(poly_x,poly_x[0])					#close polygon
	poly_y = np.append(poly_y,poly_y[0])

	poly_square = np.where((coordinates[:,0]>= np.min(poly_x)) &
				 	(coordinates[:,0]<= np.max(poly_x)) &
				 	(coordinates[:,1]>= np.min(poly_y)) &
				 	(coordinates[:,1]<= np.max(poly_y)))[0]
	
	theta_arr = np.zeros(Ncoords)
	for ii in poly_square:


		vec1_x = poly_x[0:-1] - coordinates[ii,0] 				#verticies 0 -> N-1
		vec1_y = poly_y[0:-1] - coordinates[ii,1]

		vec2_x = poly_x[1::] - coordinates[ii,0]					#vertices 1 -> N
		vec2_y = poly_y[1::] - coordinates[ii,1]

		dot_prod = vec1_x * vec2_x + vec1_y * vec2_y
		cross_prod = vec1_x * vec2_y - vec1_y * vec2_x

		theta = np.arctan(cross_prod/ dot_prod)
		for tt in range(len(theta)):
			if np.sign(dot_prod[tt]) == -1:
				theta[tt] += np.sign(cross_prod[tt])*np.pi

		theta_arr[ii] = np.sum(theta)



	# plt.imshow((theta_arr/np.pi).reshape((442,439)))
	# plt.show()
	# exit()

	in_polygon = np.where(np.abs(theta_arr) > 5)[0]					#get total of angles. will be near 2pi if in polygon

	return in_polygon


def inside_polygon_v2(coordinates,poly_x,poly_y):
	#borrowed/stolen from stack exchange: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

	path = pltPath.Path(np.array([poly_x,poly_y]).T)

	inside = path.contains_points(coordinates)

	return inside


def get_SDSS_image(RA, DEC, arcsec=90, RADEC=False,head=False):
	

	img_scale = 0.36
	if isinstance(arcsec,list):
		img_size = [int(arcsec[0]/img_scale),int(arcsec[1]/img_scale)] *2
	else:
		img_size = [int(arcsec/img_scale)] *2


	imgURL = f'http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpeg.aspx?ra={RA}&dec={DEC}&width={img_size[0]}&height={img_size[1]}&scale={img_scale}'
	response = requests.get(imgURL)
	img = Image.open(BytesIO(response.content))

	if RADEC:
		CDELT1 = -img_scale/3600.
		CDELT2 = img_scale/3600.
		NAXIS1 = img_size[0]
		NAXIS2 = img_size[1]
		CRPIX1 = NAXIS1/2
		CRPIX2 = NAXIS2/2
		CRVAL1 = RA
		CRVAL2 = DEC

		w = WCS(naxis=2)
		w.wcs.crpix = [CRPIX1,CRPIX2]
		w.wcs.cdelt = [CDELT1,CDELT2]
		w.wcs.crval = [CRVAL1,CRVAL2]
		w.wcs.ctype=["RA---SIN","DEC--SIN"]

		h = w.to_header()
		h['NAXIS1'] = NAXIS1
		h['NAXIS2'] = NAXIS2

		pix_xxyy, pix_RADEC, pix_SP = make_pix_WCS_grids(h)

		if head:
			return img, [pix_xxyy, pix_RADEC, pix_SP], h
		else:
			return img, [pix_xxyy, pix_RADEC, pix_SP]
			

	else:

		return img

def get_legacy_colour_image(RA, DEC,imSize = 300):
	"""Fetch the image at the given RA, DEC from the Legacy jpeg server"""
	#imSize should be in arcsec
	pix_scale = 0.262

	imsize = int(round(imSize/pix_scale))

	url = f"https://www.legacysurvey.org/viewer/jpeg-cutout/?ra={RA}&dec={DEC}&layer=ls-dr9&width={imsize}&height={imsize}&pixscale=0.262"
	print(url)
	r = requests.get(url)
	img = Image.open(BytesIO(r.content))

	CDELT1 = -pix_scale/3600.
	CDELT2 = pix_scale/3600.
	NAXIS1 = imSize
	NAXIS2 = imSize
	CRPIX1 = NAXIS1/2
	CRPIX2 = NAXIS2/2
	CRVAL1 = RA
	CRVAL2 = DEC

	w = WCS(naxis=2)
	w.wcs.crpix = [CRPIX1,CRPIX2]
	w.wcs.cdelt = [CDELT1,CDELT2]
	w.wcs.crval = [CRVAL1,CRVAL2]
	w.wcs.ctype=["RA---TAN","DEC--TAN"]

	h = w.to_header()
	h['NAXIS1'] = NAXIS1
	h['NAXIS2'] = NAXIS2

	pix_xxyy, pix_RADEC, pix_SP = make_pix_WCS_grids(h)

	return img, [pix_xxyy, pix_RADEC, pix_SP], h

def get_legacy_image(outfile, RA, DEC,imSize = 300,bands = 'r'):
    """Fetch the image at the given RA, DEC from the Legacy server"""
    #imSize should be in arcsec
    pix_scale = 0.262

    imsize = int(round(imSize/pix_scale))
    url = f"https://www.legacysurvey.org/viewer/fits-cutout/?ra={RA}&dec={DEC}&layer=ls-dr10&width={imsize}&height={imsize}&pixscale=0.262&bands={bands}"
    print(url)
   
    r = requests.get(url)
    if r.ok:
        with open(outfile, 'wb') as file:
            file.write(r.content)
    else:
        print('No coverage for (%s, %s)' % (RA, DEC))

def get_galex_image(outfile, RA, DEC,imSize = 300):
    """Fetch the image at the given RA, DEC from the Legacy server"""
    #imSize should be in arcsec

    pix_scale = 1.5

    imsize = int(round(imSize/pix_scale))

    url = f"https://www.legacysurvey.org/viewer/fits-cutout/?ra={RA}&dec={DEC}&layer=galex&width={imsize}&height={imsize}&pixscale=1.5"
    r = requests.get(url)
    if r.ok:
        with open(outfile, 'wb') as file:
            file.write(r.content)
    else:
        print('No coverage for (%s, %s)' % (RA, DEC))


def get_SDSS_mosaic_image(outName, RA, DEC, imSize=0.166667,filters = 'ugriz', scriptOnly=True):
	

	outName = os.path.expanduser(f'{outName}')
	
	url = f"https://dr12.sdss.org/mosaics/script?onlyprimary=False&pixelscale=0.396&ra={RA}&filters={filters}&dec={DEC}&size={imSize}"
	script = requests.get(url)
	if script.ok:
		with open(outName,'wb') as file:
			file.write(script.content)
	else:
		print('Mosaic fetch failed')

	if not scriptOnly:
		os.system(f"chmod +x {outName}")
		os.system(f"./{outName}")


def make_pix_WCS_grids_old(header):
		
	wcs =  WCS(header).celestial
	
	Nx = header['NAXIS1']
	Ny = header['NAXIS2']

	pix_xx, pix_yy = np.meshgrid(np.arange(Nx),np.arange(Ny),indexing='xy') #Naxis2=rows,Naxis1=cols


	if method == 'astropy':
		pix_RADEC = wcs.pixel_to_world(pix_xx.flatten(),pix_yy.flatten())
		pix_RA = np.asarray(pix_RADEC.ra.deg).reshape(Ny,Nx)
		pix_DEC = np.asarray(pix_RADEC.dec.deg).reshape(Ny,Nx)


	pix_xxyy = [pix_xx,pix_yy]
	pix_RADEC = [pix_RA,pix_DEC]

	return pix_xxyy, pix_RADEC


def make_pix_WCS_grids(header):
	wcs =  WCS(header).celestial
	Nx = header['NAXIS1']
	Ny = header['NAXIS2']


	#pixel coordinates .. Naxis2=rows,Naxis1=cols, FITS begins at 1
	pix_xx, pix_yy = np.meshgrid(np.arange(Nx,dtype=int) + 1,
							np.arange(Ny,dtype=int) + 1,
							indexing='xy') 

	pix_xxyy = [pix_xx.flatten(),pix_yy.flatten()]


	#pixel RA + DEC
	pix_radec = wcs.all_pix2world(np.column_stack([pix_xx.flatten(),
													pix_yy.flatten()]),1)
	pix_RA = pix_radec[:,0]
	pix_DEC = pix_radec[:,1]
	pix_RADEC = [pix_RA,pix_DEC]


	#get "corrected" pixel coordinates by reversing transform using core WCS lib only
	# after all possible distortions have been applied. Technically "corrected intermediate pixel corrdinates")
	#but in the frame of "corrected world coordinates", which I think translates to
	#the focal plane. see Calabretta+04. Projection to sky plane and world coords are then 'simple'
	pix_foc = wcs.wcs_world2pix(pix_radec,1)
	wcsprm = Wcsprm(str(header).encode()).sub(['celestial'])
	out = wcsprm.p2s(pix_foc,1)
	pix_SPxx = np.asarray(out['imgcrd'])[:,0]
	pix_SPyy = np.asarray(out['imgcrd'])[:,1]
	pix_SP = [pix_SPxx,pix_SPyy]
	#intermediate world coordinates ['imgcrd'].. ['world'] gives RA+Dec as ^ 



	# #do as simpler grid
	# PCi_j = wcs.wcs.pc
	# CDELT = wcs.wcs.cdelt
	# CRPIX = wcs.wcs.crpix
	# pix_SPxx = PCi_j[0,0]*(pix_xx.flatten() - CRPIX[0]) + PCi_j[0,1]*(pix_yy.flatten() - CRPIX[1])
	# pix_SPyy = PCi_j[1,0]*(pix_xx.flatten() - CRPIX[0]) + PCi_j[1,1]*(pix_yy.flatten() - CRPIX[1])
	# pix_SPxx *= CDELT[0]
	# pix_SPyy *= CDELT[1]
	# pix_SP = [pix_SPxx,pix_SPyy]



	# # just an RA & DEC test
	# test1 = np.asarray(out['world'])[:,0]
	# test2 = np.asarray(out['world'])[:,1]
	# print(pix_RA - test1)
	# print(np.nanmean(pix_RA - test1),np.std(pix_RA - test1))
	# exit()


	# # just an SPxxyy test
	# test1 = np.asarray(out['imgcrd'])[:,0]
	# test2 = np.asarray(out['imgcrd'])[:,1]
	# print(pix_SPxxyy1[:,0] - pix_SPxx)
	# print(np.nanmean(pix_RA - test1),np.std(pix_RA - test1))
	# exit()


	return  pix_xxyy, pix_RADEC, pix_SP


def get_wavelength_axis(header,axtype = None):
    wcs = WCS(header)
    pix_wav = np.asarray(wcs.spectral.pixel_to_world(np.arange(header['NAXIS3'])))

    if axtype == "FREQ" or "FREQ" in wcs.axis_type_names:
        pix_wav = ac.c.value / pix_wav

    return pix_wav


def get_frequency_axis(header,axtype = None):
    wcs = WCS(header)
    pix_freq = np.asarray(wcs.spectral.pixel_to_world(np.arange(header['NAXIS3'])))

    if axtype == "WAV" or "WAV" in wcs.axis_type_names:
        pix_freq = ac.c.value / pix_freq

    return pix_freq

def get_wavelength_axis_old(header, units=1.e10):
	
	wcs =  WCS(header)
	pix_WAV = units*np.asarray(wcs.sub([3]).pixel_to_world(np.arange(header['NAXIS3'])))

	return pix_WAV

def convert_skyplane_to_RADEC(pix_SP,header):
	#astropy doesnt support sky-plane -> RA/DEC, so this has to be a little manual
	wcs = WCS(header).celestial

	CDELT = wcs.wcs.cdelt
	CRPIX = wcs.wcs.crpix

	pix_foc_xx = pix_SP[:,0]/CDELT[0] + CRPIX[0]
	pix_foc_yy = pix_SP[:,1]/CDELT[1] + CRPIX[1]

	pix_RADEC = wcs.wcs_pix2world(np.column_stack([pix_foc_xx,pix_foc_yy]),1)

	return pix_RADEC

def convert_RADEC_to_skyplane(pix_RADEC,header):
	#astropy doesnt support RA/DEC -> sky-plane, so this has to be a little manual
	#wcsprm only deals with core WCS transforms
	wcsprm = Wcsprm(str(header).encode()).sub(['celestial'])

	pix_SP = wcsprm.s2p(pix_RADEC,1)['imgcrd']


	return pix_SP

def deproject_pixel_coordinates(x0, y0, pix_xx, pix_yy, PA=0, incl=0, reverse = False):
	#PA must be measured East of North!
	#aligns PA to x-axis
	#should all be in sky-plane pixel coordinates!
	PA =  (90 - PA) * np.pi/180.
	incl = incl*np.pi/180.
	
	if not reverse:
		#centre coordinates
		pix_xx_sky = pix_xx - x0
		pix_yy_sky = pix_yy - y0


		pix_xx_sky = -pix_xx_sky			# this corrects the coordinate system to right-handed

		#rotate to galaxy PA
		pix_xx_gal = pix_xx_sky*np.cos(PA) - pix_yy_sky * np.sin(PA)
		pix_yy_gal = (1.e0*pix_xx_sky*np.sin(PA) + pix_yy_sky* np.cos(PA)) 

		#incline y-axis
		pix_yy_gal *= 1./np.cos(incl)

		pix_rr_gal = 3600*np.sqrt(pix_xx_gal**2.e0 + pix_yy_gal**2.e0)
		
		return pix_xx_gal,pix_yy_gal, pix_rr_gal


	elif reverse:
		pix_yy *= np.cos(incl)
		pix_xx_sky = pix_xx*np.cos(PA) + pix_yy * np.sin(PA)
		pix_yy_sky = (-1.e0*pix_xx*np.sin(PA) + pix_yy* np.cos(PA)) 

		pix_xx_sky = -pix_xx_sky

		pix_xx_sky += x0
		pix_yy_sky += y0

		return pix_xx_sky, pix_yy_sky



def rot_mat(zrot,yrot,xrot):
	#should be right-multiplied with row vectors np.matmul([x,y,z],rot_mat)

	r1 = [np.cos(zrot)*np.cos(yrot),np.cos(zrot)*np.sin(yrot)*np.sin(xrot) - np.sin(zrot)*np.cos(xrot), np.cos(zrot)*np.sin(yrot)*np.cos(xrot)+np.sin(zrot)*np.sin(xrot)]
	r2 = [np.sin(zrot)*np.cos(yrot), np.sin(zrot)*np.sin(yrot)*np.sin(xrot)+np.cos(zrot)*np.cos(xrot), np.sin(zrot)*np.sin(yrot)*np.cos(xrot)-np.cos(zrot)*np.sin(xrot)]
	r3 = [-np.sin(yrot), np.cos(yrot)*np.sin(xrot), np.cos(yrot)*np.cos(xrot)]
	rr = np.array([r1,r2,r3])
	return rr

def make_WCSregion_polygon(region_file):

	# region_file = '/home/awatts/projects/MUSEdemo/regs.reg'

	f = open(region_file,'r')
	line1 = f.readline()

	if 'DS9' in line1:
		region_type = 'DS9'

	elif 'CRTF' in  line1:
		region_type = 'CARTA'
	
	polygons = []
	for line in f:
		if region_type == 'CARTA':
			linesplit = line.split(' ')
			shape = linesplit[0]
			RA = float(linesplit[1].split('[')[-1].split('deg,')[0])
			DEC = float(linesplit[2].split('deg],')[0])
			Rmaj = float(linesplit[3].split('[')[-1].split('arcsec,')[0])
			Rmin = float(linesplit[4].split('arcsec],')[0])
			PA = float(linesplit[5].split('deg]')[0])

			Rmaj_deg = Rmaj / 3600.
			Rmin_deg = Rmin / 3600.
			if shape == 'ellipse':
				phi = np.linspace(0,2*np.pi,360)
				xx = Rmaj_deg*np.cos(phi)
				yy =  Rmin_deg*np.sin(phi)
			
			elif shape == 'rotbox':
				xx = np.array([0.5*Rmaj_deg,0.5*Rmaj_deg,-0.5*Rmaj_deg,
								-0.5*Rmaj_deg])
				yy = np.array([-0.5*Rmin_deg,0.5*Rmin_deg,0.5*Rmin_deg,
								-0.5*Rmin_deg])

		xx_wcs = RA + xx*np.cos(-PA*np.pi/180.) - yy*np.sin(-PA*np.pi/180.)
		yy_wcs = DEC + xx*np.sin(-PA*np.pi/180.) + yy*np.cos(-PA*np.pi/180.)

		poly = [xx_wcs,yy_wcs]
		polygons.append(poly)
			
	f.close()


	return polygons


def get_WCSregion_spectrum(datacube, region_file):
	# 
	# datacube = '/home/awatts/projects/MUSEdemo/ADP.2016-07-25T13_16_34.364.fits'
	# region_file = '/home/awatts/projects/MUSEdemo/regs.reg'


	hdul = fits.open(datacube)
	head = hdul[1].header
	data = hdul[1].data
	wcs = WCS(head)

	hdul.close()

	pix_AA = get_wavelength_axis(head)


	if region_file == 'all':
		spectra = np.zeros([len(pix_AA),2])

		spectra[:,0] = pix_AA
		spectra[:,1] = np.nansum(data, axis=(1,2))
			

	else:
		region_polygons = make_WCSregion_polygon(region_file)
		
		pix_xxyy, pix_RADEC = make_pix_WCS_grids(head)


		# RA_pix = (np.arange(head['NAXIS1']) - head['CRPIX1'])*head['CD1_1'] + head['CRVAL1']
		# DEC_pix = (np.arange(head['NAXIS2']) - head['CRPIX2'])*head['CD2_2'] + head['CRVAL2']

		# RA_mesh,DEC_mesh = np.meshgrid(RA_pix,DEC_pix)
		# xxpix_mesh,yypix_mesh = np.meshgrid(np.arange(head['NAXIS1']),np.arange(head['NAXIS2']))


		pix_RADEC_flat = np.array([pix_RADEC[0].flatten(),pix_RADEC[1].flatten()]).T
		pix_xx_flat = pix_xxyy[0].flatten()
		pix_yy_flat = pix_xxyy[1].flatten()

		spectra = np.zeros([len(pix_AA),len(region_polygons)+1])
		spectra[:,0] = pix_AA
		for ii in range(len(region_polygons)):
			polygon = region_polygons[ii]
			in_poly = inside_polygon(pix_RADEC_flat,polygon[0],polygon[1])

			region_spectrum = np.nansum(data[:,pix_yy_flat[in_poly],pix_xx_flat[in_poly]],axis=1)

			spectra[:,ii+1] = region_spectrum

	header=None
	data=None
	
	return spectra

def extract_subcube(datacube,subcube_index,filepath = True,hdu=0,mask=False):

	if filepath:
		cube = SpectralCube.read(datacube,hdu=hdu)
	elif not filepath:
		cube = SpectralCube(data=datacube[0],wcs=datacube[1])


	if not isinstance(mask,bool):

		cube = cube.with_mask(mask)


	if subcube_index[0]=='all':
		subcube = cube[:,
					subcube_index[1][0]:subcube_index[1][1],
					subcube_index[2][0]:subcube_index[2][1]]
	else:
		subcube = cube[subcube_index[0][0]:subcube_index[0][1],
					subcube_index[1][0]:subcube_index[1][1],
					subcube_index[2][0]:subcube_index[2][1]]

	return subcube




def calculate_HI_mass(Sint,dL):

	MHI = 2.356e5 * dL * dL * Sint
	return MHI

def Wang16_HIsizemass(logMHI = None, logDHI = None):

	if np.ndim(logMHI)>0 or np.isscalar(logMHI != None):
		logDHI = 0.506 * logMHI - 3.293
		return logDHI

	elif np.ndim(logDHI)>0 or np.isscalar(logDHI != None):
		logMHI = (logDHI + 3.293) / 0.506
		return logMHI
	else:
		print('Incorrect input')

def xGASS_SFMS(lgMstar, sigma=0):

	sSFR = -0.344*(lgMstar - 9) - 9.822
	
	sSFR += sigma * (0.088 * (lgMstar-9) + 0.188)

	return sSFR



def int_Ez(z, omega_M = None, omega_L = None):
	if omega_M == None:
		omega_M = 1. - omega_L

	if omega_L == None:
		omega_L = 1. - omega_M

	dz = 1.e-6
	zrange = np.arange(0,z,dz)
	Ez = np.sqrt(omega_M*(1. + zrange)**3.e0 + omega_L)
	int_Ez = np.nansum(1.e0/Ez)*dz
	return int_Ez


def calculate_luminosity_distance(z,H0 = 70.e0, omega_M = 0.3, omega_L = 0.7):
	dH = 2.99792e5 / H0
	dC = dH * int_Ez(z,omega_M,omega_L)
	dM = dC
	dL = (1.e0 + z) * dM
	return dL


def ks_test(samp1,samp2,alternative='two sided'):

	result = ks_2samp(samp1,samp2,alternative=alternative)

	return result

def ad_test(samp1,samp2):
	result =  anderson_ksamp([samp1,samp2])

	return result

def weighted_ks_test(samp1,samp2,weights1 = None,weights2 = None,n1 = None,n2 = None,alternative='two-sided',method='asymp'):
	#weighted KS-test from https://stackoverflow.com/a/67638913
	#samp1,2 are data points
	#weights1, 2 are the weights, set all to 1 if not provided
	#n1,n2 number of data points in each sample, used to adjust p-value calculation if on sample containes
		# one data point contributing multiple times (e.g. one N = 250 galaxy used to create mocks for 4 others
		# should not contribute 1000 points to the significance)
	if isinstance(weights1,type(None)):
		weights1 = np.ones(len(samp1))

	if isinstance(weights2,type(None)):
		weights2 = np.ones(len(samp2))

	samp1_argsort = np.argsort(samp1)
	samp2_argsort = np.argsort(samp2)
	samp1 = samp1[samp1_argsort]
	samp2 = samp2[samp2_argsort]
	weights1 = weights1[samp1_argsort]
	weights2 = weights2[samp2_argsort]

	samp_comb = np.hstack([samp1, samp2])

	cumu_weights1 = np.hstack([0, np.cumsum(weights1)/sum(weights1)])
	cumu_weights2 = np.hstack([0, np.cumsum(weights2)/sum(weights2)])

	cdf1we = cumu_weights1[np.searchsorted(samp1, samp_comb, side='right')]
	cdf2we = cumu_weights2[np.searchsorted(samp2, samp_comb, side='right')]

	# plt.figure()
	# plt.plot(cdf1we)
	# plt.plot(cdf2we)
	# plt.show()

	if method == 'asymp':
		d = np.max(np.abs(cdf1we - cdf2we))
		# calculate p-value
		if isinstance(n1,type(None)):
			n1 = samp1.shape[0]
		if isinstance(n2,type(None)):
			n2 = samp2.shape[0]

		m, n = sorted([float(n1), float(n2)], reverse=True)
		en = m * n / (m + n)
		if alternative == 'two-sided':
			prob = distributions.kstwo.sf(d, np.round(en))
		else:
			z = np.sqrt(en) * d
			# Use Hodges' suggested approximation Eqn 5.3
			# Requires m to be the larger of (n1, n2)
			expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
			prob = np.exp(expt)
		return [d, prob]


def weighted_moment(data,weights=None,moment = 1):

	if isinstance(weights,type(None)):
		weights = np.ones_like(data)

	if moment >= 0:

		moment0 = np.sum(data*weights)
		mom = moment0

	if moment >= 1:
		moment1 =  moment0 / np.sum(weights)
		mom = moment1

	if moment >= 2:
		moment2 = np.sum(weights*(data - moment1)**2) / np.sum(weights)
		moment2 = np.sqrt(moment2)
		mom = moment2

	if moment == 3:
		mom = np.sum(weights * ((data - moment1) / moment2)**3) / np.sum(weights)

	if moment == 4:
		mom = np.sum(weights * ((data - moment1) / moment2)**4) / np.sum(weights) - 3

	
	return mom	

def weighted_moment_and_uncertainty(data,weights=None,moment=[1],noise=None,nIter=10000):

	noiseArr = rng.normal(loc=0,scale=noise,size=(nIter,len(noise))).T


	nMoments = len(moment)

	moms = []

	for mm in range(nMoments):
		mom = weighted_moment(data,weights,moment=moment[mm])
		momDist = []
		for nn in range(nIter):
			momDist.extend([weighted_moment(data,weights+noiseArr[:,nn],moment=moment[mm])])

		sigmaMom = median_absolute_deviation(np.array(momDist),Niter=5)[1]
		# plt.figure()
		# plt.hist(np.array(momDist)[np.abs(mom - np.array(momDist))<3*sigmaMom],bins=100)
		# plt.show()
		# print(moment[mm],mom,np.nanmedian(momDist),sigmaMom,np.sqrt(np.sum((noise*data)**2)),np.nanmedian(momDist)/sigmaMom, mom/np.sqrt(np.sum((noise*data)**2)))
		moms.extend([mom,sigmaMom])

	return moms

def weighted_percentile(data,weights,percentile = 50):
	weights = weights[np.isfinite(data)]
	data = data[np.isfinite(data)]
	
	data = data[np.isfinite(weights)]
	weights = weights[np.isfinite(weights)]

	if len(data)==0 or len(weights)==0:
		del weights
		return None
	else:

		weights_tot = np.nansum(weights)
		data_argsort = np.argsort(data)
		sort_weights = weights[data_argsort]
		sort_data = data[data_argsort]
		tot=0
		jj = -1
		sort_weights = sort_weights / weights_tot

		cumsum_weights = np.cumsum(sort_weights) - 0.5*sort_weights #places each point at it's centre

		value = np.interp(percentile/100, cumsum_weights,sort_data)
		del weights
		return value

	#OLD CODE FOR WEIGHTED MEDIAN
	# while(tot<0.5):
	# 	jj +=1
	# 	tot += sort_weights[jj]
	
	# kk = len(data)
	# tot=0
	# while(tot<0.5):
	# 	kk-=1
	# 	tot+= sort_weights[kk]
	# if data_argsort[kk] == data_argsort[jj]:
	# 	weighted_median = sort_data[kk]
	# else:
	# 	weighted_median = 0.5*(sort_data[kk] + sort_data[jj])

	# return weighted_median


def standard_error_on_median(sample,Nsamp=10000):
	medians = np.zeros(Nsamp)
	for ii in range(Nsamp):
		samp = rng.choice(sample, len(sample), replace=True)
		medians[ii] = np.median(samp)

	# plt.hist(medians,bins=20)
	# plt.show()

	SE = np.std(medians)
	return SE 


def median_absolute_deviation(array,Niter=1):
    # med = np.nanmedian(array)

    MAD = np.nanmedian( np.abs(array - np.nanmedian(array)) )
    MAD = 1.4826*MAD


    for ii in range(Niter):
        array = array[(np.abs(array - np.nanmedian(array)) <=2.5*MAD)]
        med = np.nanmedian(array)

        MAD = np.nanmedian( np.abs(array - med) )
        MAD = 1.4826*MAD
        print(MAD)



    return med, MAD


def equal_contribution_histogram_v1(data,bins):
	#stacks and renormalises histograms to weight each input equally
	#data = nested list of datasets to compute the histograms on
	#bins = bins


	hists = np.zeros([len(data),len(bins)-1])
	
	for jj in range(len(data)):
		if len(data[jj])>0:
			hist,bins = np.histogram(data[jj],bins=bins,density=True)
			hists[jj,:] = hist

	hists_total = np.sum(hists,axis=0)
	hists_equal = hists_total / (np.sum(hists_total*np.diff(bins)))


	return hists_equal

def equal_contribution_histogram_v2(data,bins,stats=None):
	#stacks and renormalises histograms to weight each input equally, but better
	#data = nested list of datasets to compute the histograms on
	#bins = bins

	data_all_JK = [] 
	weights_all_JK = []

	Ndata = len(data)
	weights = [ np.ones_like(dat) / (Ndata*len(dat)) for dat in data] #each dataset is weighted by 1 / Nall*Nset
																		#that way, total weights=1	
	data_all = np.hstack(data)
	weights_all = np.hstack(weights)

	for ii in range(len(data)):
		data_copy = data.copy()
		if len(data) > 1:
			data_copy = data_copy[0:ii] + data_copy[ii+1::]
		Ndata_copy = len(data_copy)
		weights_copy = [ np.ones_like(dat) / (Ndata_copy*len(data_copy)) for dat in data_copy]

		data_all_copy = np.hstack(data_copy)
		weights_all_copy = np.hstack(weights_copy)
		data_all_JK.append(data_all_copy)
		weights_all_JK.append(weights_all_copy)

	hist,bins = np.histogram(data_all,bins=bins,weights=weights_all,density=True)

	statistics = []

	if isinstance(stats,list):
		for ss in stats:
			if ss == 'median':
				stat = weighted_percentile(data_all,weights_all,percentile=50)
				stat_JKs = []
				for ii in range(len(data_all_JK)):
					stat_JK = weighted_percentile(data_all_JK[ii],weights_all_JK[ii],percentile=50)
					stat_JKs.extend([stat_JK])
			if ss == 'mean':
				stat = weighted_moment(data_all,weights_all,moment=1)
				stat_JKs = []
				for ii in range(len(data_all_JK)):
					stat_JK = weighted_moment(data_all_JK[ii],weights_all_JK[ii],moment=1)
					stat_JKs.extend([stat_JK])
			if ss == 'stddev':
				stat = weighted_moment(data_all,weights_all,moment=2)
				stat_JKs = []
				for ii in range(len(data_all_JK)):
					stat_JK = weighted_moment(data_all_JK[ii],weights_all_JK[ii],moment=2)
					stat_JKs.extend([stat_JK])
			if ss == 'skewness':
				stat = weighted_moment(data_all,weights_all,moment=3)
				stat_JKs = []
				for ii in range(len(data_all_JK)):
					stat_JK = weighted_moment(data_all_JK[ii],weights_all_JK[ii],moment=3)
					stat_JKs.extend([stat_JK])
			if ss == 'kurtosis':
				stat = weighted_moment(data_all,weights_all,moment=4)
				stat_JKs = []
				for ii in range(len(data_all_JK)):
					stat_JK = weighted_moment(data_all_JK[ii],weights_all_JK[ii],moment=4)
					stat_JKs.extend([stat_JK])
			if "P" in ss:
				percentile = int(ss.split("P")[-1])
				stat = weighted_percentile(data_all,weights_all,percentile=percentile)
				stat_JKs = []
				for ii in range(len(data_all_JK)):
					stat_JK = weighted_percentile(data_all_JK[ii],weights_all_JK[ii],percentile=percentile)
					stat_JKs.extend([stat_JK])

			stat_JKs = np.array(stat_JKs)
			stat_err = ((Ndata - 1) / Ndata) * np.sum( (stat_JKs - np.mean(stat_JKs))**2 )
			stat_err = np.sqrt(stat_err)

			statistics.append([stat,stat_err])

		return hist, statistics


	elif not isinstance(stats,type(None)):
		print("Stats needs to be a list")
		exit()


	return hist

def equal_contribution_histogram(data,bins, weights = None,stats=None,method = 'bootstrap'):
	#stacks and renormalises histograms to weight each input equally, but better
	#data = nested list of datasets to compute the histograms on
	#bins = bins

	# data_all_JK = [] 
	# weights_all_JK = []

	Ndata = len(data)
	if isinstance(weights,type(None)):
		weights = []
		for ii in range(Ndata):
			weights.extend([np.ones_like(data[ii]) / (Ndata*len(data[ii]))]) #each dataset is weighted by 1 / Nall*Nset
																		#that way, total weights=1											
	data_all = np.hstack(data)
	weights_all = np.hstack(weights)

	# print(len(data_all))
	# print(len(weights_all))
	# exit()

	# for ii in range(len(data)):
	# 	data_copy = data.copy()
	# 	if len(data) > 1:
	# 		data_copy = data_copy[0:ii] + data_copy[ii+1::]
	# 	Ndata_copy = len(data_copy)
	# 	weights_copy = [ np.ones_like(dat) / (Ndata_copy*len(data_copy)) for dat in data_copy]

	# 	data_all_copy = np.hstack(data_copy)
	# 	weights_all_copy = np.hstack(weights_copy)
	# 	data_all_JK.append(data_all_copy)
	# 	weights_all_JK.append(weights_all_copy)
	# print(len(data),len(weights),data_all.shape,weights_all.shape)

	hist, bins = np.histogram(data_all,bins=bins,weights=weights_all,density=True)

	statistics = []

	if isinstance(stats,list):
		for ss in stats:
			if ss == 'median':
				func = lambda dd, ww : weighted_percentile(dd,ww,percentile=50)
			if ss == 'mean':
				func = lambda dd, ww : weighted_moment(dd,ww,moment=1)
			if ss == 'stddev':
				func = lambda dd, ww : weighted_moment(dd,ww,moment=2)
			if ss == 'skewness':
				func = lambda dd, ww : weighted_moment(dd,ww,moment=3)
			if ss == 'kurtosis':
				func = lambda dd, ww : weighted_moment(dd,ww,moment=4)
			if "P" in ss:
				percentile = int(ss.split("P")[-1])
				func = lambda dd, ww : weighted_percentile(dd,ww,percentile=percentile)



			stat = func(data_all,weights_all)

			if method == 'jackknife':
				stat_thetas = []
				for ii in range(len(data)):
					data_copy = copy.deepcopy(data)
					if len(data) > 1:
						data_resamp = data_copy[0:ii] + data_copy[ii+1::]
					else:
						data_resamp = data_copy
					# print(len(data))
					# print(len(data_resamp))

					Ndata_resamp = len(data_resamp)
					weights_resamp = [ np.ones_like(dat) / (Ndata_resamp*len(data_resamp)) for dat in data_resamp]
					data_all_resamp = np.hstack(data_resamp)
					weights_all_resamp = np.hstack(weights_resamp)

					stat_thetas.extend([func(data_all_resamp,weights_all_resamp)])


				stat_thetas = np.array(stat_thetas)
				stat_var = ((Ndata - 1) / Ndata) * np.sum( (stat_thetas - np.mean(stat_thetas))**2 )
				stat_err = np.sqrt(stat_var)



			elif method == 'bootstrap':
				stat_thetas = []
				Nsamp = 10
				for ii in range(Nsamp):
					resamp_index = rng.choice(len(data),len(data),replace=True)

					# data_resamp = rng.choice(np.asarray(data,dtype=object), len(data), replace=True)
					data_resamp = [data[dd] for dd in resamp_index]

					# print(len(data))
					# print(len(data_resamp))
					# print(resamp_index)
					# exit()
					 
					Ndata_resamp = len(data_resamp)
					weights_resamp = [ np.ones_like(dat) / (Ndata_resamp*len(data_resamp)) for dat in data_resamp]

					data_all_resamp = np.hstack(data_resamp)
					weights_all_resamp = np.hstack(weights_resamp)

					stat_thetas.extend([func(data_all_resamp,weights_all_resamp)])



				stat_thetas = np.array(stat_thetas)
				# print(np.mean(stat_thetas))
				# plt.hist(stat_thetas - np.mean(stat_thetas),bins=100)
				# plt.show()
				# exit()
				stat_var = (1. / Nsamp) * np.sum( (stat_thetas - np.mean(stat_thetas))**2 )
				stat_err = np.sqrt(stat_var)
			elif method == None:
				stat_err = -1


			statistics.append([stat,stat_err])

		return hist, statistics


	elif not isinstance(stats,type(None)):
		print("Stats needs to be a list")
		exit()

	del weights
	return hist


def norm_gaussian(xx,mu,sigma):
	
	prob = 1. / (sigma*np.sqrt(2.e0 * np.pi)) * \
			np.exp(-0.5e0*( ((xx - mu) / sigma) *((xx - mu) / sigma) ))
	return prob

def fit_2d_gaussian(xx,yy,values,p0=None):
	from scipy.optimize import curve_fit

	coords = np.vstack((xx,yy)).T

	coords = coords[np.isfinite(values),:]
	values = values[np.isfinite(values)]


	if isinstance(p0,type(None)):
		p0 = [np.nanmax(values),np.nanmedian(coords[:,0]),np.nanmedian(coords[:,1]),0,1,0.5]

	fit, covar = curve_fit(Gaussian_2d,coords,values,
				p0=p0)

	return fit


def Gaussian_2d(data, A, x0, y0, theta, sigma_x, sigma_y):
	
	xx = data[:,0]
	yy = data[:,1]

	sigma_x *= sigma_x
	sigma_y *= sigma_y
	x = xx - x0
	y = yy - y0


	a = (np.cos(theta)**2.e0)/(2.e0 * sigma_x) + (np.sin(theta)**2.e0)/(2.e0 * sigma_y)
	b = np.sin(2.e0 * theta) / (4.e0 * sigma_y) - np.sin(2.e0 * theta) / (4.e0*sigma_x)
	c = (np.sin(theta)**2.e0)/(2.e0 * sigma_x) + (np.cos(theta)**2.e0)/(2.e0 * sigma_y)

	G = A* np.exp(-1.e0*( a*x*x + 2.e0*b*x*y + c*y*y ))
	# G = A*np.exp(-1.e0*( (xx-x0)*(xx-x0)/(2.e0*sigma_x*sigma_x) + (yy-y0)*(yy-y0)/(2.e0*sigma_y*sigma_y) ) )

	return G




def EBV_Hlines(F1 ,F2 ,lambda1 = 6562.819 ,lambda2 = 4861.333, Rint = 2.83,k_l = None):
    #lambdas in angstrom
    #F1=HA F2 = HB (default)
    
	if isinstance(k_l,type(None)):
	   k_l = lambda ll: extinction_curve(ll)

	ratio = np.log10((F1/F2) / Rint)

	kdiff = k_l(lambda2) - k_l(lambda1)

	E_BV = ratio / (0.4 * kdiff)
	# print(np.min(E_BV))

	E_BV[np.isfinite(E_BV)==False] = 0
	# print(np.min(E_BV))

	return E_BV


@np.vectorize
def extinction_curve(ll, RV = 3.1, extcurve = 'Cardelli89'):
    ##ll should be in Angstrom
    
    #stellar extinction curve
    if extcurve == 'Calzetti00':
        ll *= 1.e-4         #convert to micron
        llinv = 1.e0/ll

        if  ll >= 0.12 and ll < 0.63:
            k = 2.659*(-2.156 + 1.509*llinv - 0.196*(llinv*llinv) + 0.011*(llinv*llinv*llinv)) + RV

        elif ll >=0.63 and ll <=2.20:
            k = 2.659*(-1.857 + 1.040*llinv) + RV
        else:
            k = np.nan

    #MW attenuation curve
    if extcurve == 'Cardelli89':
        ll *= 1.e-4
        llinv = 1.e0/ll

        if llinv>=1.1 and llinv<=0.3:
            aa = 0.574*llinv**1.61 
            bb = -0.527*llinv**1.61

        elif llinv <= 3.3 and llinv>=1.1:
            yy = llinv - 1.82
            aa = 1 + 0.17699*yy - 0.50447*yy**2 - 0.02427*yy**3 + 0.72085*yy**4 + 0.01979*yy**5 - 0.77530*yy**6 + 0.32999*yy**7
            bb = 1.41338*yy + 2.28305*yy**2 + 1.07233*yy**3 - 5.38434*yy**4 - 0.62251*yy**5 + 5.30260*yy**6 - 2.09002*yy**7

        elif llinv >= 3.3 and llinv <=8:
            if llinv >=5.9 and llinv <=8:
                Faa = -0.04473*(llinv - 5.9)**2 - 0.009779*(llinv - 5.9)**3
                Fbb = 0.2130*(llinv - 5.9)**2 - 0.1207*(llinv - 5.9)**3
            elif llinv < 5.9:
                Faa  = 0
                Fbb = 0

            aa = 1.752 - 0.316*llinv - (0.104/( (llinv - 4.67)**2 + 0.341 )) + Faa
            bb = -3.090 + 1.825*llinv + (1.206/( (llinv - 4.62)**2 + 0.263 )) + Fbb
        else:
            aa = np.nan
            bb = np.nan


        Al_AV = aa + bb/RV
        k = (Al_AV) * RV

    return k

# def kewley06_BPT_classification


def lgMstarMsun_Zibetti09(colour_data,absMag_data, band1='g', band2='i'):

	# Colour ag bg ar br ai bi az bz aJ bJ aH bH aK bK
	# u − g −1.628 1.360 −1.319 1.093 −1.277 0.980 −1.315 0.913 −1.350 0.804 −1.467 0.750 −1.578 0.739
	# u − r −1.427 0.835 −1.157 0.672 −1.130 0.602 −1.181 0.561 −1.235 0.495 −1.361 0.463 −1.471 0.455
	# u − i −1.468 0.716 −1.193 0.577 −1.160 0.517 −1.206 0.481 −1.256 0.422 −1.374 0.393 −1.477 0.384
	# u − z −1.559 0.658 −1.268 0.531 −1.225 0.474 −1.260 0.439 −1.297 0.383 −1.407 0.355 −1.501 0.344
	# g − r −1.030 2.053 −0.840 1.654 −0.845 1.481 −0.914 1.382 −1.007 1.225 −1.147 1.144 −1.257 1.119
	# g − i −1.197 1.431 −0.977 1.157 −0.963 1.032 −1.019 0.955 −1.098 0.844 −1.222 0.780 −1.321 0.754
	# g − z −1.370 1.190 −1.122 0.965 −1.089 0.858 −1.129 0.791 −1.183 0.689 −1.291 0.632 −1.379 0.604
	# r − i −1.405 4.280 −1.155 3.482 −1.114 3.087 −1.145 2.828 −1.199 2.467 −1.296 2.234 −1.371 2.109
	# r − z −1.576 2.490 −1.298 2.032 −1.238 1.797 −1.250 1.635 −1.271 1.398 −1.347 1.247 −1.405 1.157

	colour=f"{band1}{band2}"

	if colour == 'gi':
		if band2 == 'g':
			a = -1.197
			b = 1.431
		elif band2 == 'i':
			a = -0.963
			b =  1.032

			absMag_sun = 4.58
	elif colour == 'gr':
		if band2 == 'r':
			a = -0.840
			b = 1.654
			absMag_sun = 4.65



	logMLr = a + b*colour_data

	logMstarMsun = logMLr + 0.4*(absMag_sun - absMag_data)

	return logMstarMsun

def lgMstarMsun_Taylor11(colour_data, absMag_data):

	# absMag_sun = 4.58


	# logMLr = -0.68 + 0.7*colour_data


	# logMstarMsun = logMLr + 0.4*(absMag_sun - absMag_data)

	logMstarMsun = 1.15+0.7*(colour_data) - 0.4*absMag_data

	return logMstarMsun

def lgMstarMsun_TaylorSAMI(gi_colour, obs_imag, D_comov, z):
	#D_comov should be in Mpc
	Dmod = 5*(np.log10(D_comov) - 1 + 6)

	logMstarMsun = -0.4*obs_imag + 0.4*Dmod - np.log10(1 + z) + \
					(1.2117 - 0.5893*z) + (0.7106 - 0.1467*z)*gi_colour


	return logMstarMsun




def logOH_Scal_PG16(HB, OIII, NII, SII):

    R3 = OIII / HB
    N2 = NII / HB
    S2 = SII / HB

    logOH12_upper = (8.424 + 0.030*np.log10(R3/S2) + 0.751*np.log10(N2) + \
                    (-0.349+0.182*np.log10(R3/S2) + 0.508*np.log10(N2) )*np.log10(S2))

    logOH12_lower = (8.072 + 0.789*np.log10(R3/S2) + 0.726*np.log10(N2) + \
                    (1.069+0.170*np.log10(R3/S2) + 0.022*np.log10(N2) )*np.log10(S2))


    logOH = np.full(len(HB),np.nan)
    logOH[np.log10(N2)>=-0.6] = logOH12_upper[np.log10(N2)>=-0.6]
    logOH[np.log10(N2)<-0.6] = logOH12_lower[np.log10(N2)<-0.6]


    return logOH

def logOH_N2S2Ha_D16(HA,NIIr,SII):

	y = np.log10(NIIr/SII) + 0.264 * np.log10(NIIr/HA)

	logOH = 8.77 + y

	return logOH

def logOH_C20(R_obs,calib = 'N2'):
	from scipy.interpolate import CubicSpline

	# ['R2', 0.435, -1.362, -5.655, -4.851, -0.478, 0.736, 0.11, 0.10]
	# ['R3', -0.277, -3.549, -3.593, -0.981, 0, 0, 0.09, 0.07]
	# ['O3O2', -0.691, -2.944, -1.308, 0, 0, 0, 0.15, 0.14]
	# ['R23', 0.527, -1.569, -1.652, -0.421, 0, 0, 0.06, 0.12]
	# ['N2', -0.489, 1.513, -2.554, -5.293, -2.867, 0, 0.16, 0.10]
	# ['O3N2', 0.281, -4.765, -2.268, 0, 0, 0, 0.21, 0.09]
	# ['S2', -0.442, -0.360, -6.271, -8.339, -3.559, 0, 0.11, 0.06]
	# ['RS32', -0.054, -2.546, -1.970, 0.082, 0.222, 0, 0.07, 0.08]
	# ['O3S2',  0.191, -4.292, -2.538, 0.053, 0.332, 0, 0.17, 0.11]

	logOH_array = np.arange(7.75,8.85,0.001)

	if calib == 'R3':
		logR = lambda x: -0.277 -3.549*x -3.593*x**2 -0.981*x**3
		logOH_array = np.arange(8.003,8.85,0.001)

	if calib == 'N2':
		logR = lambda x: -0.489 + 1.513*x - 2.554*x**2 - 5.293*x**3 - 2.867*x**4

	elif calib == 'O3N2':
		logR = lambda x: 0.281 -4.765*x - 2.268*x**2

	elif calib == 'RS32':
		logR = lambda x: -0.054 -2.546*x -1.970*x**2 + 0.082*x**3 + 0.222*x**4
		logOH_array = np.arange(8,8.85,0.001)

	elif calib == 'O3S2':
		logR =lambda x: 0.191 -4.292*x -2.538*x**2 + 0.053*x**3 + 0.332*x**4

	# logRobs_logRmod_diff = np.full([len(logOH_array),len(R_obs)],np.log10(R_obs)).T - logR(logOH_array - 8.6)		#computes for each input Robs

	logR_mod = logR(logOH_array - 8.69)

	if logR_mod[0] - logR_mod[-1] > 0:
		logR_mod = logR_mod[::-1]
		logOH_array = logOH_array[::-1]


	logOH_func = CubicSpline(logR_mod,logOH_array,extrapolate=False)

	logOH = logOH_func(np.log10(R_obs))

	return logOH



def reproject_exact_AdamProper(input_data,output_projection,hdu = 0):
	try:
		from reproject import reproject_exact, reproject_adaptive
	except:
		print('No reproject!!')
		exit()


	data = input_data[hdu].data

	input_header = input_data[hdu].header
	input_WCS = WCS(input_header)
	# print(input_header)

	output_shape = (input_WCS.array_shape[0], WCS(output_projection).array_shape[0],WCS(output_projection).array_shape[1])
	pixarea_in = input_WCS.celestial.proj_plane_pixel_area().value*(3600**2)
	pixarea_out =   WCS(output_projection).celestial.proj_plane_pixel_area().value *(3600**2)
	pixel_conversion = pixarea_out/pixarea_in

	if input_header['NAXIS'] == 3:

		output_data = np.zeros(output_shape)

		# diffs = []
		for ii in range(input_header['NAXIS3']):
			input_slice = data[ii,...]

			output_slice = reproject_exact((input_slice,input_WCS.celestial),output_projection,
									return_footprint=False) * pixel_conversion
			output_data[ii,...] = output_slice 

			# diffs.extend([np.nansum(input_slice)/np.nansum(output_slice)])

			if ii% int(data.shape[0]/10) == 0:
				print(int(round(100*ii/data.shape[0])),r'% done')


	# output_data *= pixel_conversion

	return output_data


def makeThresholdMask(spectrumSN, narrowSN=3, broadSN=1.5, growSN = None, 
                        segJoin = None, allSeg=False,plot=False):
    
    maskNarrow = np.zeros_like(spectrumSN)
    maskBroad = np.zeros_like(spectrumSN)

    for cc in range(1,len(spectrumSN)-1):
        if spectrumSN[cc] >= narrowSN:
            maskNarrow[cc] = 2
        if np.all(spectrumSN[cc-1:cc+1] >= broadSN):
            maskBroad[cc-1:cc+1] = 1

    maskComb = maskNarrow + maskBroad

    #identify spectrally continous mask segments
    maskSegments = []
    seg = []
    for mm, val in enumerate(maskComb):
        if val != 0:
            seg.extend([mm])
        elif val == 0 or mm == len(maskComb)-1:
            #end of segment, only keep if it has a narrowMask component
            if len(seg) != 0 and np.any(maskComb[seg] >= 2):    
                maskSegments.append(seg)
            seg = []

    #grow mask 
    if isinstance(growSN,float) or isinstance(growSN,int): 
        maskSegmentsGrow = []            
        #grow mask segments down closer to 0
        for ss, seg in enumerate(maskSegments):
            segLen = 0
            while len(seg) != segLen:
                segLen = len(seg)
                if np.min(seg)-1>=0 and spectrumSN[np.min(seg)-1] > growSN:
                    seg = [np.min(seg)-1] + seg
                if np.max(seg)+1<len(spectrumSN)-1 and spectrumSN[np.max(seg)+1] > growSN:
                    seg = seg + [np.max(seg)+1]

            maskSegmentsGrow.append(seg)
        maskSegments = maskSegmentsGrow


    #join overlapping mask segments, or mask segments separated by a tolerance (segJoin)
    if not isinstance(segJoin,type(None)) and len(maskSegments)>1: 
        nSeg = 0
        while len(maskSegments) != nSeg and len(maskSegments) > 1:
            nSeg = len(maskSegments)
            maskSegmentsJoin = []            


            for seg,segNext in zip(maskSegments[:-1],maskSegments[1:]):
                if np.min(segNext) - np.max(seg) <= segJoin:
                    segJoined = list(range(np.min(seg+segNext),np.max(seg+segNext)+1))
                    maskSegmentsJoin.append(segJoined)
                else:
                    maskSegmentsJoin.append(seg)
                    if segNext == maskSegments[-1]:
                        maskSegmentsJoin.append(segNext)
        
            maskSegments = maskSegmentsJoin


    #find the largest mask segment -  assuming this corresponds to the spectral line of interest
    maxLen = 0
    maskFinal = np.zeros_like(maskComb)
    for ss, seg in enumerate(maskSegments):
        if allSeg:                              #keep all segments with narrowMask component
            maskFinal[seg] = 1
        else:
            if len(seg) > maxLen:
                maskFinal = np.zeros_like(maskComb)
                maskFinal[seg] = 1
                maxLen = len(seg)

                # print(seg)
            # while len(seg) > maxlen:
            #     maxlen = len(seg)
            #     if np.min(seg)-1>=0 and spectrumSN[np.min(seg)-1] > growSN:
            #         seg = [np.min(seg)-1] + seg
            #     if np.max(seg)+1<len(spectrumSN)-1 and spectrumSN[np.max(seg)+1] > growSN:
            #         seg = seg + [np.max(seg)+1]
             

    #keep only positive values.. unsure if right yet. 
    maskFinal[spectrumSN<0] = 0

    if plot:
        plt.figure()
        plt.plot([broadSN]*len(spectrumSN),color='Red')
        plt.plot(0.5*maskBroad*np.max(spectrumSN),color='Red')

        plt.plot(spectrumSN,color='Grey')
        plt.plot([narrowSN]*len(spectrumSN),color='Orange')
        plt.plot(0.33*maskNarrow*np.max(spectrumSN),color='Orange',label='')

        # plt.plot(0.3*maskComb*np.max(spectrumSN),color='Blue')
        
        plt.plot(spectrumSN*maskFinal,color='Black')
        plt.plot(maskFinal*np.max(spectrumSN),ls='--',color='Magenta')
        plt.show()

    return maskFinal






def makeLineMasks(spectrum, linLambda, lineLambdas, 
        z = 0,
        clipWidth = 1500, maskWidth = 500, baselineDeg = 2,
        broadSN = 1.5, narrowSN = 3, growSN = 3,
        segJoin = 0,allSeg=False,
        plot=False):

    linLambda = linLambda / (1+z)
    logLambda = np.log(linLambda)

    nLines = len(lineLambdas)
    nLambda = spectrum.shape[0]


    clipWidth /= (ac.c.value *1.e-3)     #km/s
    maskWidth /= (ac.c.value *1.e-3)


    # spectrum = convolve(spectrum,Box1DKernel(3))

    outputs = []
    for lineLambda in lineLambdas:

        maskRange = np.where((logLambda > np.log(lineLambda*(1 - maskWidth)))  & 
                            (logLambda < np.log(lineLambda*(1 + maskWidth))))[0]

     
        clipRange = np.where((logLambda > np.log(lineLambda*(1 - clipWidth)))  & 
                            (logLambda < np.log(lineLambda*(1 + clipWidth))))[0] 
     
        clippedSpectrum = sigma_clip(spectrum[clipRange],2.5,maxiters=5)
        
        baselinePoly = npPoly.Polynomial.fit(linLambda[clipRange[~clippedSpectrum.mask]],
                                                                clippedSpectrum[~clippedSpectrum.mask], 
                                                                deg=baselineDeg)

        clippedSpectrum = sigma_clip(spectrum[clipRange] - baselinePoly(linLambda[clipRange]), 2.5, maxiters=5)
        specNoise = np.std(clippedSpectrum[~clippedSpectrum.mask])

        maskSpectrum = spectrum[maskRange] - baselinePoly(linLambda[maskRange])

        maskVel = (ac.c.value*1.e-3)*(logLambda[maskRange] - np.log(lineLambda))

        spectrumSN = maskSpectrum / specNoise


        maskFinal = makeThresholdMask(spectrumSN,narrowSN = narrowSN, broadSN=broadSN,growSN=growSN, segJoin=segJoin,allSeg=allSeg,plot=plot)

                     
        outputs.append([specNoise,maskFinal, maskVel, maskSpectrum])

        if plot:
            plt.figure()
            plt.plot(maskVel,spectrum[maskRange])

            plt.plot(maskVel,baselinePoly(linLambda[maskRange]))
            plt.plot(maskVel,maskSpectrum*maskFinal)
            plt.show()
                     

    return outputs
                 
        




def baselineFunc(xx,aa,bb,cc):
    yy = aa*xx**2 + bb*xx + cc
    return yy






if __name__ == '__main__':
	print('No main function bro')
	# make_WCSregion_polygon()
	# get_WCSregion_spectrum()
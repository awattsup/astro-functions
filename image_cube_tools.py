from astropy.io import fits
from astropy.wcs import WCS, Wcsprm
from astropy import units as u
from astropy.stats import sigma_clip
import astropy.constants as ac
import astropy.units as au
import scipy.stats as sps
from scipy.stats import distributions
from scipy.optimize import least_squares
import subprocess
import numpy.polynomial as npPoly
import matplotlib.path as pltPath

import numpy as np
import numpy.polynomial as npPoly


try:
	import requests
	from io import BytesIO
	from PIL import Image
except:
	print('cant get SDSS images')



emlines = { 'HI':{      'lambda':[21.106114/100.], 'nu':[1420.405751], 'ratio':[1]},
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


import stats_tools as ST


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


def get_SDSS_mosaic_image(outfile, RA, DEC, imSize=1/6,filters = 'ugriz', scriptOnly=True):
	#imSize needs to be in degrees
    if subprocess.run(['which','swarp']).returncode == 1:
        print("sWarp was not found in yourn $PATH, exiting")
        exit()
        
    outfile1 = os.path.expanduser(f'{oufilee}.sh')
    objName = oufilee.split('/')[-1]
    cwd = oufilee.split(objName)[0]

    url = f"https://dr12.sdss.org/mosaics/script?onlyprimary=False&pixelscale=0.396&ra={RA}&filters={filters}&dec={DEC}&size={imSize}"
    script = requests.get(url)
    if script.ok:
        with open(outfile1,'wb') as file:
            file.write(script.content)
    else:
        print('Mosaic fetch failed')
    os.system(f"chmod +x {outfile1}")


    if not scriptOnly:
        print('Running shell script to make mosaic')
        subprocess.run(['bash',outfile1],cwd=cwd)
        print('Renaming output and cleaning up')
        subprocess.run(["rm frame*"],cwd=cwd,shell=True)
        subprocess.run(["rm *weight*"],cwd=cwd,shell=True)
        subprocess.run(["rm *.sh"],cwd=cwd,shell=True)
        subprocess.run(["rm *.swarp"],cwd=cwd,shell=True)	
        subprocess.run([f"mv J*.fits {objName}.fits"],cwd=cwd,shell=True)





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

def convert_pixcoord_to_pixcoord(pix_xxyy,headerFrom,headerTo):

	pix_RADEC = WCS(headerFrom).celestial.all_pix2world(pix_xxyy,1)
	pix_xxyy_H2 = WCS(headerTo).celestial.all_world2pix(pix_RADEC,1)

	return pix_xxyy_H2


def deproject_pixel_coordinates(pix_xx, pix_yy, x0, y0, PA=0, incl=0, reverse = False):
	#PA must be measured East of North!
	#aligns PA to x-axis
	#should all be in sky-plane pixel coordinates!
	PA =  (PA-90) * np.pi/180.
	incl = incl*np.pi/180.
	
	if not reverse:
		# print(PA*180/np.pi)
		#centre coordinates
		pix_xx_sky = pix_xx - x0
		pix_yy_sky = pix_yy - y0


		# pix_xx_sky = -pix_xx_sky			# this corrects from right-handed sky plane to left-handed
		# pix_yy_sky = -pix_yy_sky

		#rotate to galaxy PA
		pix_xx_gal = pix_xx_sky*np.cos(PA) - pix_yy_sky * np.sin(PA)
		pix_yy_gal = (1.e0*pix_xx_sky*np.sin(PA) + pix_yy_sky* np.cos(PA)) 

		#incline y-axis
		pix_yy_gal *= 1./np.cos(incl)
		pix_xx_gal *= -1

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




def make_WCSregion_polygon(region_file):

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
	
	try:
		from spectral_cube import SpectralCube
	except:
		print('no spectral cube installed, exiting')
		exit()
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



def reproject_exact_cube(input_data,output_projection,hdu = 0):
	try:
		from reproject import reproject_exact
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
	else:
		output_data = reproject_exact((data,input_WCS.celestial),output_projection,
									return_footprint=False) * pixel_conversion


	return output_data





def spectrumNoiseEstimate(spec, xaxis = None, baselineDeg = None, clip1=2.5, clip2=3.5, returnBaseline =  False):

	if isinstance(xaxis,type(None)):
		xaxis = np.arange(len(spec))


	clippedSpectrum = sigma_clip(spec,clip1,maxiters=10)
	if isinstance(baselineDeg,int):    
		baselinePoly = npPoly.Polynomial.fit(xaxis[~clippedSpectrum.mask],
												clippedSpectrum[~clippedSpectrum.mask], 
												deg=baselineDeg)

		clippedSpectrum = sigma_clip(spec - baselinePoly(xaxis), clip2, maxiters=10)

	elif not isinstance(baselineDeg,type(None)):
		print('Baseline polynomal needs to be an integer or type(None)')
		exit()


	# clippedSpectrum = sigma_clip(spectrum[clipRange] - baselinePoly(linLambda[clipRange]), 3.5, maxiters=5)
	specNoise = np.std(clippedSpectrum[~clippedSpectrum.mask])

	if returnBaseline:
		return specNoise, baselinePoly
	else:
		return specNoise


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

def multiEmissionLinesFunction(linLambda,lineLambdas, parameters):

	V = parameters[0]
	sigma = parameters[1]
	lineAmps = parameters[2:]

	spectrum = np.zeros(len(linLambda))

	for lineLambda,lineAmp in zip(lineLambdas,lineAmps):
		xx_kms = 1.e-3*ac.c.value * (np.log(linLambda) - np.log(lineLambda))
		spectrum += lineAmp*ST.gaussianPDF(xx_kms,V,sigma)

	return spectrum




def detectEmissionLines_mask(spectrum, linLambda, lineLambdas, 
        						z = 0,
        						clipWidth = 1500, maskWidth = 500, baselineDeg = 2,
        						broadSN = 1.5, narrowSN = 3, growSN = 3,
        						segJoin = 0, allSeg=False,
        						plot=False):

    linLambda = linLambda / (1+z)
    logLambda = np.log(linLambda)

    nLines = len(lineLambdas)
    nLambda = spectrum.shape[0]


    clipWidth /= (ac.c.value*1.e-3)     #km/s
    maskWidth /= (ac.c.value*1.e-3)

    outputs = []
    for lineLambda in lineLambdas:

        maskRange = np.where((logLambda > np.log(lineLambda*(1 - maskWidth)))  & 
                            (logLambda < np.log(lineLambda*(1 + maskWidth))))[0]

        clipRange = np.where((logLambda > np.log(lineLambda*(1 - clipWidth)))  & 
                            (logLambda < np.log(lineLambda*(1 + clipWidth))))[0] 
     
        specNoise, baselinePoly = spectrumNoiseEstimate(spectrum[clipRange],
        												xaxis = linLambda[clipRange],
        												baselineDeg = baselineDeg,
        												returnBaseline = True)

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
                
def detectEmissionLines_fit(spectrum, linLambda, lineLambdas, 
        						z = 0,
        						clipWidth = 1500, maskWidth = 500, baselineDeg = 2,
        						plot=False):

    linLambda = linLambda / (1+z)
    logLambda = np.log(linLambda)

    nLines = len(lineLambdas)
    nLambda = spectrum.shape[0]


    clipWidth /= (ac.c.value*1.e-3)     #km/s
    maskWidth /= (ac.c.value*1.e-3)

    outputs = []
    specNoise_all = np.array([])
    linLambda_all = np.array([])
    maskSpectrum_all = np.array([])



    for lineLambda in lineLambdas:

        maskRange = np.where((logLambda > np.log(lineLambda*(1 - maskWidth)))  & 
                            (logLambda < np.log(lineLambda*(1 + maskWidth))))[0]

        clipRange = np.where((logLambda > np.log(lineLambda*(1 - clipWidth)))  & 
                            (logLambda < np.log(lineLambda*(1 + clipWidth))))[0] 
     
        specNoise, baselinePoly = spectrumNoiseEstimate(spectrum[clipRange],
                                                        xaxis = linLambda[clipRange],
                                                        baselineDeg = baselineDeg,
                                                        returnBaseline = True)

        maskSpectrum = spectrum[maskRange] - baselinePoly(linLambda[maskRange])

        specNoise_all = np.hstack([specNoise_all,specNoise])
        linLambda_all = np.hstack([linLambda_all,linLambda[maskRange]])
        maskSpectrum_all = np.hstack([maskSpectrum_all,maskSpectrum])

	# maskVel = (ac.c.value*1.e-3)*(logLambda[maskRange] - np.log(lineLambda))

    multiLineFunc = lambda ll,params: multiEmissionLinesFunction(ll,[lineLambdas], parameters = params)

    residualFunc = lambda params,xdata,ydata: multiLineFunc(xdata,params) - ydata

    p0 = [0,20]+[10]*len(lineLambdas)

    emlinesFit = least_squares(residualFunc,x0=p0,verbose=0,args = (linLambda_all, maskSpectrum_all))

    lineFluxes = emlines.x[2:]
    FHWMs = 

    outputs.append([specNoise,emlinesFit.x,linLambda_all,maskSpectrum_all,multiLineFunc(linLambda_all,emlinesFit.x)])

    # outputs.append([specNoise,maskFinal, maskVel, maskSpectrum])

    # if plot:
    #     plt.figure()
    #     plt.plot(maskVel,spectrum[maskRange])

    #     plt.plot(maskVel,baselinePoly(linLambda[maskRange]))
    #     plt.plot(maskVel,maskSpectrum*maskFinal)
    #     plt.show()
                     

    return outputs



def measureAllLineMoments(maskInfo,moments = [0],noiseIter=None):
	nLines = len(maskInfo)
	lineMoments = []
	for ll in range(nLines):
		specNoise = maskInfo[ll][0]    
		mask =  maskInfo[ll][1] 
		vel  = maskInfo[ll][2]
		spec = maskInfo[ll][3]

		moments = measureLineMoments(spec, specNoise, vel, mask = mask, moments = moments,noiseIter = noiseIter)

		lineMoments.append([specNoise] + moments)
	return lineMoments


def measureLineMoments(spectrum, specNoise, coordinate, mask = None, moments = [0],noiseIter = None): 

	if isinstance(mask,type(None)):
		mask = np.ones_like(spectrum)

	if np.any(mask==1):

		if isinstance(noiseIter,int):
			noiseArr = rng.normal(loc=0,scale=specNoise,size=(noiseIter,spectrum.shape[0])).T
		else:
			noiseArr = None

		moms = []
		for mm in moments:
			if mm == 0:
				coord = np.full_like(coordinate,np.abs(np.diff(coordinate))[0])
			else:
				coord = coordinate
			mom = weighted_moment(coord[mask.astype(bool)],weights=spectrum[mask.astype(bool)],moment=mm)
			# else:
			# 	coord = coordinate[mask.astype(bool)]
			# 	mom = weighted_moment(coord,spectrum[mask.astype(bool)],moment=mm)
			
			if not isinstance(noiseArr,type(None)):
				momDist = []
				for nn in range(noiseIter):
					momDist.extend([weighted_moment(coord[mask.astype(bool)],weights=(spectrum+noiseArr[:,nn])[mask.astype(bool)],moment=mm)])

				sigmaMom = median_absolute_deviation(np.array(momDist),Niter=5)[1]
			elif noiseIter == 'analytic':
				if mm == 0:
					sigmaMom = np.sqrt(np.nansum((specNoise*coord[mask.astype(bool)])**2))
				else:
					sigmaMom = np.nan
			else:
				sigmaMom = np.nan
				
			moms.extend([mom,sigmaMom])
	else:
		moms = [np.nan,np.nan]*len(moments)

	return moms



def baselineFunc(xx,aa,bb,cc):
    yy = aa*xx**2 + bb*xx + cc
    return yy


def extractLineMeasureOutputs(output):

	moments = np.zeros([len(output),len(output[0][1]),len(output[0][1][0])])

	maskSpecAll = []
	velSpecAll = []
	finalSpecAll = []

	specAll = []

	for ll in range(len(output[0][1])):

		maskLine = []
		velLine = []
		specLine = []

	
		for ss in range(len(output)):
			moments[ss,ll,:] = np.array(output[ss][1][ll][:]) 
			maskLine.append(output[ss][0][ll][1])
			velLine.append(output[ss][0][ll][2])
			specLine.append(output[ss][0][ll][3])

		# maskSpecAll.extend(np.array(maskLine))
		# velSpecAll.extend(np.array(velLine))
		# finalSpecAll.extend(np.array(specLine))
		specAll.append([np.array(velLine),np.array(maskLine),np.array(specLine)])

	return specAll,moments


def plotMoments(moments,names,imgShape,nameExt = ''):

	labels1  = ['M0: Jy/beam km/s']+[f'M{mm}: km/s' for mm in range(1,(moments.shape[2]-1)//2)]
	labels2  = ['S/N']+[f'unc. M{mm}: km/s' for mm in range(1,(moments.shape[2]-1)//2)]


	for ll in range(moments.shape[1]):

		fig, ax = plt.subplots(2,(moments.shape[2]-1)//2)
		fig.set_figheight(10)
		fig.set_figwidth(4*len(ax[0]))

		print((moments.shape[2]-1)//2)
		for mm in range((moments.shape[2]-1)//2):

			if mm==0:
				vmin1=0
				vmax1=np.percentile(moments[:,ll,1+2*mm][np.isfinite(moments[:,ll,1])],99)

				vmin2=0
				vmax2 = np.percentile((moments[:,ll,1+2*mm]/moments[:,ll,2*(mm+1)])[np.isfinite(moments[:,ll,1])],95)
			elif mm==1:
				med,MAD = median_absolute_deviation(moments[:,ll,1+2*mm],Niter=5)
				moments[:,ll,1+2*mm] -= med
				vmin1 = med-1.5*MAD
				vmax1 = med+1.5*MAD

				vmin2 = 0
				vmax2 = np.percentile(moments[:,ll,2*(mm+1)][np.isfinite(moments[:,ll,1])],95)
			elif mm == 2:	
				vmin1 = 0
				vmax1 = np.percentile(moments[:,ll,1+2*mm][np.isfinite(moments[:,ll,1])],50)

				vmin2 = 0
				vmax2 = np.percentile(moments[:,ll,2*(mm+1)][np.isfinite(moments[:,ll,1])],95)

			else:
				vmin1 = np.percentile(moments[:,ll,1+2*mm][np.isfinite(moments[:,ll,1])],5)
				vmax1 = np.percentile(moments[:,ll,1+2*mm][np.isfinite(moments[:,ll,1])],95)
				vmax1 = np.max([np.abs(vmin1),vmax1])
				vmin1 = -vmax1

				vmin2 = 0
				vmax2 = np.percentile(moments[:,ll,2*(mm+1)][np.isfinite(moments[:,ll,1])],95)


			img = ax[0,mm].imshow(moments[:,ll,1+2*mm].reshape(imgShape),cmap='viridis',vmin=vmin1,vmax=vmax1,origin='lower')
			fig.colorbar(img,ax=ax[0,mm],orientation='horizontal',label=labels1[mm])

			if mm == 0:
				img2 = ax[1,mm].imshow((moments[:,ll,1+2*mm]/moments[:,ll,2*(mm+1)]).reshape(imgShape),
		                 vmin=vmin2,vmax=vmax2,origin='lower')
			else:
				img2 = ax[1,mm].imshow(moments[:,ll,2*(mm+1)].reshape(imgShape),vmin=vmin2,vmax=vmax2,cmap='viridis',origin='lower')
			

			fig.colorbar(img2,ax=ax[1,mm],orientation='horizontal',label=labels2[mm])


		fig.tight_layout()
		fig.savefig(f"./figures/{names[ll][0].split('rms_')[-1]}_moment_maps{nameExt}.png")

def saveMoments(moments, imgShape,names, header,outname="./moments.fits"):
	
	mapsHDU = fits.HDUList([fits.PrimaryHDU()])
	for ll in range(moments.shape[1]):
		for nn in range(len(names[0])):
			mapsHDU.append(fits.ImageHDU(data=moments[:,ll,nn].reshape(imgShape),header=header,name=names[ll][nn]))
	        
	mapsHDU.writeto(outname,overwrite=True)


def saveSubcubes(spectra,imgShape,header,outname="./subcubes.fits"):
	cubesHDU = fits.HDUList([fits.PrimaryHDU()])

	# subMask = subcubeSpectra[0][1].reshape(imgShape[1],imgShape[0],np.array(subcubeSpectra[0][0][0]).shape[0])
	# subVel = subcubeSpectra[0][0].reshape(imgShape[1],imgShape[0],np.array(subcubeSpectra[0][0][0]).shape[0])
	# subSpectra = subcubeSpectra[0][2].reshape(imgShape[1],imgShape[0],np.array(subcubeSpectra[0][0][0]).shape[0])
	for ll in range(len(spectra)):
		for nn in range(len(spectra[0])):
			cubesHDU.append(fits.ImageHDU(data=np.swapaxes(spectra[ll][nn].reshape(imgShape[1],imgShape[0],np.array(spectra[0][0][0]).shape[0]),0,1).T,header=header))
		cubesHDU.append(fits.ImageHDU(data=np.swapaxes((spectra[ll][2]*spectra[ll][1]).reshape(imgShape[1],imgShape[0],np.array(spectra[0][0][0]).shape[0]),0,1).T,header=header))
	       
	cubesHDU.writeto(outname,overwrite=True)

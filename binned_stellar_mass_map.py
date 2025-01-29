##  Dr Adam B. Watts
##  Jan 2025
##  ICRAR / UWA
##  https://github.com/awattsup/astro-functions/tree/main
##  awattsup.github.io

import numpy as np
import copy

import os
import sys
import subprocess
import requests

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.convolution import convolve, Gaussian2DKernel

from reproject import reproject_exact



def spatiallyBinnedStellarMassMap(band1HDU, band2HDU, binmapHDU, Dcomov = 0, 
                                    convolve = False, band1FWHM = 1.3, band2FWHM = 1.3, targetFWHM = 0,
                                    stellarMassFunc = lambda gi_colour,obsmag_iband, Dcomov: lgMstarMsun_TaylorSAMI(gi_colour, obsmag_iband, Dcomov, z=0),
                                    surfaceDensity = True, outname = None
                                    ):
    ''' 
    Function to generate a stellar mass map from broadband imaging and a colour + magnitude calibration, mapped onto spatially-binned data (e.g. voronoi binning)
    also with Gaussian convolution for matching to lower spatial resoltuion data. 

    band1HDU: HDU           fits HDU containing image data and header
    band2HDU: HDU           fits HDU containing image data and header
    binmapHDU: HDU          fits hdu containing a header and an image where the pixel values are the spatial bin number they belong to

    Dcomov: float           in Mpc, for distance modulus calculation
    convolve: bool          include spatial convolution to match lower resoltuion data if True, don't convolve if False
    band1FWHM: float        image 1 FWHM (in arcsec) 
    band2FWHM: float        image 2 FWHM (in arcsec) 
    targetFWHM: float       FWHM for the stellar mass map - should match the data products you are trying to match this map to, e.g. MUSE/narrow-band H-alpha map
    stellarMassFunc: func   function that takes color, magnitude, and distance to compute log(Mstar / Msun). The default and other options are included in this file
    surfaceDensity: bool    output in units of Msun pc^-2 if True, output in Msun/pix if False 
    outName: str            output name (e.g. NGC4383)

    '''

    #get images
    obsFlux_band1 = band1HDU.data
    obsFlux_band2 = band2HDU.data

    broadbandPixsize_arcsec = WCS(band1HDU.header).proj_plane_pixel_scales()[0].value*3600
    broadbandPixarea_deg = WCS(band2HDU.header).proj_plane_pixel_area().value

    binmap = binmapHDU.data
    binmapPixsize_arcsec = WCS(binmapHDU.header).proj_plane_pixel_scales()[0].value*3600
    binmapPixarea_deg = WCS(binmapHDU.header).proj_plane_pixel_area().value


    ## Convolve seeing if you want to, note that typically MUSE has higher res than legacy/SDSS broadband imaging
    ##Maybe useful: Dey+19 DIQ for Legacy g-band is 1.3 arcsec, i-band not listed but redder bands are usually better, so I assume the same

    ## NOTE convolve() gets slow for large kernels, these are small so it is fine. convolve_fft() is better for going to e.g. 9" WISE resolution
    ##Preserve_nan == True is kept as otherwise the convolution makes data outside the observed region. Not much of an issue for small kernels, bad for large ones
    
    if convolve:
        sigmaDiff_band1 = np.sqrt(targetFWHM**2  - band1FWHM**2) / 2.355
        sigmaDiff_band2 = np.sqrt(targetFWHM**2  - band2FWHM**2) / 2.355

        obsFlux_band1 = convolve(obsFlux_band1,Gaussian2DKernel(sigmaDiff_band1/broadbandPixsize_arcsec),preserve_nan=True)
        obsFlux_band2 = convolve(obsFlux_band2,Gaussian2DKernel(sigmaDiff_band2/broadbandPixsize_arcsec),preserve_nan=True)
            
    


    ## Project broadband fluxes onto the binmap pixel scale, repoject_exact is the best version, but does not conserve flux properly
    pixarea_ratio = binmapPixarea_deg / broadbandPixarea_deg

    obsFluxReproj_band1 = reproject_exact((obsFlux_band1,band1HDU.header),binmapHDU.header,return_footprint=False) * pixarea_ratio
    obsFluxReproj_band1 = reproject_exact((obsFlux_band2,band2HDU.header),binmapHDU.header,return_footprint=False) * pixarea_ratio


    ## Sum the total flux in each bin and divide by number of pixels, so summing all pixels would return the total flux
    binmapMask = np.isfinite(binmap)

    obsFlux_band1_binned = np.full_like(binmap,np.nan)
    obsFlux_band2_binned = np.full_like(binmap,np.nan)
    binmapUnique = np.unique(binmap[binmapMask])
    for vv, vb in enumerate(binmapUnique):
        inbin = np.where(binmap == vb)
        obsFlux_band1_binned[inbin] = np.sum(obsFluxReproj_band1[inbin])   / len(inbin[0])   
        obsFlux_band2_binned[inbin] = np.sum(obsFluxReproj_band2[inbin]) / len(inbin[0])

    
    obsMag_band1_binned = 22.5 - 2.5*np.log10(obsFlux_band1_binned)
    obsMag_band2_binned = 22.5 - 2.5*np.log10(obsFlux_band2_binned)
    band12_colour_binned = obsMag_band1_binned - obsMag_band2_binned    
    
    MstarMsun_binned = 10.**stellarMassFunc(band12_colour, absMag_band2_binned, Dcomov)

    
    outHeader = copy.deepcopy(binmapHDU.header)
    if surfaceDensity:
        binmapPixarea_parsec = (Dcomov*1.6*np.tan(np.deg2rad(binmapPixsize_muse_deg)))**2.

        HDUname = "MSTARPC2"
        outNameExt = "Mstarpc2"
        outHDU = fits.ImageHDU(data=MstarMsun_binned/binmapPixarea_parsec,header = outHeader,name=HDUname)


    else:
        HDUname = "MSTAR"
        outNameExt = "Mstar"
        outHDU = fits.ImageHDU(data=MstarMsun_binned,header = outHeader,name=HDUname)

    outHDU.writeto(f"{outName}-{outNameExt}.fits",overwrite=True)





#These are my image-get functions 
def get_legacy_image(outfile, RA, DEC,imSize = 300,bands = 'r'):
    '''
    Fetch the image at the given RA, DEC from the Legacy server
    Keep in mind that at the time of writing, the Legacy data can be over-subtracted around galaxy outskirts, leading to a 'bowling' effect.
    #imSize should be in arcsec
    '''
    pix_scale = 0.262

    imsize = int(round(imSize/pix_scale)) #convert to pixels
    url = f"https://www.legacysurvey.org/viewer/fits-cutout/?ra={RA}&dec={DEC}&layer=ls-dr10&width={imsize}&height={imsize}&pixscale=0.262&bands={bands}"
   
    r = requests.get(url)
    if r.ok:
        with open(outfile, 'wb') as file:
            file.write(r.content)
    else:
        print('No coverage for (%s, %s)' % (RA, DEC))

        
def get_SDSS_mosaic_image(outFile, RA, DEC, imSize=1/6,filters = 'ugriz', scriptOnly=True):
    ####APPARENTLY THIS SERVICE HAS BEEN DECOMISSIONED, SORRY :(        
	#imSize needs to be in degrees
    if subprocess.run(['which','swarp']).returncode == 1:
        print("sWarp was not found in yourn $PATH, exiting")
        sys.exit(0)
        
    scriptFile = os.path.expanduser(f'{outFile.split('.fits')[0]}.sh')
    fileNameExt = outFile.split('/')[-1]
    cwd = outFile.split(fileNameExt)[0]

    url = f"https://dr12.sdss.org/mosaics/script?onlyprimary=False&pixelscale=0.396&ra={RA}&filters={filters}&dec={DEC}&size={imSize}"
    script = requests.get(url)
    if script.ok:
        with open(scriptFile,'wb') as file:
            file.write(script.content)
    else:
        print('Mosaic fetch failed, exiting')
        sys.exit(0)
    os.system(f"chmod +x {scriptFile}")


    if not scriptOnly:
        print('Running shell script to make mosaic')
        subprocess.run(['bash',scriptFile],cwd=cwd)
        print('Renaming output and cleaning up')
        subprocess.run(["rm frame*"],cwd=cwd,shell=True)
        subprocess.run(["rm *weight*"],cwd=cwd,shell=True)
        subprocess.run(["rm *.sh"],cwd=cwd,shell=True)
        subprocess.run(["rm *.swarp"],cwd=cwd,shell=True)	
        subprocess.run([f"mv J*.fits {outFile}"],cwd=cwd,shell=True)










###### Available stellar-mass calibrations ######


def lgMstarMsun_TaylorSAMI(obs_gmag, obs_imag, Dcomov, z=0):
	'''
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.447.2857B/abstract
    Equation 5

    #Dcomov should be in Mpc

    '''
    
    gi_color = obs_gmag - obs_imag

	Dmod = 5*(np.log10(Dcomov) - 1 + 6)

	logMstarMsun = -0.4*obs_imag + 0.4*Dmod - np.log10(1 + z) + \
					(1.2117 - 0.5893*z) + (0.7106 - 0.1467*z)*gi_colour


	return logMstarMsun


def lgMstarMsun_Zibetti09(obsmag_band1,obsmag_band2, Dcomov, band1='g', band2='i'):
	# https://ui.adsabs.harvard.edu/abs/2009MNRAS.400.1181Z

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


    Dmod = 5*(np.log10(Dcomov) - 1 + 6)

    colour_data = obsmag_band1 - obsmag_band2

	colour = f"{band1}{band2}"

    #only a couple presciptions are implemnted at the moment. 
	if colour == 'gi':
		if band2 == 'g':
			a = -1.197
			b = 1.431
            absMag_data = obsmag_band1 - Dmod
		elif band2 == 'i':
			a = -0.963
			b =  1.032
            absMag_data = obsmag_band2 - Dmod
			absMag_sun = 4.58
	elif colour == 'gr':
		if band2 == 'r':
			a = -0.840
			b = 1.654
			absMag_sun = 4.65

            absMag_data = obsmag_band2 - Dmod




	logMLr = a + b*colour_data

	logMstarMsun = logMLr + 0.4*(absMag_sun - absMag_data)

	return logMstarMsun

def lgMstarMsun_Taylor11(gi_colour, obsMag_i, Dcomov):
	#https://ui.adsabs.harvard.edu/abs/2011MNRAS.418.1587T

    Dmod = 5*(np.log10(Dcomov) - 1 + 6)

    absMag_i = obsMag_i - Dmod

	logMstarMsun = 1.15+0.7*(gi_colour) - 0.4*absMag_i

	return logMstarMsun


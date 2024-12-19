import numpy as np
import numpy.polynomial as npPoly

import matplotlib.pyplot as plt


import copy
import os


import astropy.constants as ac
import astropy.units as au



rng = np.random.default_rng()

#some useful things







def rot_mat(zrot,yrot,xrot):
	#should be right-multiplied with row vectors np.matmul([x,y,z],rot_mat)

	r1 = [np.cos(zrot)*np.cos(yrot),np.cos(zrot)*np.sin(yrot)*np.sin(xrot) - np.sin(zrot)*np.cos(xrot), np.cos(zrot)*np.sin(yrot)*np.cos(xrot)+np.sin(zrot)*np.sin(xrot)]
	r2 = [np.sin(zrot)*np.cos(yrot), np.sin(zrot)*np.sin(yrot)*np.sin(xrot)+np.cos(zrot)*np.cos(xrot), np.sin(zrot)*np.sin(yrot)*np.cos(xrot)-np.cos(zrot)*np.sin(xrot)]
	r3 = [-np.sin(yrot), np.cos(yrot)*np.sin(xrot), np.cos(yrot)*np.cos(xrot)]
	rr = np.array([r1,r2,r3])
	return rr


def calculate_luminosity_distance(z,H0 = 70.e0, omega_M = 0.3, omega_L = 0.7):
	dH = 2.99792e5 / H0
	dC = dH * int_Ez(z,omega_M,omega_L)
	dM = dC
	dL = (1.e0 + z) * dM
	return dL


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






if __name__ == '__main__':
	print('No main function bro')
	# make_WCSregion_polygon()
	# get_WCSregion_spectrum()
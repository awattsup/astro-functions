from scipy.stats import ks_2samp,anderson_ksamp
import numpy as np
from astropy.stats import sigma_clip
import astropy.constants as ac
import astropy.units as au
import scipy.stats as sps
from scipy.stats import distributions
import matplotlib.path as pltPath



rng = np.random.default_rng()




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


    return med, MAD



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





def gaussianPDF(xx,mu,sigma):
	
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





def inside_polygon(xcoords,ycoords,poly_x,poly_y):
	#borrowed/stolen from stack exchange: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

	path = pltPath.Path(np.array([poly_x,poly_y]).T)

	inside = path.contains_points(np.array([xcoords,ycoords]).T)

	return inside






def inside_polygon_slow(coordinates, poly_x, poly_y):
	#inspired by /copied from  the IDL-coyote routine 'inside', uses cross product, and is slow for many points
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

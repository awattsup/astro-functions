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



#### for manipulating particle data
def calc_COM(coordinates, masses, Rmax = None, Zmax = None):

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	z = coordinates[:,2]

	if Rmax != None:
		if Zmax != None:
			coordinates = coordinates[(radii < Rmax) & (z < Zmax)]
			masses = masses[(radii < Rmax) & (z < Zmax)]
		else:
			coordinates = coordinates[radii < Rmax]
			masses = masses[radii < Rmax]
	
	COM = np.array([np.nansum(coordinates[:,0]*masses, axis=0) / np.nansum(masses),\
					np.nansum(coordinates[:,1]*masses, axis=0) / np.nansum(masses),\
					np.nansum(coordinates[:,2]*masses, axis=0) / np.nansum(masses)])
	return COM

def calc_COV(coordinates, velocities, masses, Rmax = None, Zmax = None):

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	z = coordinates[:,2]

	if Rmax != None:
		if Zmax != None:
			velocities = velocities[(radii < Rmax) & (z < Zmax)]
			masses = masses[(radii < Rmax) & (z < Zmax)]
		else:
			velocities = velocities[radii < Rmax]
			masses = masses[radii < Rmax]

	COV = np.array([np.nansum(velocities[:,0]*masses, axis=0) / np.nansum(masses),\
					np.nansum(velocities[:,1]*masses, axis=0) / np.nansum(masses),\
					np.nansum(velocities[:,2]*masses, axis=0) / np.nansum(masses)])
	return COV

def diagonalise_inertia(coordinates, masses, rad):

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	coordinates = coordinates[radii < rad]
	masses = masses[radii < rad]
	
	I = np.zeros([3,3])
	for ii in range(3):
		for jj in range(3):
			if ii == jj:
				I[ii,jj] = np.nansum( (coordinates[:,(ii + 1)%3]**2.e0 + 
											coordinates[:,(jj + 2)%3]**2.e0 )*masses )
			else:
				I[ii,jj] = -1.e0*np.nansum(coordinates[:,ii]*coordinates[:,jj]*masses)

	eigval, eigvec = np.linalg.eig(I)
	eigval_argsort = eigval.argsort()
	# eigval = eigval[eigval_argsort]
	# eigvec = eigvec[eigval_argsort]
	# eigvec = np.linalg.inv(eigvec)
	eigvec = eigvec[eigval_argsort]

	return eigvec	

def orientation_matrix(coordinates, masses, show = False):

	Iprev = [[1,0,0],[0,1,0],[0,0,1]]
	eigvec_list = []

	rad = [1,2,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,55,60,80,100,150,200]		#kpc
	rr = 0
	Idiff = 1
	while(Idiff > 1.e-4):
		# print(rr,rad[rr])
		# eigvec = diagonalise_inertia(coordinates, masses, rad[rr])
		eigvec = diagonalise_inertia(coordinates, masses, 5)				#not sure why 5kpc works, but most galaxies converge to x-y orientation
		coordinates = coordinates @ eigvec 
		if show == True:
			plt.scatter(coordinates[:,0],coordinates[:,2],s=0.05)
			plt.xlim([-40,40])
			plt.ylim([-40,40])
			plt.show()
			plt.close()
		eigvec_list.append(eigvec)

		I = np.zeros([3,3])
		for ii in range(3):
			for jj in range(3):
				if ii == jj:
					I[ii,jj] = np.nansum( (coordinates[:,(ii + 1)%3]**2.e0 + 
												coordinates[:,(jj + 2)%3]**2.e0 )*masses )
				else:
					I[ii,jj] = -1.e0*np.nansum(coordinates[:,ii]*coordinates[:,jj]*masses)

		Idiff = np.abs((I[2][2] - Iprev[2][2]) / Iprev[2][2])
		# print(Idiff)
		Iprev = I
		# rr+=1
		if show == True:
			plt.scatter(coordinates[:,0],coordinates[:,2],s=0.1)
			plt.xlim([-40,40])
			plt.ylim([-40,40])
			plt.show()
			plt.close()


	eigvec = eigvec_list[0]
	for ii in range(1,len(eigvec_list)):
		eigvec = eigvec @ eigvec_list[ii]

	return eigvec

def calc_coords_obs(coordinates, view_phi, view_theta):
	view_theta *= np.pi/180.e0
	view_phi *= np.pi/180.e0

	coords_obs = np.zeros([len(coordinates),2])
	coords_obs[:,0] = (coordinates[:,0] * np.sin(view_phi) + 
							coordinates[:,1]*np.cos(view_phi))
	coords_obs[:,1] = ( (coordinates[:,0] * np.cos(view_phi) - 
							coordinates[:,1]*np.sin(view_phi)) * np.cos(view_theta) +
							coordinates[:,2] * np.sin(view_theta) )
	return coords_obs

def calc_vel_obs(velocities, view_phi, view_theta):
	view_theta *= np.pi/180.e0
	view_phi *= np.pi/180.e0

	vel_LOS = velocities[:,0]*np.cos(view_phi) * np.abs(np.sin(view_theta)) +\
				velocities[:,1] * -1.e0*np.sin(view_phi) * np.abs(np.sin(view_theta)) +\
				velocities[:,2] * np.cos(view_theta)								
	return vel_LOS

def calc_spatial_dist(coordinates, masses, Rmax):
	dim = 200
	image = np.zeros([dim,dim])
	dx = 2*Rmax/dim
	spacebins = np.arange(-1*Rmax,Rmax+dx,dx)
	area = dx*dx*1.e6
	for xx in range(len(spacebins)-1):
		xx_low = spacebins[xx]
		xx_high = spacebins[xx+1]
		for yy in range(len(spacebins)-1):
			yy_low = spacebins[yy]
			yy_high = spacebins[yy+1]

			image[yy,xx] = np.log10(np.nansum(masses[(coordinates[:,0] >= xx_low) & (coordinates[:,0]<xx_high) &\
									(coordinates[:,1]>=yy_low) & (coordinates[:,1]<yy_high)])/area)

	return spacebins[0:-1] + dx, image

def calc_sigma(coordinates, masses, Rmax = None, Zmax = None):

	z = coordinates[:,2]
	coordinates = calc_coords_obs(coordinates, 0, 0)
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))

	if Rmax != None:
		if Zmax != None:
			coordinates = coordinates[(radii < Rmax) & (z < Zmax)]
			masses = masses[(radii < Rmax) & (z < Zmax)]
		else:
			coordinates = coordinates[radii < Rmax]
			masses = masses[radii < Rmax]
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	radii_argsort = np.argsort(radii)
	radii = radii[radii_argsort]
	masses = masses[radii_argsort]

	Npart = len(masses)
	Nbins = 20
	rad_points = np.zeros(Nbins)
	sigma = np.zeros(Nbins)

	for ii in range(Nbins):
		low = (ii) * int(Npart / (Nbins))
		high = (ii + 1) * int(Npart / (Nbins))
		inbin_radii = radii[low:high]
		inbin_masses = masses[low:high]

		minrad = np.min(inbin_radii)
		maxrad = np.max(inbin_radii)
		area = np.pi * (maxrad * maxrad - minrad * minrad) * 1.e6
		sigma[ii] = np.log10(np.nansum(inbin_masses)/area)
		rad_points[ii] = np.median(inbin_radii)

	return rad_points, sigma

def calc_RC(coordinates, velocities, Rmax=None, Zmax=None):

	z = coordinates[:,2]
	coordinates = calc_coords_obs(coordinates, 0, 0)
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	
	if Rmax != None:
		if Zmax != None:
			coordinates = coordinates[(radii < Rmax) & (z < Zmax)]
			velocities = velocities[(radii < Rmax) & (z < Zmax)]
		else:
			coordinates = coordinates[radii < Rmax]
			velocities = velocities[radii < Rmax]
	vcirc = np.sqrt(velocities[:,0]**2.e0 + velocities[:,1]**2.e0)
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	radii_argsort = np.argsort(radii)
	radii = radii[radii_argsort]
	vcirc = vcirc[radii_argsort]

	Npart = len(coordinates)
	Nbins = 20
	rad_points = np.zeros(Nbins)
	rot_cur = np.zeros(Nbins)

	for ii in range(Nbins):
		low = (ii) * int(Npart / (Nbins))
		high = (ii + 1) * int(Npart / (Nbins))
		inbin_radii = radii[low:high]
		inbin_vcirc = vcirc[low:high]

		rot_cur[ii] = np.median(inbin_vcirc)
		rad_points[ii] = np.median(inbin_radii)

	return rad_points,rot_cur





if __name__ == '__main__':
	print('No main function bro')
	# make_WCSregion_polygon()
	# get_WCSregion_spectrum()
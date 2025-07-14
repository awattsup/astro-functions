import numpy as np 
from mpdaf.obj import Cube,Image
import matplotlib.pyplot as plt
# from astropy.






def main():


	cube = Cube('./NGC4383_DATACUBE_mosaic.fits')

	imI = cube.get_band_image('SDSS_i')

	# print(imI)
	# fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))


	# imI.plot(ax=ax1)

	# print(imI.get_data_hdu())
	# print(cube.data_header)
	# print(imI.data_header)

	# imI.write("./NGC4383_SDSS_i.fits")

	# plt.show()

	imI = Image("./NGC4383_SDSS_i.fits")
	print(imI.background())
	imI = imI - imI.background()[0]




if __name__ == "__main__":
	main()
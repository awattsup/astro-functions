"""
Object-oriented astronomy data analysis classes for datacubes.
Converted from functional style in image_cube_tools.py
"""

from astropy.io import fits
from astropy.wcs import WCS, Wcsprm
import astropy.constants as ac
import numpy as np

class DataCube:
    """
    A class to handle astronomical datacube operations.
    """
    
    def __init__(self, filepath=None, data=None, header=None):
        """
        Initialize DataCube object.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to FITS file containing datacube
        data : numpy.ndarray, optional
            3D data array (wavelength, y, x)
        header : astropy.io.fits.Header, optional
            FITS header with WCS information
        """
        # Core data attributes
        self.filepath = filepath
        self.data = data
        self.header = header
        self.wcs = None
        
        # Spectral axis attributes
        self.wavelength_axis = None
        self.frequency_axis = None
        
        # Individual pixel coordinate arrays
        self.pix_x = None
        self.pix_y = None
        self.pix_ra = None
        self.pix_dec = None
        self.pix_sp_x = None
        self.pix_sp_y = None
        
        # Essential header information
        self.ra_center = None
        self.dec_center = None
        self.pixel_scale = None
        self.spectral_resolution = None
        
        # Seeing and PSF attributes
        self.seeing_fwhm = None  # arcsec
        self.seeing_fwhm_pixels = None  # pixels
        self.psf_model = None
        self.psf_parameters = None
        
        # Initialize from file or data
        if filepath is not None:
            self.load_from_file(filepath)
        elif data is not None and header is not None:
            self.data = data
            self.header = header
            self._initialize()
    
    def load_from_file(self, filepath, hdu=1):
        """Load datacube from FITS file."""
        with fits.open(filepath) as hdul:
            self.data = hdul[hdu].data
            self.header = hdul[hdu].header
        self.filepath = filepath
        self._initialize()
    
    def _initialize(self):
        """Initialize all computed attributes."""
        if self.header is None:
            return
            
        # Create WCS object
        self.wcs = WCS(self.header)
        
        # Extract essential coordinate information
        self.ra_center = self.header.get('CRVAL1', None)
        self.dec_center = self.header.get('CRVAL2', None)
        
        # Calculate pixel scale using astropy's built-in functions
        if self.wcs is not None and hasattr(self.wcs, 'celestial'):
            try:
                # Use astropy's proj_plane_pixel_area for accurate pixel scale
                pixel_area = self.wcs.celestial.proj_plane_pixel_area()
                self.pixel_scale = np.sqrt(pixel_area.to('arcsec2').value)  # arcsec/pixel
            except:
                self.pixel_scale = None
        
        # Extract spectral resolution
        if 'NAXIS3' in self.header and self.header['NAXIS3'] > 1:
            cdelt3 = self.header.get('CDELT3', None)
            if cdelt3 is not None:
                self.spectral_resolution = abs(cdelt3)
        
        # Compute coordinate grids and spectral axes
        self.compute_wavelength_axis()
        self.compute_frequency_axis()
        self.compute_pixel_grids()
    
    @property
    def shape(self):
        """Return shape of datacube."""
        return self.data.shape if self.data is not None else None
    
    def compute_wavelength_axis(self, axtype=None):
        """Compute wavelength axis from WCS and store in self.wavelength_axis."""
        if self.wcs is None or self.header is None:
            return
            
        pix_wav = np.asarray(self.wcs.spectral.pixel_to_world(np.arange(self.header['NAXIS3'])))
        
        if axtype == "FREQ" or "FREQ" in self.wcs.axis_type_names:
            pix_wav = ac.c.value / pix_wav
            
        self.wavelength_axis = pix_wav
    
    def compute_frequency_axis(self, axtype=None):
        """Compute frequency axis from WCS and store in self.frequency_axis."""
        if self.wcs is None or self.header is None:
            return
            
        pix_freq = np.asarray(self.wcs.spectral.pixel_to_world(np.arange(self.header['NAXIS3'])))
        
        if axtype == "WAV" or "WAV" in self.wcs.axis_type_names:
            pix_freq = ac.c.value / pix_freq
            
        self.frequency_axis = pix_freq
    
    def compute_pixel_grids(self):
        """Create pixel coordinate grids and store as individual attributes."""
        if self.wcs is None or self.header is None:
            return
            
        wcs_celestial = self.wcs.celestial
        Nx = self.header['NAXIS1']
        Ny = self.header['NAXIS2']

        # Pixel coordinates (FITS begins at 1)
        pix_xx, pix_yy = np.meshgrid(np.arange(Nx, dtype=int) + 1,
                                   np.arange(Ny, dtype=int) + 1,
                                   indexing='xy')

        # Store individual pixel coordinate arrays
        self.pix_x = pix_xx.flatten()
        self.pix_y = pix_yy.flatten()

        # Pixel RA + DEC
        pix_radec = wcs_celestial.all_pix2world(np.column_stack([pix_xx.flatten(),
                                                               pix_yy.flatten()]), 1)
        
        # Store individual RA/DEC arrays
        self.pix_ra = pix_radec[:, 0]
        self.pix_dec = pix_radec[:, 1]

        # Sky plane coordinates
        pix_foc = wcs_celestial.wcs_world2pix(pix_radec, 1)
        wcsprm = Wcsprm(str(self.header).encode()).sub(['celestial'])
        out = wcsprm.p2s(pix_foc, 1)
        
        # Store individual sky-plane coordinate arrays
        self.pix_sp_x = np.asarray(out['imgcrd'])[:, 0]
        self.pix_sp_y = np.asarray(out['imgcrd'])[:, 1]
    
    def extract_subcube(self, subcube_index, mask=False):
        """
        Extract a subcube from the main datacube.
        
        Parameters:
        -----------
        subcube_index : list
            Indices for [wavelength, y, x] dimensions
        mask : array-like, optional
            Mask to apply to subcube
        
        Returns:
        --------
        DataCube
            New DataCube object containing the subcube
        """
        try:
            from spectral_cube import SpectralCube
        except ImportError:
            print('spectral_cube not installed, using basic slicing')
            return self._extract_subcube_basic(subcube_index)
        
        cube = SpectralCube(data=self.data, wcs=self.wcs)
        
        if not isinstance(mask, bool):
            cube = cube.with_mask(mask)
        
        if subcube_index[0] == 'all':
            subcube = cube[:,
                          subcube_index[1][0]:subcube_index[1][1],
                          subcube_index[2][0]:subcube_index[2][1]]
        else:
            subcube = cube[subcube_index[0][0]:subcube_index[0][1],
                          subcube_index[1][0]:subcube_index[1][1],
                          subcube_index[2][0]:subcube_index[2][1]]
        
        return DataCube(data=subcube.filled_data[:], header=subcube.header)
    
    def _extract_subcube_basic(self, subcube_index):
        """Basic subcube extraction without spectral_cube."""
        if subcube_index[0] == 'all':
            sub_data = self.data[:,
                               subcube_index[1][0]:subcube_index[1][1],
                               subcube_index[2][0]:subcube_index[2][1]]
        else:
            sub_data = self.data[subcube_index[0][0]:subcube_index[0][1],
                               subcube_index[1][0]:subcube_index[1][1],
                               subcube_index[2][0]:subcube_index[2][1]]
        
        # Create new header with updated dimensions
        new_header = self.header.copy()
        # Update NAXIS values based on subcube dimensions
        new_header['NAXIS1'] = sub_data.shape[2]
        new_header['NAXIS2'] = sub_data.shape[1]
        new_header['NAXIS3'] = sub_data.shape[0]
        
        return DataCube(data=sub_data, header=new_header)
    
    def convert_pixcoord_to_pixcoord(self, pix_xxyy, header_to):
        """Convert pixel coordinates from this datacube to another coordinate system and store result."""
        if self.wcs is None or self.header is None:
            return
            
        # Convert to RA/DEC first
        pix_radec = self.wcs.celestial.all_pix2world(pix_xxyy, 1)
        
        # Convert to new pixel coordinates
        wcs_to = WCS(header_to).celestial
        pix_xxyy_new = wcs_to.all_world2pix(pix_radec, 1)
        
        # Store converted coordinates
        self.converted_pix_x = pix_xxyy_new[:, 0]
        self.converted_pix_y = pix_xxyy_new[:, 1]
    
    def set_seeing_fwhm(self, fwhm_arcsec, psf_model=None, psf_parameters=None):
        """
        Set the seeing FWHM and related PSF information.
        
        Parameters:
        -----------
        fwhm_arcsec : float
            Seeing FWHM in arcseconds
        psf_model : str, optional
            PSF model type (e.g., 'gaussian', 'moffat')
        psf_parameters : dict, optional
            Additional PSF parameters
        """
        self.seeing_fwhm = fwhm_arcsec
        
        # Convert to pixels if pixel scale is available
        if self.pixel_scale is not None:
            self.seeing_fwhm_pixels = fwhm_arcsec / self.pixel_scale
        
        # Store PSF information
        if psf_model is not None:
            self.psf_model = psf_model
        if psf_parameters is not None:
            self.psf_parameters = psf_parameters
    
    def reproject_exact(self, output_projection, hdu=0):
        """Reproject datacube to new coordinate system using exact reprojection and return new DataCube."""
        try:
            from reproject import reproject_exact
        except ImportError:
            print('reproject package not available')
            return None
            
        input_wcs = self.wcs
        output_wcs = WCS(output_projection)
        output_shape = (input_wcs.array_shape[0], 
                       output_wcs.array_shape[0],
                       output_wcs.array_shape[1])
        
        pixarea_in = input_wcs.celestial.proj_plane_pixel_area().value * (3600**2)
        pixarea_out = output_wcs.celestial.proj_plane_pixel_area().value * (3600**2)
        pixel_conversion = pixarea_out / pixarea_in
        
        if self.header['NAXIS'] == 3:
            output_data = np.zeros(output_shape)
            
            for ii in range(self.header['NAXIS3']):
                input_slice = self.data[ii, ...]
                
                output_slice = reproject_exact((input_slice, input_wcs.celestial), 
                                             output_projection,
                                             return_footprint=False) * pixel_conversion
                output_data[ii, ...] = output_slice
                
                if ii % int(self.data.shape[0] / 10) == 0:
                    print(int(round(100 * ii / self.data.shape[0])), '% done')
        else:
            output_data = reproject_exact((self.data, input_wcs.celestial), 
                                        output_projection,
                                        return_footprint=False) * pixel_conversion
        
        # Create new header for reprojected datacube
        new_header = output_projection.copy()
        if self.header['NAXIS'] == 3:
            new_header['NAXIS3'] = self.header['NAXIS3']
            new_header['CRVAL3'] = self.header.get('CRVAL3', 0)
            new_header['CDELT3'] = self.header.get('CDELT3', 1)
            new_header['CRPIX3'] = self.header.get('CRPIX3', 1)
            new_header['CTYPE3'] = self.header.get('CTYPE3', 'WAVE')
            new_header['CUNIT3'] = self.header.get('CUNIT3', 'Angstrom')
        
        return DataCube(data=output_data, header=new_header)
    
    def deproject_pixel_coordinates(self, x0, y0, PA=0, incl=0):
        """
        Deproject pixel coordinates for galaxy analysis and store results.
        
        Parameters:
        -----------
        x0, y0 : float
            Center coordinates in sky-plane pixels
        PA : float
            Position angle in degrees (measured East of North)
        incl : float
            Inclination angle in degrees
        
        Results stored in:
        -----------------
        self.deprojected_x : numpy.ndarray
            Deprojected x coordinates
        self.deprojected_y : numpy.ndarray
            Deprojected y coordinates
        self.deprojected_r : numpy.ndarray
            Deprojected radial distances (arcsec)
        """
        if self.pix_sp_x is None or self.pix_sp_y is None:
            print("Sky-plane coordinates not available. Run compute_pixel_grids() first.")
            return
        
        # Convert angles to radians
        PA_rad = (PA - 90) * np.pi / 180.
        incl_rad = incl * np.pi / 180.
        
        # Center coordinates
        pix_xx_sky = self.pix_sp_x - x0
        pix_yy_sky = self.pix_sp_y - y0
        
        # Rotate to galaxy PA
        pix_xx_gal = pix_xx_sky * np.cos(PA_rad) - pix_yy_sky * np.sin(PA_rad)
        pix_yy_gal = (pix_xx_sky * np.sin(PA_rad) + pix_yy_sky * np.cos(PA_rad))
        
        # Incline y-axis
        pix_yy_gal *= 1. / np.cos(incl_rad)
        pix_xx_gal *= -1
        
        # Calculate radial distance in arcseconds
        pix_rr_gal = 3600 * np.sqrt(pix_xx_gal**2 + pix_yy_gal**2)
        
        # Store results
        self.deprojected_x = pix_xx_gal
        self.deprojected_y = pix_yy_gal
        self.deprojected_r = pix_rr_gal
    
    def get_sdss_image(self, arcsec=90, return_wcs=False):
        """
        Get SDSS image centered on datacube coordinates using image_cube_tools function.
        
        Parameters:
        -----------
        arcsec : float or list
            Size of image in arcseconds
        return_wcs : bool
            Whether to return WCS information
        
        Returns:
        --------
        PIL.Image or tuple
            SDSS image, optionally with coordinate grids and header
        """
        if self.ra_center is None or self.dec_center is None:
            print("RA/DEC center coordinates not available.")
            return None
        
        from image_cube_tools import get_SDSS_image
        return get_SDSS_image(self.ra_center, self.dec_center, arcsec=arcsec, RADEC=return_wcs)
    
    def get_legacy_colour_image(self, im_size=300):
        """
        Get Legacy Survey color image centered on datacube coordinates using image_cube_tools function.
        
        Parameters:
        -----------
        im_size : float
            Size of image in arcseconds
        
        Returns:
        --------
        tuple
            Legacy color image with coordinate grids and header
        """
        if self.ra_center is None or self.dec_center is None:
            print("RA/DEC center coordinates not available.")
            return None
        
        from image_cube_tools import get_legacy_colour_image
        return get_legacy_colour_image(self.ra_center, self.dec_center, imSize=im_size)
    
    def get_legacy_image(self, outfile, im_size=300, bands='r'):
        """
        Get Legacy Survey FITS image centered on datacube coordinates using image_cube_tools function.
        
        Parameters:
        -----------
        outfile : str
            Output filename
        im_size : float
            Size of image in arcseconds
        bands : str
            Bands to retrieve
        """
        if self.ra_center is None or self.dec_center is None:
            print("RA/DEC center coordinates not available.")
            return
        
        from image_cube_tools import get_legacy_image
        get_legacy_image(outfile, self.ra_center, self.dec_center, imSize=im_size, bands=bands)
    
    def get_galex_image(self, outfile, im_size=300):
        """
        Get GALEX image centered on datacube coordinates using image_cube_tools function.
        
        Parameters:
        -----------
        outfile : str
            Output filename
        im_size : float
            Size of image in arcseconds
        """
        if self.ra_center is None or self.dec_center is None:
            print("RA/DEC center coordinates not available.")
            return
        
        from image_cube_tools import get_galex_image
        get_galex_image(outfile, self.ra_center, self.dec_center, imSize=im_size)


class Image:
    """
    A class to handle astronomical 2D image operations.
    """
    
    def __init__(self, filepath=None, data=None, header=None):
        """
        Initialize Image object.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to FITS file containing image
        data : numpy.ndarray, optional
            2D data array (y, x)
        header : astropy.io.fits.Header, optional
            FITS header with WCS information
        """
        # Core data attributes
        self.filepath = filepath
        self.data = data
        self.header = header
        self.wcs = None
        
        # Individual pixel coordinate arrays
        self.pix_x = None
        self.pix_y = None
        self.pix_ra = None
        self.pix_dec = None
        self.pix_sp_x = None
        self.pix_sp_y = None
        
        # Essential header information
        self.ra_center = None
        self.dec_center = None
        self.pixel_scale = None
        
        # Seeing and PSF attributes
        self.seeing_fwhm = None  # arcsec
        self.seeing_fwhm_pixels = None  # pixels
        self.psf_model = None
        self.psf_parameters = None
        
        # Initialize from file or data
        if filepath is not None:
            self.load_from_file(filepath)
        elif data is not None and header is not None:
            self.data = data
            self.header = header
            self._initialize()
    
    def load_from_file(self, filepath, hdu=0):
        """Load image from FITS file."""
        with fits.open(filepath) as hdul:
            self.data = hdul[hdu].data
            self.header = hdul[hdu].header
        self.filepath = filepath
        self._initialize()
    
    def _initialize(self):
        """Initialize all computed attributes."""
        if self.header is None:
            return
            
        # Create WCS object
        self.wcs = WCS(self.header)
        
        # Extract essential coordinate information
        self.ra_center = self.header.get('CRVAL1', None)
        self.dec_center = self.header.get('CRVAL2', None)
        
        # Calculate pixel scale using astropy's built-in functions
        if self.wcs is not None and hasattr(self.wcs, 'celestial'):
            try:
                # Use astropy's proj_plane_pixel_area for accurate pixel scale
                pixel_area = self.wcs.celestial.proj_plane_pixel_area()
                self.pixel_scale = np.sqrt(pixel_area.to('arcsec2').value)  # arcsec/pixel
            except:
                self.pixel_scale = None
        
        # Compute coordinate grids
        self.compute_pixel_grids()
    
    @property
    def shape(self):
        """Return shape of image."""
        return self.data.shape if self.data is not None else None
    
    def compute_pixel_grids(self):
        """Create pixel coordinate grids and store as individual attributes."""
        if self.wcs is None or self.header is None:
            return
            
        wcs_celestial = self.wcs.celestial
        Nx = self.header['NAXIS1']
        Ny = self.header['NAXIS2']

        # Pixel coordinates (FITS begins at 1)
        pix_xx, pix_yy = np.meshgrid(np.arange(Nx, dtype=int) + 1,
                                   np.arange(Ny, dtype=int) + 1,
                                   indexing='xy')

        # Store individual pixel coordinate arrays
        self.pix_x = pix_xx.flatten()
        self.pix_y = pix_yy.flatten()

        # Pixel RA + DEC
        pix_radec = wcs_celestial.all_pix2world(np.column_stack([pix_xx.flatten(),
                                                               pix_yy.flatten()]), 1)
        
        # Store individual RA/DEC arrays
        self.pix_ra = pix_radec[:, 0]
        self.pix_dec = pix_radec[:, 1]

        # Sky plane coordinates
        pix_foc = wcs_celestial.wcs_world2pix(pix_radec, 1)
        wcsprm = Wcsprm(str(self.header).encode()).sub(['celestial'])
        out = wcsprm.p2s(pix_foc, 1)
        
        # Store individual sky-plane coordinate arrays
        self.pix_sp_x = np.asarray(out['imgcrd'])[:, 0]
        self.pix_sp_y = np.asarray(out['imgcrd'])[:, 1]
    
    def extract_subimage(self, subimage_index):
        """
        Extract a subimage from the main image.
        
        Parameters:
        -----------
        subimage_index : list
            Indices for [y, x] dimensions [[y_start, y_end], [x_start, x_end]]
        
        Returns:
        --------
        Image
            New Image object containing the subimage
        """
        sub_data = self.data[subimage_index[0][0]:subimage_index[0][1],
                           subimage_index[1][0]:subimage_index[1][1]]
        
        # Create new header with updated dimensions
        new_header = self.header.copy()
        new_header['NAXIS1'] = sub_data.shape[1]
        new_header['NAXIS2'] = sub_data.shape[0]
        
        # Update reference pixel positions
        if 'CRPIX1' in new_header:
            new_header['CRPIX1'] -= subimage_index[1][0]
        if 'CRPIX2' in new_header:
            new_header['CRPIX2'] -= subimage_index[0][0]
        
        return Image(data=sub_data, header=new_header)
    
    def convert_pixcoord_to_pixcoord(self, pix_xxyy, header_to):
        """Convert pixel coordinates from this image to another coordinate system and store result."""
        if self.wcs is None or self.header is None:
            return
            
        # Convert to RA/DEC first
        pix_radec = self.wcs.celestial.all_pix2world(pix_xxyy, 1)
        
        # Convert to new pixel coordinates
        wcs_to = WCS(header_to).celestial
        pix_xxyy_new = wcs_to.all_world2pix(pix_radec, 1)
        
        # Store converted coordinates
        self.converted_pix_x = pix_xxyy_new[:, 0]
        self.converted_pix_y = pix_xxyy_new[:, 1]
    
    def set_seeing_fwhm(self, fwhm_arcsec, psf_model=None, psf_parameters=None):
        """
        Set the seeing FWHM and related PSF information.
        
        Parameters:
        -----------
        fwhm_arcsec : float
            Seeing FWHM in arcseconds
        psf_model : str, optional
            PSF model type (e.g., 'gaussian', 'moffat')
        psf_parameters : dict, optional
            Additional PSF parameters
        """
        self.seeing_fwhm = fwhm_arcsec
        
        # Convert to pixels if pixel scale is available
        if self.pixel_scale is not None:
            self.seeing_fwhm_pixels = fwhm_arcsec / self.pixel_scale
        
        # Store PSF information
        if psf_model is not None:
            self.psf_model = psf_model
        if psf_parameters is not None:
            self.psf_parameters = psf_parameters
    
    def reproject_exact(self, output_projection):
        """Reproject image to new coordinate system using exact reprojection and return new Image."""
        try:
            from reproject import reproject_exact
        except ImportError:
            print('reproject package not available')
            return None
            
        input_wcs = self.wcs.celestial
        output_wcs = WCS(output_projection).celestial
        
        pixarea_in = input_wcs.proj_plane_pixel_area().value * (3600**2)
        pixarea_out = output_wcs.proj_plane_pixel_area().value * (3600**2)
        pixel_conversion = pixarea_out / pixarea_in
        
        output_data = reproject_exact((self.data, input_wcs), 
                                    output_projection,
                                    return_footprint=False) * pixel_conversion
        
        return Image(data=output_data, header=output_projection)
    
    def deproject_pixel_coordinates(self, x0, y0, PA=0, incl=0):
        """
        Deproject pixel coordinates for galaxy analysis and store results.
        
        Parameters:
        -----------
        x0, y0 : float
            Center coordinates in sky-plane pixels
        PA : float
            Position angle in degrees (measured East of North)
        incl : float
            Inclination angle in degrees
        
        Results stored in:
        -----------------
        self.deprojected_x : numpy.ndarray
            Deprojected x coordinates
        self.deprojected_y : numpy.ndarray
            Deprojected y coordinates
        self.deprojected_r : numpy.ndarray
            Deprojected radial distances (arcsec)
        """
        if self.pix_sp_x is None or self.pix_sp_y is None:
            print("Sky-plane coordinates not available. Run compute_pixel_grids() first.")
            return
        
        # Convert angles to radians
        PA_rad = (PA - 90) * np.pi / 180.
        incl_rad = incl * np.pi / 180.
        
        # Center coordinates
        pix_xx_sky = self.pix_sp_x - x0
        pix_yy_sky = self.pix_sp_y - y0
        
        # Rotate to galaxy PA
        pix_xx_gal = pix_xx_sky * np.cos(PA_rad) - pix_yy_sky * np.sin(PA_rad)
        pix_yy_gal = (pix_xx_sky * np.sin(PA_rad) + pix_yy_sky * np.cos(PA_rad))
        
        # Incline y-axis
        pix_yy_gal *= 1. / np.cos(incl_rad)
        pix_xx_gal *= -1
        
        # Calculate radial distance in arcseconds
        pix_rr_gal = 3600 * np.sqrt(pix_xx_gal**2 + pix_yy_gal**2)
        
        # Store results
        self.deprojected_x = pix_xx_gal
        self.deprojected_y = pix_yy_gal
        self.deprojected_r = pix_rr_gal
    
    def get_sdss_image(self, arcsec=90, return_wcs=False):
        """
        Get SDSS image centered on image coordinates using image_cube_tools function.
        
        Parameters:
        -----------
        arcsec : float or list
            Size of image in arcseconds
        return_wcs : bool
            Whether to return WCS information
        
        Returns:
        --------
        PIL.Image or tuple
            SDSS image, optionally with coordinate grids and header
        """
        if self.ra_center is None or self.dec_center is None:
            print("RA/DEC center coordinates not available.")
            return None
        
        from image_cube_tools import get_SDSS_image
        return get_SDSS_image(self.ra_center, self.dec_center, arcsec=arcsec, RADEC=return_wcs)
    
    def get_legacy_colour_image(self, im_size=300):
        """
        Get Legacy Survey color image centered on image coordinates using image_cube_tools function.
        
        Parameters:
        -----------
        im_size : float
            Size of image in arcseconds
        
        Returns:
        --------
        tuple
            Legacy color image with coordinate grids and header
        """
        if self.ra_center is None or self.dec_center is None:
            print("RA/DEC center coordinates not available.")
            return None
        
        from image_cube_tools import get_legacy_colour_image
        return get_legacy_colour_image(self.ra_center, self.dec_center, imSize=im_size)
    
    def get_legacy_image(self, outfile, im_size=300, bands='r'):
        """
        Get Legacy Survey FITS image centered on image coordinates using image_cube_tools function.
        
        Parameters:
        -----------
        outfile : str
            Output filename
        im_size : float
            Size of image in arcseconds
        bands : str
            Bands to retrieve
        """
        if self.ra_center is None or self.dec_center is None:
            print("RA/DEC center coordinates not available.")
            return
        
        from image_cube_tools import get_legacy_image
        get_legacy_image(outfile, self.ra_center, self.dec_center, imSize=im_size, bands=bands)
    
    def get_galex_image(self, outfile, im_size=300):
        """
        Get GALEX image centered on image coordinates using image_cube_tools function.
        
        Parameters:
        -----------
        outfile : str
            Output filename
        im_size : float
            Size of image in arcseconds
        """
        if self.ra_center is None or self.dec_center is None:
            print("RA/DEC center coordinates not available.")
            return
        
        from image_cube_tools import get_galex_image
        get_galex_image(outfile, self.ra_center, self.dec_center, imSize=im_size)

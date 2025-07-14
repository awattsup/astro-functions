#!/usr/bin/env python3
"""
Simple demonstration of the DataCube class conversion from functional to OOP style.
"""

import numpy as np
from astropy.io import fits
from datacube_classes import DataCube

def demo_datacube_basics():
    """
    Demonstrate basic DataCube functionality.
    """
    print("=== DataCube Class Demo ===\n")
    
    # Create mock datacube data
    print("1. Creating mock datacube...")
    nwave, ny, nx = 100, 50, 50
    wavelengths = np.linspace(4800, 5200, nwave)  # Angstroms
    data = np.random.normal(0, 0.1, (nwave, ny, nx))
    
    # Create mock header
    header = fits.Header()
    header['NAXIS'] = 3
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny  
    header['NAXIS3'] = nwave
    header['CRPIX1'] = nx // 2
    header['CRPIX2'] = ny // 2
    header['CRPIX3'] = 1
    header['CRVAL1'] = 150.0  # RA
    header['CRVAL2'] = 2.0    # DEC
    header['CRVAL3'] = wavelengths[0]
    header['CDELT1'] = -0.001  # degrees
    header['CDELT2'] = 0.001   # degrees
    header['CDELT3'] = wavelengths[1] - wavelengths[0]  # Angstroms
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CTYPE3'] = 'WAVE'
    header['CUNIT3'] = 'Angstrom'
    
    print(f"   Created mock data with shape: {data.shape}")
    
    print("\n2. FUNCTIONAL APPROACH (old way):")
    print("   from image_cube_tools import get_wavelength_axis, make_pix_WCS_grids")
    print("   wavelength_axis = get_wavelength_axis(header)")
    print("   pix_xxyy, pix_RADEC, pix_SP = make_pix_WCS_grids(header)")
    print("   # Manual management of all these variables...")
    
    print("\n3. OBJECT-ORIENTED APPROACH (new way):")
    
    # Create DataCube object
    datacube = DataCube(data=data, header=header)
    print(f"   Created DataCube: {datacube}")
    print(f"   Shape: {datacube.shape}")
    
    # Access computed attributes
    print(f"   Wavelength axis: {len(datacube.wavelength_axis)} points")
    print(f"   Range: {datacube.wavelength_axis[0]:.1f} - {datacube.wavelength_axis[-1]:.1f} Å")
    
    # Access individual pixel coordinate arrays
    print(f"   Pixel grids computed: {len(datacube.pix_x)} pixels")
    print(f"   RA range: {np.min(datacube.pix_ra):.3f} - {np.max(datacube.pix_ra):.3f} deg")
    print(f"   DEC range: {np.min(datacube.pix_dec):.3f} - {np.max(datacube.pix_dec):.3f} deg")
    print(f"   Sky-plane X range: {np.min(datacube.pix_sp_x):.6f} - {np.max(datacube.pix_sp_x):.6f} deg")
    print(f"   Sky-plane Y range: {np.min(datacube.pix_sp_y):.6f} - {np.max(datacube.pix_sp_y):.6f} deg")
    
    # Show extracted header information
    print(f"   Essential header info extracted automatically:")
    print(f"   - RA center: {datacube.ra_center:.6f} deg" if datacube.ra_center else "   - RA center: Not available")
    print(f"   - DEC center: {datacube.dec_center:.6f} deg" if datacube.dec_center else "   - DEC center: Not available")
    print(f"   - Pixel scale: {datacube.pixel_scale:.3f} arcsec/pixel" if datacube.pixel_scale else "   - Pixel scale: Not available")
    print(f"   - Spectral resolution: {datacube.spectral_resolution:.2f} Å" if datacube.spectral_resolution else "   - Spectral resolution: Not available")
    
    return datacube

def demo_subcube_extraction(datacube):
    """
    Demonstrate subcube extraction.
    """
    print("\n4. Subcube extraction:")
    
    # Extract a subcube
    subcube_indices = [[20, 80], [10, 40], [10, 40]]  # [wave, y, x]
    subcube = datacube.extract_subcube(subcube_indices)
    
    print(f"   Original cube shape: {datacube.shape}")
    print(f"   Subcube shape: {subcube.shape}")
    print(f"   Subcube wavelength range: {subcube.wavelength_axis[0]:.1f} - {subcube.wavelength_axis[-1]:.1f} Å")
    
    return subcube

def demo_coordinate_access(datacube):
    """
    Demonstrate accessing individual coordinate arrays.
    """
    print("\n5. Individual coordinate arrays:")
    
    # Show how to access different coordinate systems
    print(f"   Pixel coordinates: datacube.pix_x, datacube.pix_y")
    print(f"   RA/DEC coordinates: datacube.pix_ra, datacube.pix_dec")
    print(f"   Sky-plane coordinates: datacube.pix_sp_x, datacube.pix_sp_y")
    
    # Example: Find pixel closest to a specific RA/DEC
    target_ra, target_dec = 150.0, 2.0
    distances = np.sqrt((datacube.pix_ra - target_ra)**2 + (datacube.pix_dec - target_dec)**2)
    closest_idx = np.argmin(distances)
    
    print(f"   Example: Closest pixel to RA={target_ra}, DEC={target_dec}:")
    print(f"   Pixel coordinates: ({datacube.pix_x[closest_idx]:.0f}, {datacube.pix_y[closest_idx]:.0f})")
    print(f"   Actual RA/DEC: ({datacube.pix_ra[closest_idx]:.6f}, {datacube.pix_dec[closest_idx]:.6f})")

def demo_seeing_measurements(datacube):
    """
    Demonstrate seeing FWHM measurements and PSF information.
    """
    print("\n6. Seeing and PSF measurements:")
    
    # Set seeing FWHM
    datacube.set_seeing_fwhm(1.2, psf_model='gaussian', psf_parameters={'beta': 2.5})
    
    print(f"   Set seeing FWHM: {datacube.seeing_fwhm} arcsec")
    print(f"   Seeing FWHM in pixels: {datacube.seeing_fwhm_pixels:.2f} pixels" if datacube.seeing_fwhm_pixels else "   Seeing FWHM in pixels: Not available (no pixel scale)")
    print(f"   PSF model: {datacube.psf_model}")
    print(f"   PSF parameters: {datacube.psf_parameters}")

def demo_comparison():
    """
    Show the key differences between functional and OOP approaches.
    """
    print("\n8. KEY ADVANTAGES OF OOP APPROACH:")
    print("   ✓ Encapsulation: Data and methods bundled together")
    print("   ✓ State management: Wavelength axis and pixel grids computed once")
    print("   ✓ Individual coordinate access: datacube.pix_ra instead of pix_RADEC[0]")
    print("   ✓ Cleaner interface: datacube.wavelength_axis vs get_wavelength_axis(header)")
    print("   ✓ Method chaining: datacube.extract_subcube().compute_wavelength_axis()")
    print("   ✓ Easier to extend: Add new methods without changing function signatures")
    print("   ✓ Automatic header extraction: All useful header info extracted on initialization")
    print("   ✓ Seeing measurements: Built-in support for PSF and seeing information")
    
    print("\n9. MIGRATION STRATEGY:")
    print("   - Replace manual FITS loading with DataCube(filepath='...')")
    print("   - Replace get_wavelength_axis(header) with datacube.wavelength_axis")
    print("   - Replace make_pix_WCS_grids(header) with individual coordinate arrays:")
    print("     * pix_xxyy[0] → datacube.pix_x")
    print("     * pix_xxyy[1] → datacube.pix_y") 
    print("     * pix_RADEC[0] → datacube.pix_ra")
    print("     * pix_RADEC[1] → datacube.pix_dec")
    print("     * pix_SP[0] → datacube.pix_sp_x")
    print("     * pix_SP[1] → datacube.pix_sp_y")
    print("   - Use datacube.extract_subcube() instead of manual array slicing")

if __name__ == "__main__":
    print("Simple DataCube Class Demonstration")
    print("=" * 50)
    
    try:
        # Run demonstrations
        datacube = demo_datacube_basics()
        subcube = demo_subcube_extraction(datacube)
        demo_coordinate_access(datacube)
        demo_seeing_measurements(datacube)
        demo_comparison()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nYour DataCube class is ready to use!")
        print("Next steps:")
        print("- Replace your functional datacube loading with: datacube = DataCube('file.fits')")
        print("- Access wavelength axis with: datacube.wavelength_axis")
        print("- Access pixel grids with: datacube.pixel_grids")
        print("- Extract subcubes with: datacube.extract_subcube(indices)")
        
    except Exception as e:
        print(f"Demo encountered an error: {e}")
        print("Check that astropy is installed and working correctly.")

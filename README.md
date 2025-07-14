# Object-oriented data analysis

I'm working on converting my historically functional style of working with images and datacubes to an object-oriented programming (OOP) style. 

*UNDER CONSTRUCTION*
The conversion isn't yet complete and still has a few redundancies



The new object-oriented approach provides:
- Better code organization and maintainability
- Encapsulation of data and methods
- Automatic state management and caching
- Easier extensibility and reusability

## New Object-Oriented Classes

### DataCube Class (`datacube.py`)

The main class for handling astronomical datacubes.

```python
from datacube import DataCube

# Load from FITS file
datacube = DataCube(filepath='my_datacube.fits')

# Or create from data arrays
datacube = DataCube(data=data_array, header=fits_header)

# Access properties (automatically computed)
wavelengths = datacube.wavelength_axis
frequencies = datacube.frequency_axis
pixel_xx = datacube.pixel_x
pixel_RA = datacube.pixel_ra

shape = datacube.shape

# Extract subcubes
subcube = datacube.extract_subcube([[100, 200], [50, 100], [50, 100]])


#Get VO images
datacube.get_sdss_image()
```

### Image Class (`datacube.py`)

The main class for handling astronomical images.

```python
from datacube import Image

# Load from FITS file
image = Image(filepath='my_image.fits')

# Or create from data arrays
image = Image(data=data_array, header=fits_header)

# Access properties (automatically computed)
pixel_xx = image.pixel_x
pixel_RA = image.pixel_ra

shape = image.shape

#Get VO images
image.get_sdss_image()
```


## Running the Demo

Execute the demonstration script to see the conversion in action:

```bash
python datacube_demo.py
```


## Dependencies

- `astropy` - FITS file handling, WCS transformations, units
- `numpy` - Numerical computations
- `scipy` - Statistical functions, optimization
- `matplotlib` - Plotting and visualization

## Future Extensions

Things I'm working on adding:

- **Simple emssion line identification**: For continuum-subtracted cubes 



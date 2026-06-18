# flyspanker
A tool for quickly examining fluxes in an astronomical image.

## Installation

```bash
pip install flyspanker
```

Or install directly from the repository:

```bash
git clone https://github.com/zkbt/flyspanker.git
cd flyspanker
pip install -e .
```

## Usage in a Jupyter Notebook

```python
%matplotlib widget          # enable interactive matplotlib backend
from flyspanker import Spanker

s = Spanker("my_image.fits")
```

This will display the FITS image.  **Click on any star** to:

* place a circular aperture centred on the click,
* compute the centroid position within the aperture, and
* measure the total flux (sum of pixel values) inside it.

An **Aperture radius** slider appears below the image so you can adjust the
aperture size interactively without clicking again.
A **radial profile?** checkbox appears below the slider; when enabled, flyspanker
plots a radial brightness profile from the centroid out to the current aperture
radius and estimates the star FWHM.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filename` | — | Path to the FITS file |
| `ext` | `0` | FITS extension index to read |
| `cmap` | `'gray'` | Matplotlib colormap |
| `figsize` | `(8, 8)` | Figure size in inches |

### Accessing measurements

The most recent measurement result is stored as `s.last_result`:

```python
print(s.last_result)
# {'x': 312.4, 'y': 198.1,
#  'x_centroid': 311.87, 'y_centroid': 197.65,
#  'flux': 48312.5, 'fwhm': 6.42}
```

### Calling `measure()` programmatically

```python
result = s.measure(x=311, y=197, radius=12)
print(result["flux"])
```

## Dependencies

* [astropy](https://www.astropy.org/) — FITS file loading and ZScale display stretching
* [matplotlib](https://matplotlib.org/) — image display and click interaction
* [photutils](https://photutils.readthedocs.io/) — centroid calculation and aperture photometry
* [ipywidgets](https://ipywidgets.readthedocs.io/) — interactive slider
* [ipympl](https://matplotlib.org/ipympl/) — interactive matplotlib backend for Jupyter

## Running tests

```bash
pip install -e ".[test]"
pytest
```

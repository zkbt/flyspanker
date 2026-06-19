"""
Tests for flyspanker.Spanker.

These tests use a synthetic FITS file with a known Gaussian star profile so
that the centroid and flux results are predictable.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import pytest
from astropy.io import fits

from flyspanker import Spanker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fits(tmp_path: Path, *, nx: int = 100, ny: int = 100,
               star_x: float = 50.0, star_y: float = 50.0,
               amplitude: float = 1000.0, sigma: float = 3.0,
               noise: float = 0.0, background: float = 0.0,
               filename: str = "test_image.fits") -> Path:
    """Create a FITS file with a synthetic Gaussian star and return its path."""
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    xx, yy = np.meshgrid(x, y)
    data = background + amplitude * np.exp(
        -((xx - star_x) ** 2 + (yy - star_y) ** 2) / (2 * sigma ** 2)
    )
    if noise > 0:
        rng = np.random.default_rng(42)
        data += rng.normal(0, noise, data.shape)

    fitsfile = tmp_path / filename
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.writeto(fitsfile, overwrite=True)
    return fitsfile


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpankerInit:
    def test_loads_data(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        assert s.data.ndim == 2
        assert s.data.shape == (100, 100)

    def test_default_radius(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        assert s.radius == 10.0

    def test_repr(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        r = repr(s)
        assert "Spanker" in r
        assert "100" in r

    def test_last_result_initially_none(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        assert s.last_result is None

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            Spanker(tmp_path / "nonexistent.fits")

    def test_bad_extension_raises(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        with pytest.raises(Exception):
            Spanker(fitsfile, ext=99)


class TestMeasure:
    """Tests for Spanker.measure() — the core centroid+flux routine."""

    def test_centroid_near_star(self, tmp_path):
        star_x, star_y = 50.0, 50.0
        fitsfile = _make_fits(tmp_path, star_x=star_x, star_y=star_y)
        s = Spanker(fitsfile)
        result = s.measure(star_x + 2, star_y - 1, radius=10)
        assert abs(result["x_centroid"] - star_x) < 1.0
        assert abs(result["y_centroid"] - star_y) < 1.0

    def test_flux_positive(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        result = s.measure(50, 50, radius=10)
        assert result["flux"] > 0

    def test_flux_increases_with_radius(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        flux_small = s.measure(50, 50, radius=5)["flux"]
        flux_large = s.measure(50, 50, radius=15)["flux"]
        assert flux_large > flux_small

    def test_result_keys(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        result = s.measure(50, 50, radius=10)
        for key in ("x", "y", "x_centroid", "y_centroid", "flux"):
            assert key in result
        for key in (
            "raw_flux",
            "background_per_pixel",
            "background_flux",
            "subtract_background",
            "sky_inner_radius",
            "sky_outer_radius",
        ):
            assert key in result

    def test_click_out_of_bounds_does_not_crash(self, tmp_path):
        """Clicking far outside valid data should return nan flux gracefully."""
        fitsfile = _make_fits(tmp_path, nx=50, ny=50)
        s = Spanker(fitsfile)
        result = s.measure(-5, -5, radius=2)
        assert math.isnan(result["flux"]) or result["flux"] >= 0

    def test_different_radius_changes_flux(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        r1 = s.measure(50, 50, radius=8)
        r2 = s.measure(50, 50, radius=20)
        assert r1["flux"] != r2["flux"]

    def test_off_centre_click_still_centroid_correct(self, tmp_path):
        """Clicking slightly off-center still finds the correct centroid."""
        fitsfile = _make_fits(tmp_path, star_x=40.0, star_y=60.0)
        s = Spanker(fitsfile)
        result = s.measure(42, 58, radius=12)
        assert abs(result["x_centroid"] - 40.0) < 1.5
        assert abs(result["y_centroid"] - 60.0) < 1.5

    def test_background_subtraction_reduces_flux(self, tmp_path):
        fitsfile = _make_fits(tmp_path, background=100.0)
        s = Spanker(fitsfile)
        raw = s.measure(50, 50, radius=10)
        sub = s.measure(
            50,
            50,
            radius=10,
            subtract_background=True,
            sky_gap=3.0,
            sky_outer_radius=20.0,
        )
        assert sub["flux"] < raw["flux"]
        assert sub["background_per_pixel"] == pytest.approx(100.0, rel=0.05)

    def test_background_subtraction_matches_zero_background_case(self, tmp_path):
        fitsfile_bg = _make_fits(tmp_path, background=75.0, filename="with_bg.fits")
        fitsfile_no_bg = _make_fits(tmp_path, background=0.0, filename="no_bg.fits")
        s_bg = Spanker(fitsfile_bg)
        s_no_bg = Spanker(fitsfile_no_bg)

        sub = s_bg.measure(
            50,
            50,
            radius=10,
            subtract_background=True,
            sky_gap=3.0,
            sky_outer_radius=20.0,
        )
        no_bg = s_no_bg.measure(50, 50, radius=10)
        assert sub["flux"] == pytest.approx(no_bg["flux"], rel=0.05)


class TestDisplay:
    """Smoke tests: verify the figure and axes are created without errors."""

    def test_figure_created(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        assert s.fig is not None
        assert s.ax is not None

    def test_draw_aperture_does_not_crash(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        s._draw_aperture(50, 50)
        assert s._aperture_patch is not None

    def test_draw_aperture_twice(self, tmp_path):
        """Drawing the aperture twice replaces the old one without error."""
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        s._draw_aperture(40, 40)
        s._draw_aperture(60, 60)
        assert s._aperture_patch is not None

    def test_update_status_does_not_crash(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        result = s.measure(50, 50, radius=10)
        s._update_status(result)

    def test_auto_centroid_default_true(self, tmp_path):
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)
        assert s.auto_centroid is True

    def test_auto_centroid_toggle_moves_aperture(self, tmp_path):
        """Toggling auto_centroid=False keeps the aperture at the original click."""
        fitsfile = _make_fits(tmp_path, star_x=50.0, star_y=50.0)
        s = Spanker(fitsfile)
        s.auto_centroid = False

        class _FakeEvent:
            inaxes = s.ax
            xdata = 55.0
            ydata = 45.0

        s._on_click(_FakeEvent())
        assert s.last_result is not None
        assert s.last_result["x_centroid"] == 55.0
        assert s.last_result["y_centroid"] == 45.0

    def test_click_ignored_during_zoom(self, tmp_path):
        """No aperture is placed when the toolbar mode is zoom or pan."""
        fitsfile = _make_fits(tmp_path)
        s = Spanker(fitsfile)

        # Simulate the toolbar being in zoom mode
        class _FakeToolbar:
            mode = "zoom rect"

        s.fig.canvas.toolbar = _FakeToolbar()

        class _FakeEvent:
            inaxes = s.ax
            xdata = 50.0
            ydata = 50.0

        s._on_click(_FakeEvent())
        assert s.last_result is None  # no measurement should have been made


class TestFitsVariants:
    """Test loading FITS files with various quirks."""

    def test_cube_collapses_to_2d(self, tmp_path):
        """A 3-D data cube should be collapsed to the first 2-D slice."""
        data = np.ones((3, 64, 64), dtype=np.float32)
        fitsfile = tmp_path / "cube.fits"
        fits.PrimaryHDU(data=data).writeto(fitsfile)
        s = Spanker(fitsfile)
        assert s.data.ndim == 2

    def test_extension_selection(self, tmp_path):
        """Reading a named extension works."""
        primary = fits.PrimaryHDU()
        image_hdu = fits.ImageHDU(data=np.ones((50, 50), dtype=np.float32))
        hdul = fits.HDUList([primary, image_hdu])
        fitsfile = tmp_path / "multi.fits"
        hdul.writeto(fitsfile)
        s = Spanker(fitsfile, ext=1)
        assert s.data.shape == (50, 50)

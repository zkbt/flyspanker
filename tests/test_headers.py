"""
Tests for flyspanker.headers.summarize_fits_headers.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from flyspanker.headers import summarize_fits_headers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fits(path, header_kw):
    """Write a minimal FITS file with the given keyword dict."""
    hdr = fits.Header()
    for key, val in header_kw.items():
        hdr[key] = val
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4)), header=hdr)
    hdu.writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

def test_returns_table(tmp_path):
    _make_fits(tmp_path / "a.fits", {"EXPTIME": 10.0, "OBJECT": "M31"})
    result = summarize_fits_headers(tmp_path)
    assert isinstance(result, Table)


def test_filename_column_first(tmp_path):
    _make_fits(tmp_path / "a.fits", {"EXPTIME": 10.0})
    result = summarize_fits_headers(tmp_path)
    assert result.colnames[0] == "filename"


def test_one_row_per_file(tmp_path):
    for i in range(3):
        _make_fits(tmp_path / f"img{i}.fits", {"EXPTIME": float(i)})
    result = summarize_fits_headers(tmp_path)
    assert len(result) == 3


def test_filenames_in_table(tmp_path):
    names = ["alpha.fits", "beta.fits"]
    for n in names:
        _make_fits(tmp_path / n, {"EXPTIME": 1.0})
    result = summarize_fits_headers(tmp_path)
    assert set(result["filename"]) == set(names)


def test_common_keywords_present(tmp_path):
    _make_fits(tmp_path / "a.fits", {"EXPTIME": 30.0, "FILTER": "V"})
    _make_fits(tmp_path / "b.fits", {"EXPTIME": 60.0, "FILTER": "B"})
    result = summarize_fits_headers(tmp_path)
    assert "EXPTIME" in result.colnames
    assert "FILTER" in result.colnames


def test_missing_keyword_is_masked(tmp_path):
    _make_fits(tmp_path / "a.fits", {"EXPTIME": 30.0, "FILTER": "V"})
    _make_fits(tmp_path / "b.fits", {"EXPTIME": 60.0})  # no FILTER
    result = summarize_fits_headers(tmp_path)
    # find row for b.fits
    idx = list(result["filename"]).index("b.fits")
    assert result["FILTER"].mask[idx]


def test_numeric_column_dtype(tmp_path):
    _make_fits(tmp_path / "a.fits", {"EXPTIME": 10.0})
    _make_fits(tmp_path / "b.fits", {"EXPTIME": 20.0})
    result = summarize_fits_headers(tmp_path)
    assert np.issubdtype(result["EXPTIME"].dtype, np.floating)


def test_string_column_dtype(tmp_path):
    _make_fits(tmp_path / "a.fits", {"OBJECT": "M31"})
    _make_fits(tmp_path / "b.fits", {"OBJECT": "M42"})
    result = summarize_fits_headers(tmp_path)
    assert result["OBJECT"].dtype.kind in ("U", "S", "O")


def test_directory_not_found_raises():
    with pytest.raises(FileNotFoundError):
        summarize_fits_headers("/nonexistent/path/xyz")


def test_no_fits_files_raises(tmp_path):
    (tmp_path / "readme.txt").write_text("hello")
    with pytest.raises(ValueError, match="No FITS files"):
        summarize_fits_headers(tmp_path)


def test_custom_pattern(tmp_path):
    _make_fits(tmp_path / "a.fit", {"EXPTIME": 1.0})
    _make_fits(tmp_path / "b.fits", {"EXPTIME": 2.0})
    # Only match *.fit (not *.fits)
    result = summarize_fits_headers(tmp_path, pattern="*.fit")
    assert len(result) == 1
    assert result["filename"][0] == "a.fit"


def test_values_correct(tmp_path):
    _make_fits(tmp_path / "single.fits", {"EXPTIME": 99.0, "OBJECT": "NGC1"})
    result = summarize_fits_headers(tmp_path)
    row = result[result["filename"] == "single.fits"][0]
    assert float(row["EXPTIME"]) == pytest.approx(99.0)
    assert str(row["OBJECT"]).strip() == "NGC1"

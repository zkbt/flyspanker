"""
Tools for summarizing FITS header metadata.
"""

import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, MaskedColumn


def summarize_fits_headers(directory, pattern="*.fit*", hdu=0):
    """
    Summarize header metadata for all FITS files in a directory into a table.

    Each row of the returned table corresponds to one FITS file.  Each column
    corresponds to a header keyword found in at least one of the files.  When a
    file does not contain a particular keyword, that cell is masked.

    Parameters
    ----------
    directory : str or path-like
        Directory to search for FITS files.
    pattern : str, optional
        Glob pattern used to match FITS files (default ``"*.fit*"`` matches
        ``*.fits``, ``*.fit``, ``*.fits.gz``, etc.).
    hdu : int or str, optional
        HDU index (or name) whose header is read (default ``0``, the primary
        HDU).

    Returns
    -------
    astropy.table.Table
        A masked table with one row per FITS file and one column per unique
        header keyword.  A special ``"filename"`` column always appears first
        and contains the file name (not the full path).

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    ValueError
        If no FITS files matching *pattern* are found in *directory*.

    Examples
    --------
    >>> from flyspanker.headers import summarize_fits_headers
    >>> t = summarize_fits_headers("/path/to/images")
    >>> t.pprint()
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    fits_files = sorted(directory.glob(pattern))
    if not fits_files:
        raise ValueError(
            f"No FITS files matching '{pattern}' found in {directory}"
        )

    # --- first pass: collect all headers -----------------------------------
    headers = []
    filenames = []
    for path in fits_files:
        try:
            with fits.open(path, memmap=False) as hdul:
                hdr = hdul[hdu].header
            # Convert to a plain dict, skipping blank / comment cards
            hdr_dict = {
                key: hdr[key]
                for key in hdr.keys()
                if key and key not in ("COMMENT", "HISTORY", "")
            }
            headers.append(hdr_dict)
            filenames.append(path.name)
        except Exception as exc:
            warnings.warn(f"Could not read {path.name}: {exc}", stacklevel=2)

    if not headers:
        raise ValueError("No FITS files could be read successfully.")

    # --- second pass: build unified column list (preserving encounter order) ---
    seen_keys = {}
    for hdr_dict in headers:
        for key in hdr_dict:
            seen_keys.setdefault(key, None)

    keywords = list(seen_keys.keys())

    # --- build the table ---------------------------------------------------
    # Start with the filename column
    rows = {kw: [] for kw in keywords}
    for hdr_dict in headers:
        for kw in keywords:
            rows[kw].append(hdr_dict.get(kw))

    table = Table()
    table["filename"] = filenames

    for kw in keywords:
        col_data = rows[kw]
        mask = [v is None for v in col_data]

        # Replace None with a suitable fill value based on data type
        non_null = [v for v in col_data if v is not None]
        if not non_null:
            # All masked – store as strings
            filled = [""] * len(col_data)
            dtype = str
        elif all(isinstance(v, bool) for v in non_null):
            filled = [v if v is not None else False for v in col_data]
            dtype = bool
        elif all(isinstance(v, (int, np.integer)) for v in non_null):
            filled = [v if v is not None else 0 for v in col_data]
            dtype = int
        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in non_null):
            filled = [v if v is not None else np.nan for v in col_data]
            dtype = float
        else:
            filled = [str(v) if v is not None else "" for v in col_data]
            dtype = str

        table[kw] = MaskedColumn(filled, mask=mask, dtype=dtype, name=kw)

    return table

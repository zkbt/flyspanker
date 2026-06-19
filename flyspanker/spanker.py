"""
Core Spanker class for interactive aperture photometry in Jupyter notebooks.

Usage
-----
In a Jupyter notebook cell::

    %matplotlib widget
    from flyspanker import Spanker
    s = Spanker("my_image.fits")

Then click on a star in the displayed image to place an aperture and read off
the centroid position and total flux.  Use the *Aperture radius* slider to
adjust the aperture size on the fly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com

try:
    import ipywidgets as widgets
    from IPython.display import display as ipython_display

    _HAS_WIDGETS = True
except ImportError:  # pragma: no cover
    _HAS_WIDGETS = False


class Spanker:
    """Interactive aperture photometry tool for a FITS image.

    Parameters
    ----------
    filename : str or path-like
        Path to the FITS file to display.
    ext : int, optional
        FITS extension index to read image data from (default 0).
    cmap : str, optional
        Matplotlib colormap name used for the image (default ``'gray'``).
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches (default ``(8, 8)``).

    Attributes
    ----------
    data : numpy.ndarray
        2-D image array loaded from the FITS file.
    radius : float
        Current aperture radius in pixels.
    auto_centroid : bool
        When ``True`` (default) the aperture is moved to the flux-weighted
        centroid after each click.  When ``False`` the aperture stays exactly
        where the user clicked.
    last_result : dict or None
        Dictionary with keys ``x``, ``y``, ``x_centroid``, ``y_centroid``,
        and ``flux`` from the most recent aperture measurement, or ``None``
        if no measurement has been made yet.
    """

    def __init__(
        self,
        filename: str | Path,
        ext: int = 0,
        cmap: str = "gray",
        figsize: tuple[float, float] = (8, 8),
    ) -> None:
        self.filename = Path(filename)
        self.ext = ext
        self.cmap = cmap
        self.figsize = figsize

        self.last_result: Optional[dict] = None
        self._aperture_patch: Optional[mpatches.Circle] = None

        # Whether to auto-centroid on each click
        self.auto_centroid: bool = True

        # Load data
        self.data, self.header = self._load_fits()

        # Default aperture radius (pixels)
        self.radius: float = 10.0

        # Build display
        self._setup_display()

    # ------------------------------------------------------------------
    # FITS loading
    # ------------------------------------------------------------------

    def _load_fits(self) -> tuple[np.ndarray, fits.Header]:
        """Return (data_2d, header) from the FITS file."""
        with fits.open(self.filename) as hdul:
            hdu = hdul[self.ext]
            data = hdu.data
            header = hdu.header

        if data is None or data.ndim == 0:
            raise ValueError(
                f"Extension {self.ext} of '{self.filename}' contains no image data."
            )

        # Collapse extra dimensions (e.g. data cubes) to 2-D
        while data.ndim > 2:
            data = data[0]

        return data.astype(float), header

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _setup_display(self) -> None:
        """Create the matplotlib figure and connect event handlers."""
        # Compute display limits with ZScale
        interval = ZScaleInterval()
        try:
            vmin, vmax = interval.get_limits(self.data)
        except Exception:
            vmin, vmax = np.nanpercentile(self.data, [1, 99])

        # Use plt.ioff() so that the ipympl backend does not auto-display the
        # figure inline.  We take control of display ourselves (via the HBox
        # below), which prevents a duplicate figure from appearing.
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_title(self.filename.name)
        self.im = self.ax.imshow(
            self.data,
            origin="lower",
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(self.im, ax=self.ax, label="Pixel value")
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        # Status text box (top of axes)
        self._status_text = self.ax.text(
            0.01,
            0.99,
            "Click on a star to measure flux.",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
        )

        self._cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        if _HAS_WIDGETS:
            self._build_widgets()
        else:
            warnings.warn(
                "ipywidgets is not available; the aperture size slider will not be shown.",
                stacklevel=2,
            )

        plt.tight_layout()

    def _build_widgets(self) -> None:
        """Create ipywidgets controls and display them to the right of the figure.

        When the ipympl backend is active the figure canvas is itself a widget,
        so we embed it together with the controls in an HBox.  In other backends
        (e.g. Agg during testing) the controls are displayed on their own.
        """
        ny, nx = self.data.shape
        max_radius = min(nx, ny) // 4

        self._radius_slider = widgets.FloatSlider(
            value=self.radius,
            min=1.0,
            max=float(max_radius),
            step=0.5,
            description="Radius (px):",
            style={"description_width": "initial"},
            layout=widgets.Layout(height="200px"),
            orientation="vertical",
        )
        self._radius_slider.observe(self._on_radius_change, names="value")

        self._centroid_checkbox = widgets.Checkbox(
            value=self.auto_centroid,
            description="auto-centroid?",
            style={"description_width": "initial"},
        )
        self._centroid_checkbox.observe(self._on_centroid_change, names="value")

        self._output_box = widgets.Output()

        controls = widgets.VBox(
            [self._radius_slider, self._centroid_checkbox, self._output_box],
            layout=widgets.Layout(padding="10px", align_items="center"),
        )

        # With the ipympl backend the canvas is a widget; place the figure
        # and controls side-by-side in an HBox.  In other backends fall back
        # to displaying only the controls panel (the figure is shown normally).
        if isinstance(self.fig.canvas, widgets.Widget):
            ipython_display(widgets.HBox([self.fig.canvas, controls]))
        else:
            ipython_display(controls)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        """Handle a mouse-click event on the axes.

        Aperture placement is skipped when the toolbar is in zoom or pan mode
        so that the user can navigate the image freely without accidentally
        triggering a measurement.
        """
        # Ignore clicks while the zoom or pan tool is active
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", "") != "":
            return

        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x_click = event.xdata
        y_click = event.ydata

        result = self.measure(x_click, y_click, self.radius, centroid=self.auto_centroid)
        self.last_result = result

        self._draw_aperture(result["x_centroid"], result["y_centroid"])
        self._update_status(result)

    def _on_radius_change(self, change) -> None:
        """Handle aperture-radius slider change."""
        self.radius = float(change["new"])
        if self.last_result is not None:
            result = self.measure(
                self.last_result["x_centroid"],
                self.last_result["y_centroid"],
                self.radius,
                centroid=self.auto_centroid,
            )
            self.last_result = result
            self._draw_aperture(result["x_centroid"], result["y_centroid"])
            self._update_status(result)

    def _on_centroid_change(self, change) -> None:
        """Handle auto-centroid checkbox change."""
        self.auto_centroid = bool(change["new"])
        if self.last_result is not None:
            result = self.measure(
                self.last_result["x"],
                self.last_result["y"],
                self.radius,
                centroid=self.auto_centroid,
            )
            self.last_result = result
            self._draw_aperture(result["x_centroid"], result["y_centroid"])
            self._update_status(result)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, x: float, y: float, radius: float, centroid: bool = True) -> dict:
        """Measure centroid and flux inside a circular aperture.

        Parameters
        ----------
        x, y : float
            Initial guess for the aperture centre (pixel coordinates,
            0-indexed, column/row).
        radius : float
            Aperture radius in pixels.
        centroid : bool, optional
            When ``True`` (default) the aperture centre is refined to the
            flux-weighted centroid of pixels within the aperture.  When
            ``False`` the aperture is placed exactly at ``(x, y)``.

        Returns
        -------
        dict
            Keys: ``x`` (input x), ``y`` (input y),
            ``x_centroid``, ``y_centroid``, ``flux``.
        """
        ny, nx = self.data.shape
        r = max(1.0, float(radius))

        if not centroid:
            # Place aperture exactly at the click position without centroiding
            x_centroid = float(np.clip(x, 0, nx - 1))
            y_centroid = float(np.clip(y, 0, ny - 1))
            aperture = CircularAperture((x_centroid, y_centroid), r=r)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phot_table = aperture_photometry(self.data, aperture)
            flux = float(phot_table["aperture_sum"][0])
            return {
                "x": x,
                "y": y,
                "x_centroid": x_centroid,
                "y_centroid": y_centroid,
                "flux": flux,
            }

        # Extract a bounding box around the click position for centroiding
        x0 = int(max(0, np.floor(x - r)))
        x1 = int(min(nx, np.ceil(x + r) + 1))
        y0 = int(max(0, np.floor(y - r)))
        y1 = int(min(ny, np.ceil(y + r) + 1))

        cutout = self.data[y0:y1, x0:x1]

        if cutout.size == 0 or np.all(~np.isfinite(cutout)):
            return {
                "x": x,
                "y": y,
                "x_centroid": x,
                "y_centroid": y,
                "flux": np.nan,
            }

        # Centroid (within the cutout, then convert back to full-image coords)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xc_cut, yc_cut = centroid_com(cutout)

        x_centroid = x0 + xc_cut
        y_centroid = y0 + yc_cut

        # Clamp centroid to image bounds
        x_centroid = float(np.clip(x_centroid, 0, nx - 1))
        y_centroid = float(np.clip(y_centroid, 0, ny - 1))

        # Aperture photometry
        aperture = CircularAperture((x_centroid, y_centroid), r=r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phot_table = aperture_photometry(self.data, aperture)
        flux = float(phot_table["aperture_sum"][0])

        return {
            "x": x,
            "y": y,
            "x_centroid": x_centroid,
            "y_centroid": y_centroid,
            "flux": flux,
        }

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _draw_aperture(self, x: float, y: float) -> None:
        """Draw (or redraw) the aperture circle on the image."""
        if self._aperture_patch is not None:
            self._aperture_patch.remove()

        self._aperture_patch = mpatches.Circle(
            (x, y),
            radius=self.radius,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
            linestyle="--",
        )
        self.ax.add_patch(self._aperture_patch)
        self.fig.canvas.draw_idle()

    def _update_status(self, result: dict) -> None:
        """Update the status text overlay and the widget output box."""
        msg = (
            f"x={result['x_centroid']:.2f}  y={result['y_centroid']:.2f}  "
            f"flux={result['flux']:.4g}  r={self.radius:.1f} px"
        )
        self._status_text.set_text(msg)
        self.fig.canvas.draw_idle()

        if _HAS_WIDGETS:
            self._output_box.clear_output(wait=True)
            with self._output_box:
                print(
                    f"Centroid : x = {result['x_centroid']:.3f} px, "
                    f"y = {result['y_centroid']:.3f} px"
                )
                print(f"Flux     : {result['flux']:.6g} (sum of pixel values in aperture)")
                print(f"Radius   : {self.radius:.1f} px")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Spanker(filename='{self.filename}', "
            f"shape={self.data.shape}, radius={self.radius})"
        )

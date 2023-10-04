from __future__ import annotations
import dLux
import jax.numpy as np
from jax import vmap

__all__ = ["Toliman"]


class Toliman(dLux.Instrument):
    """
    A pre-built dLux instrument object for the Toliman telescope.

    Attributes
    ----------
    optics : dLux.core.BaseOptics
        The optics object to be used in the instrument.
    source : dLux.sources.BaseSource
        The source object to be used in the instrument.

    Methods
    -------
    normalise()
    """

    optics: None
    source: None

    def __init__(self, optics, source):
        """
        Parameters
        ----------
        optics : dLux.core.BaseOptics
            The optics object to be used in the instrument.
        source : dLux.sources.BaseSource
            The source object to be used in the instrument.
        """
        self.optics = optics
        self.source = source
        super().__init__()

    def __getattr__(self, key):
        """
        Returns an attribute of the instrument given a key.

        Parameters
        ----------
        key : str
            The key of the attribute to be returned.
        """
        for attribute in self.__dict__.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        # if key in self.sources.keys():
        #     return self.sources[key]
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute " f"{key}."
        )

    def normalise(self):
        """
        Normalises the source flux to 1.
        """
        return self.set("source", self.source.normalise())

    def model(self):
        """
        Method to model the Instrument source through the optics, giving the PSF of the instrument.
        """
        return self.optics.model(self.source)

    def linear_jitter_model(
        self,
        magnitude: float,
        angle: float,
        n_psfs: int = 5,
        centre: tuple = (0, 0),
    ):
        """
        Returns a radially jittered PSF by summing a number of shifted PSFs along a straight line.

        Parameters
        ----------
        magnitude : float
            The magnitude, or length, of the jitter in pixels.
        angle : float, optional
            The angle of the jitter in radians, by default 0
        n_psfs : int, optional
            The number of PSFs to sum, by default 5
        centre : tuple, optional
            The centre of the jitter in arcseconds, by default (0,0)

        Returns
        -------
        np.ndarray
            The jittered PSF.
        """

        centre_and_model = lambda optics, source, x, y: optics.model(
            source.set(["x_position", "y_position"], [x, y])
        )
        vmap_prop = vmap(centre_and_model, in_axes=(None, None, 0, 0))
        pixel_scale = self.optics.psf_pixel_scale

        # converting to cartesian angular coordinates
        x = magnitude / 2 * np.cos(angle)
        y = magnitude / 2 * np.sin(angle)
        xs = pixel_scale * np.linspace(-x, x, n_psfs) + centre[0]  # arcseconds
        ys = pixel_scale * np.linspace(-y, y, n_psfs) + centre[1]  # arcseconds

        psfs = vmap_prop(self.optics, self.source, xs, ys)

        return psfs.sum(0) / n_psfs  # adding and renormalising

    def full_model(self):
        """
        "Oh pretty sure that was for modelling the diffraction spikes in the corners as well as the central psf
        for some computer frying multi-hundred gb covariance matrix calculations
        tbh you wouldnt even need that random number game just run that model on someones computer and
        watch it die" - L. Desdoigts, 2023.
        """
        return self.optics.full_model(self.source)

    def perturb(self, X, parameters):  # TODO : fix this
        """
        Under Construction.
        """
        for parameter, x in zip(parameters, X):
            perturbed_self = self.add(parameter, x)
        return perturbed_self

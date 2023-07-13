from __future__ import annotations
import dLux
import jax.numpy as np
from jax import vmap

__all__ = ["Toliman"]


class Toliman(dLux.instruments.BaseInstrument):
    """
    A pre-built dLux instrument object for the Toliman telescope.
    """
    source: None
    optics: None

    def __init__(self, optics, source):
        self.optics = optics
        self.source = source
        super().__init__()

    def __getattr__(self, key):
        for attribute in self.__dict__.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        # if key in self.sources.keys():
        #     return self.sources[key]
        raise AttributeError(f"{self.__class__.__name__} has no attribute "
                             f"{key}.")

    def normalise(self):
        return self.set('source', self.source.normalise())

    def model(self):
        return self.optics.model(self.source)

    def jitter_model(self, radius: float, angle: float, n_psfs: int = 5, centre: tuple = (0, 0)):
        """
            Returns a jittered PSF by summing a number of shifted PSFs.

            Parameters
            ----------
            radius : float
                The radius of the jitter in pixels.
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

        def centre_and_model(optics, source, x, y):
            """A function to set the source position and propagate through the optics"""
            src = source.set(['x_position', 'y_position'], [x, y])
            return optics.model(src)

        vmap_prop = vmap(centre_and_model, in_axes=(None, None, 0, 0))
        pixel_scale = self.optics.psf_pixel_scale

        # converting to cartesian angular coordinates
        x = radius / 2 * np.cos(angle)
        y = radius / 2 * np.sin(angle)
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
        for parameter, x in zip(parameters, X):
            perturbed_self = self.add(parameter, x)
        return perturbed_self

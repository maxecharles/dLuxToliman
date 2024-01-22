from __future__ import annotations
import dLux
import jax.numpy as np
from jax import vmap, Array
from dLux import BaseSource, BaseOpticalSystem


__all__ = ["Toliman", "JitteredToliman"]


class Toliman(dLux.Telescope):
    """
    A pre-built dLux telescope object for the Toliman telescope.

    Attributes
    ----------
    optics : dLux.core.BaseOptics
        The optics object to be used in the telescope.
    source : dLux.sources.BaseSource
        The source object to be used in the telescope.

    Methods
    -------
    normalise()
    """

    optics: BaseOpticalSystem
    source: BaseSource

    def __init__(self, optics, source):
        """
        Parameters
        ----------
        optics : dLux.core.BaseOptics
            The optical system to be used in the telescope.
        source : dLux.sources.BaseSource
            The source object to be used in the telescope.
        """
        self.optics = optics
        self.source = source
        super().__init__(optics, source)

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
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")

    def normalise(self):
        """
        Normalises the source flux to 1.
        """
        return self.set("source", self.source.normalise())

    def full_model(self):
        """
        "Oh pretty sure that was for modelling the diffraction spikes in the corners as well as the central psf
        for some computer frying multi-hundred gb covariance matrix calculations
        tbh you wouldnt even need that random number game just run that model on someones computer and
        watch it die" - L. Desdoigts, 2023.
        """
        # TODO may break in dLux 0.14
        return self.optics.full_model(self.source)

    def perturb(self, X, parameters):
        """
        Under Construction.
        """
        # TODO : fix this
        for parameter, x in zip(parameters, X):
            perturbed_self = self.add(parameter, x)
        return perturbed_self


class JitteredToliman(Toliman):
    """
    A child class of Toliman to facilitate the modelling of telescope jitter.
    This is done by modelling and summing offset PSFs - as a result, the compute
    time increases linearly with the number of PSFs. The jitter is assumed to be
    either a linear smear of simple harmonic vibration.
    """

    jitter_mag: Array | float
    jitter_angle: Array | float
    n_psfs: int
    jitter_shape: str

    def __init__(
        self,
        optics,
        source,
        jitter_mag: Array | float,
        jitter_angle: Array | float,
        n_psfs: int = 6,
        jitter_shape: str = "linear",
    ):
        """
        Parameters
        ----------
        optics : dLux.core.BaseOptics
            The optical system to be used in the telescope.
        source : dLux.sources.BaseSource
            The source object to be used in the telescope.
        jitter_mag : float
            The magnitude of the jitter in arcseconds.
        jitter_angle : float
            The angle of the jitter in degrees.
        n_psfs : int
            The number of PSFs to be modelled.
        jitter_shape : str
            The shape of the jitter. Either "linear" or "shm".
        """

        super().__init__(optics, source)  # calling super
        self.jitter_mag = np.array(jitter_mag, dtype=np.float64)
        self.jitter_angle = np.array(jitter_angle, dtype=np.float64)
        if n_psfs < 2:
            raise ValueError("n_psfs must be greater than 1.")
        self.n_psfs = n_psfs
        if jitter_shape not in ["linear", "shm"]:
            raise ValueError(
                "Only jitter_shape values of 'linear' or 'shm' are supported."
            )
        self.jitter_shape = jitter_shape

    @staticmethod
    def get_bounds(xs):
        """
        Helper method to grab the integration bounds for shm jitter_model.
        """
        return np.concatenate(
            (
                np.array(
                    [
                        xs[0],
                    ]
                ),
                np.array(xs[1:] + xs[:-1]) / 2,
                np.array(
                    [
                        xs[-1],
                    ]
                ),
            )
        )

    @staticmethod
    def machine_epsilon(dtype):
        """
        Method to fetch machine epsilon for a given dtype (e.g. float32, float64).

        Parameters
        ----------
        dtype : np.dtype
            The dtype to fetch the machine epsilon for.
        """
        info = np.finfo(dtype)
        return info.eps

    def inv_shm(self, x):
        """
        Inverse function for the simple harmonic equation x(t) = Asin(t)
        """
        return np.arcsin(np.divide(x, self.jitter_mag / 2))

    def centre_and_model(self, x, y):
        """
        Function to offset the source and model the PSF. This is vmapped over
        in the jitter_model method.
        """
        recentered_tel = self.set(["x_position", "y_position"], [x, y])
        return recentered_tel.model()

    def jitter_model(self):
        """
        Method to model the jittered PSF. This is done by modelling and summing
        offset PSFs - as a result, the compute time increases linearly with the
        number of PSFs. The jitter is assumed to be either a linear smear of
        simple harmonic vibration. The shape is determined by the `jitter_shape`
        parameter.
        """
        # horizontal and vertical components of the jitter
        x = np.cos(dLux.utils.deg2rad(self.jitter_angle))
        y = np.sin(dLux.utils.deg2rad(self.jitter_angle))

        # grabbing machine epsilon to avoid function evaluation at the asymptote
        eps = self.machine_epsilon(self.jitter_mag.dtype)

        # generating the base array for PSF positions
        spacing = np.linspace(
            -self.jitter_mag / 2 + eps, self.jitter_mag / 2 - eps, self.n_psfs
        )

        # grabbing the x and y positions of the PSFs
        xs = spacing * x + self.x_position  # arcseconds
        ys = spacing * y + self.y_position  # arcseconds

        # for linear jitter
        if self.jitter_shape == "linear":
            weights = np.ones(self.n_psfs) / self.n_psfs  # equal weighting

        # for simple harmonic jitter
        if self.jitter_shape == "shm":
            bounds = self.get_bounds(spacing)  # grabbing bounds for integration
            # weighting by value of integral between bounds
            weights = self.inv_shm(bounds[:-1]) - self.inv_shm(bounds[1:])
            weights /= weights.sum()  # normalising

        # vmapping over all PSF positions
        psfs = vmap(self.centre_and_model, in_axes=(0, 0))(xs, ys)

        return np.tensordot(
            psfs, weights, axes=(0, 0)
        )  # multiplying by weights and summing

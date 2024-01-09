from __future__ import annotations
from abc import abstractmethod
import dLux
import dLux.utils as dlu
from jax import numpy as np
from jax import Array
from jax.scipy.stats import multivariate_normal

__all__ = ["GaussianJitter", "SHMJitter"]

PSF = lambda: dLux.PSFs.PSF
DetectorLayer = lambda: dLux.layers.detector_layers.DetectorLayer


class BaseJitter(DetectorLayer()):
    """
    Base class for jitter layers.
    """

    kernel_size: int

    def __init__(self: DetectorLayer, kernel_size: int):
        """
        Constructor for the BaseJitter class.

        Parameters
        ----------
        kernel_size : odd
            The size of the convolution kernel in pixels to use.
        """
        self.kernel_size = kernel_size
        super().__init__()

    def apply(self: DetectorLayer, psf: PSF()) -> PSF():
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """
        kernel = self.generate_kernel(dLux.utils.rad2arcsec(psf.pixel_scale))

        return psf.convolve(kernel)

    @abstractmethod
    def generate_kernel(self, pixel_scale: float) -> Array:
        pass


class GaussianJitter(BaseJitter):
    """
    Convolves the image with a Gaussian kernel parameterised by the standard
    deviation (sigma).

    Attributes
    ----------
    kernel_size : int
        The size in pixels of the convolution kernel to use.
    r : float, arcseconds
        The magnitude of the jitter.
    shear : float
        The shear of the jitter.
    phi : float, degrees
        The angle of the jitter.
    kernel_size : int, odd
        The size of the convolution kernel in pixels to use.
    kernel_oversample : int
        The oversampling factor for the kernel generation.
    """

    r: float
    shear: float = None
    phi: float = None
    kernel_oversample: int

    def __init__(
        self: BaseJitter,
        r: float,
        shear: float = 0,
        phi: float = 0,
        kernel_size: int = 11,
        kernel_oversample: int = 1,
    ):
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        r : float
            The jitter magnitude, defined as the determinant of the covariance
            matrix of the multivariate Gaussian kernel. This is the product of the
            standard deviations of the minor and major axes of the kernel, given in
            arcseconds.
        shear : float, [0, 1)
            A measure of how asymmetric the jitter is. Defined as one minus the ratio between
            the standard deviations of the minor/major axes of the multivariate
            Gaussian kernel. It must lie on the interval [0, 1). A shear of 0
            corresponds to a symmetric jitter, while as shear approaches one the
            jitter kernel becomes more linear.
        phi : float
            The angle of the jitter in degrees.
        kernel_size : int = 10
            The size of the convolution kernel in pixels to use.
        """

        # checking for odd kernel size
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer")

        # checking shear is valid
        if shear >= 1 or shear < 0:
            raise ValueError("shear must lie on the interval [0, 1)")

        super().__init__(kernel_size)

        self.r = r
        self.shear = shear
        self.phi = phi
        self.kernel_oversample = kernel_oversample

    @property
    def covariance_matrix(self):
        """
        Generates the covariance matrix for the multivariate normal distribution.

        Returns
        -------
        covariance_matrix : Array
            The covariance matrix.
        """
        # Compute the rotation angle
        rot_angle = np.radians(self.phi)

        # Construct the rotation matrix
        R = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )

        # calculating the eigenvalues (lambda1 > lambda2)
        lambda1 = (self.r / (1 - self.shear)) ** 0.25
        lambda2 = lambda1 * (1 - self.shear)

        # Construct the skew matrix
        base_matrix = np.array(
            [
                [lambda1**2, 0],
                [0, lambda2**2],
            ]
        )

        # Compute the covariance matrix
        covariance_matrix = np.dot(
            np.dot(R, base_matrix), R.T
        )
        
        return covariance_matrix

    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        Generates the normalised multivariate Gaussian kernel.

        Parameters
        ----------
        pixel_scale : float
            The pixel scale of the image in arcseconds per pixel.

        Returns
        -------
        kernel : Array
            The normalised Gaussian kernel.
        """
        # Generate distribution
        extent = pixel_scale * self.kernel_size  # kernel size in arcseconds
        x = (
            np.linspace(0, extent, self.kernel_oversample * self.kernel_size)
            - 0.5 * extent
        )
        xs, ys = np.meshgrid(x, x)
        pos = np.dstack((xs, ys))

        kernel = dlu.downsample(
            multivariate_normal.pdf(
                pos, mean=np.array([0.0, 0.0]), cov=self.covariance_matrix
            ),
            self.kernel_oversample,
        )

        return kernel / np.sum(kernel)


class SHMJitter(BaseJitter):
    """
    A class to simulate the effect of a one-dimensional simple harmonic jitter.
    This would be used in the case that the jitter is dominated by a single
    high-frequency vibration.
    """

    A: float
    phi: float

    def __init__(
        self: BaseJitter,
        A: float,  # amplitude in arcseconds
        phi: float,  # angle
        kernel_size: int = 11,
    ):
        """
        Constructor for the ApplySHMJitter class.

        Parameters
        ----------
        A : float, arcseconds
            The amplitude of the oscillation.
        phi : float, deg
            The angle of the jitter in degrees.
        kernel_size : int = 9
            The size of the convolution kernel in pixels to use. It is important for
            simple harmonic jitter that the kernel size is large enough to capture
            the full extent of the jitter. The optimal_kernel_size method will return
            a value for kernel_size that it recommended, or at least is a lower bound
            on usable values.
        """

        # setting parameters
        if A < 0.:
            raise ValueError("A must be positive")

        self.A = A
        self.phi = phi

        # calling parent constructor
        super().__init__(kernel_size)

    def optimal_kernel_size(self, pixel_scale: float) -> int:
        """
        Calculates the recommended kernel size in pixels for a given pixel scale.

        Parameters
        ----------
        pixel_scale : float
            The pixel scale of the image in arcseconds per pixel.

        Returns
        -------
        kernel_size : int
            The kernel size in pixels.
        """
        kernel_size = np.ceil(2 * self.A / pixel_scale).astype(int)
        kernel_size += 1 - kernel_size % 2  # ensuring odd integer

        return kernel_size

    @staticmethod
    def machine_epsilon(dtype):
        """
        Method to fetch machine epsilon for a given dtype (e.g. float32, float64).
        """
        info = np.finfo(dtype)
        return info.eps

    def CDF(self, x):
        """
        The cumulative distribution function for the simple harmonic motion.

        Parameters
        ----------
        x : Array
            The input values.
        eps : float
            A optional small offset to avoid divide by zero errors.
        """
        # TODO add compatibility for self.A = 0
        return np.arcsin(dlu.nandiv(x, self.A))

    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        Generates the normalised multivariate Gaussian kernel.

        Parameters
        ----------
        pixel_scale : float
            The pixel scale of the image in arcseconds per pixel.

        Returns
        -------
        kernel : Array
            The normalised convolution kernel.
        """

        # coordinates of pixel edges in one dimension
        pixel_edges = dlu.nd_coords(
            npixels=self.kernel_size + 1, pixel_scales=pixel_scale
        )

        # used to offset to avoid gradient evaluation at discontinuous boundary
        eps = self.machine_epsilon(pixel_edges.dtype)

        # replacing all pixels boundary values outside the oscillation domain with -A or A
        pixel_edges = np.clip(pixel_edges, -self.A + eps, self.A - eps)

        # calculating the fluxes of the pixels in one dimension (ignoring the factor of pi)
        fluxes = self.CDF(pixel_edges[1:]) - self.CDF(pixel_edges[:-1])

        # a padding of zeros for extrapolating to the full 2D kernel
        zero_pad = np.zeros(shape=(self.kernel_size // 2, self.kernel_size))

        # padding and rotating kernel
        kernel = dlu.rotate(
            np.vstack((zero_pad, fluxes, zero_pad)), np.radians(self.phi)
        )

        return kernel / np.sum(kernel)  # returning normalised kernel

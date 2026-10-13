from __future__ import annotations
from abc import abstractmethod
import dLux
import dLux.utils as dlu
from jax import numpy as np
from jax import Array

__all__ = ["GaussianJitter", "SHMJitter"]

PSF = lambda: dLux.PSFs.PSF
DetectorLayer = lambda: dLux.layers.detector_layers.DetectorLayer


class BaseJitter(DetectorLayer()):
    """
    Base class for jitter layers. Apply method applies a convolution.
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
        psf : PSF
            The PSF object to operate on.

        Returns
        -------
        psf : PSF
            The convolved PSF object.
        """
        kernel = self.generate_kernel(dLux.utils.rad2arcsec(psf.pixel_scale))

        return psf.convolve(kernel)

    @abstractmethod
    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        Generates the convolution kernel to be applied.
        """
        pass


class GaussianJitter(BaseJitter):
    """
    Convolves the image with a multivariate Gaussian kernel parameterised by the
    magnitude, shear and angle.

    Attributes
    ----------
    kernel_size : int
        The size in pixels of the convolution kernel to use.
    r : float, 1e-6 arcseconds^4
        The magnitude of the jitter, (sigma_major * sigma_minor)^2.
    shear : float
        The shear of the jitter.
    phi : float, degrees
        The angle of the jitter.
    kernel_size : int, odd
        The size of the convolution kernel in pixels to use.
    kernel_oversample : int
        Unused; the kernel is built in Fourier space and needs no oversampling.
        Kept so that existing code passing it still runs.
    """

    r: float | Array
    shear: float | Array = None
    phi: float | Array = None
    kernel_oversample: int

    def __init__(
        self: BaseJitter,
        r: float | Array,
        shear: float | Array = 0,
        phi: float | Array = 0,
        kernel_size: int = 11,
        kernel_oversample: int = 1,
    ):
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        r : float
            The jitter magnitude, defined as the determinant of the covariance
            matrix of the multivariate Gaussian kernel. This is the square of the
            product of the standard deviations of the minor and major axes of the
            kernel, (sigma_major * sigma_minor)^2, in units of 1e-6 arcseconds^4.
            A symmetric kernel has variance 1e-3 * sqrt(r) arcseconds^2.
        shear : float, [0, 1)
            A measure of how asymmetric the jitter is. Defined as one minus the ratio between
            the standard deviations of the minor/major axes of the multivariate
            Gaussian kernel. It must lie on the interval [0, 1). A shear of 0
            corresponds to a symmetric jitter, while as shear approaches one the
            jitter kernel becomes more linear.
        phi : float
            The angle of the jitter in degrees.
        kernel_size : int = 11
            The PSF is zero-padded by kernel_size // 2 pixels before the blur, and
            generate_kernel returns a kernel of this size. It should span at least
            ~8 standard deviations of the major axis, so that light blurred past
            the image edge is not wrapped onto the opposite edge.
        kernel_oversample : int = 1
            Unused; kept so that existing code passing it still runs.
        """

        # checking for odd kernel size
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer")

        # checking shear is valid
        if shear >= 1 or shear < 0:
            raise ValueError("shear must lie on the interval [0, 1)")

        super().__init__(kernel_size)

        self.r = np.array(r)
        self.shear = np.array(shear)
        self.phi = np.array(phi)
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

        # calculating the variances (var1 > var2)
        var1 = 1e-3 * np.sqrt(self.r) / (1 - self.shear)
        var2 = var1 * (1 - self.shear) ** 2

        # Construct the skew matrix
        base_matrix = np.array(
            [
                [var1, 0],
                [0, var2],
            ]
        )

        # Compute the covariance matrix
        covariance_matrix = np.dot(np.dot(R, base_matrix), R.T)

        return covariance_matrix

    def transfer_function(self, fx: Array, fy: Array) -> Array:
        """
        The Fourier transform of the Gaussian kernel, exp(-2 pi^2 f^T C f).

        Parameters
        ----------
        fx : Array
            Spatial frequencies along x (columns), in cycles per arcsecond.
        fy : Array
            Spatial frequencies along y (rows), in cycles per arcsecond.

        Returns
        -------
        transfer : Array
            The transfer function evaluated at (fx, fy).
        """
        cov = self.covariance_matrix
        quad_form = cov[0, 0] * fx**2 + 2 * cov[0, 1] * fx * fy + cov[1, 1] * fy**2
        return np.exp(-2 * np.pi**2 * quad_form)

    def apply(self: DetectorLayer, psf: PSF()) -> PSF():
        """
        Blurs the PSF by multiplying its Fourier transform by the Gaussian's.

        This is exact for a well-sampled PSF at any jitter width. A kernel sampled
        on the pixel grid is not: once the jitter is narrower than about a pixel it
        puts almost all of its weight in the central pixel and drops most of the
        blur, worst along the minor axis of a sheared kernel. The PSF is
        zero-padded by kernel_size // 2 pixels on each side first, so the edges
        behave like a convolution with a kernel_size kernel.

        Parameters
        ----------
        psf : PSF
            The PSF object to operate on.

        Returns
        -------
        psf : PSF
            The blurred PSF object.
        """
        pixel_scale = dLux.utils.rad2arcsec(psf.pixel_scale)
        pad = self.kernel_size // 2
        data = np.pad(psf.data, pad)

        # spatial frequencies of the padded image, cycles per arcsecond
        fy = np.fft.fftfreq(data.shape[0], d=pixel_scale)
        fx = np.fft.fftfreq(data.shape[1], d=pixel_scale)
        fx, fy = np.meshgrid(fx, fy)  # x along columns, y along rows

        blurred = np.fft.ifft2(np.fft.fft2(data) * self.transfer_function(fx, fy)).real
        ny, nx = psf.data.shape
        return psf.set("data", blurred[pad : pad + ny, pad : pad + nx])

    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        The blur that `apply` performs, as a normalised kernel_size x kernel_size
        kernel, e.g. for plotting. It is computed on a grid four times larger and
        cropped, so that it matches `apply`. Below about one pixel it has small
        negative side lobes, as any band-limited blur does.

        Parameters
        ----------
        pixel_scale : float
            The pixel scale of the image in arcseconds per pixel.

        Returns
        -------
        kernel : Array
            The normalised Gaussian kernel.
        """
        n = 4 * self.kernel_size
        freqs = np.fft.fftfreq(n, d=pixel_scale)  # cycles per arcsecond
        fx, fy = np.meshgrid(freqs, freqs)  # x along columns, y along rows

        # back to real space with zero lag at the centre, then crop
        kernel = np.fft.fftshift(np.fft.ifft2(self.transfer_function(fx, fy)).real)
        c, h = n // 2, self.kernel_size // 2
        kernel = kernel[c - h : c + h + 1, c - h : c + h + 1]

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
        if A < 0.0:
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

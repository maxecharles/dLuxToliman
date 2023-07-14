from __future__ import annotations
import jax.numpy as np
import dLux.utils as dlu
from jax import Array, vmap
import dLux
import os

MixedAlphaCen = lambda: dLuxToliman.sources.MixedAlphaCen

__all__ = ["TolimanOptics"]

OpticalLayer = lambda: dLux.optical_layers.OpticalLayer
AngularOptics = lambda: dLux.optics.AngularOptics


class TolimanOptics(AngularOptics()):

    def __init__(self,

                 wf_npixels: int = 256,
                 psf_npixels: int = 256,
                 psf_oversample: int = 2,
                 psf_pixel_scale: float = 0.375,  # arcsec

                 mask: Array = None,

                 radial_orders: Array = None,
                 noll_indices: Array = None,
                 coefficients: Array = None,

                 m1_diameter: float = 0.125,
                 m2_diameter: float = 0.032,

                 n_struts: int = 3,
                 strut_width: float = 0.002,
                 strut_rotation: float = -np.pi / 2,

                 ) -> TolimanOptics:
        """
        A pre-built dLux optics layer of the Toliman optical system. Note TolimanOptics uses units of arcseconds.

        Parameters
        ----------
        wf_npixels : int
            The pixel width the wavefront layer.
        psf_npixels : int
            The pixel width of the PSF.
        psf_oversample : int
            The Nyquist oversampling factor of the PSF.
        psf_pixel_scale : float
            The pixel scale of the PSF in arcseconds per pixel.
        mask : Array
            The diffractive mask array to apply to the wavefront layer.
        radial_orders : Array = None
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indices argument should be used instead.
        noll_indices : Array
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus.
        coefficients : Array
            The coefficients of the Zernike polynomials.
        m1_diameter : float
            The outer diameter of the primary mirror in metres.
        m2_diameter : float
            The diameter of the secondary mirror in metres.
        n_struts : int
            The number of uniformly spaced struts holding the secondary mirror.
        strut_width : float
            The width of the struts in metres.
        strut_rotation : float
            The angular rotation of the struts in radians.
        """

        # Diameter
        diameter = m1_diameter

        # Generate Aperture
        aperture = dLux.apertures.ApertureFactory(
            npixels=wf_npixels,
            radial_orders=radial_orders,
            noll_indices=noll_indices,
            coefficients=coefficients,
            secondary_ratio=m2_diameter / m1_diameter,
            nstruts=n_struts,
            strut_ratio=strut_width / m1_diameter,
            strut_rotation=strut_rotation,
        )

        # Generate Mask
        if mask is None:
            path = os.path.join(os.path.dirname(__file__), "diffractive_pupil.npy")
            mask = dlu.scale_array(np.load(path), wf_npixels, order=1)

            # Enforce full binary
            mask = mask.at[np.where(mask <= 0.5)].set(0.)
            mask = mask.at[np.where(mask > 0.5)].set(1.)

            # Enforce full binary
            mask = dlu.phase_to_opd(mask * np.pi, 585e-9)

        # Propagator Properties
        psf_npixels = int(psf_npixels)
        psf_oversample = float(psf_oversample)
        psf_pixel_scale = float(psf_pixel_scale)

        super().__init__(wf_npixels=wf_npixels, diameter=diameter,
                         aperture=aperture, mask=mask, psf_npixels=psf_npixels,
                         psf_oversample=psf_oversample, psf_pixel_scale=psf_pixel_scale)

    def _apply_aperture(self, wavelength, offset):
        """
        Overwrite so mask can be stored as array
        """
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.aperture
        wf = wf.normalise()
        wf += self.mask
        return wf


class TolimanSpikes(TolimanOptics):
    """
    A pre-built dLux optics layer of the Toliman optical system with diffraction spikes.

    Attributes
    ----------
    grating_depth : float
        The depth of the grating in nanometres.
    grating_period : float
        The period of the grating in microns.
    spike_npixels : int
        The pixel width of the diffraction spikes.
    """

    grating_depth: float
    grating_period: float
    spike_npixels: int

    def __init__(self,

                 wf_npixels=256,
                 psf_npixels=256,
                 psf_oversample=2,
                 psf_pixel_scale=0.375,  # arcsec
                 spike_npixels=512,

                 mask=None,
                 
                 radial_orders: Array = None,
                 noll_indices: Array = None,
                 coefficients: Array = None,

                 m1_diameter: float = 0.125,
                 m2_diameter: float = 0.032,

                 n_struts=3,
                 strut_width=0.002,
                 strut_rotation=-np.pi / 2,

                 grating_depth=100.,  # nm
                 grating_period=300,  # um

                 ) -> TolimanOptics:
        """
        A pre-built dLux optics layer of the Toliman optical system with diffraction spikes.

        Parameters
        ----------
        wf_npixels : int
            The pixel width the wavefront layer.
        psf_npixels : int
            The pixel width of the PSF.
        psf_oversample : int
            The Nyquist oversampling factor of the PSF.
        psf_pixel_scale : float
            The pixel scale of the PSF in arcseconds per pixel.
        mask : Array
            The diffractive mask array to apply to the wavefront layer.
        radial_orders : Array = None
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indices argument should be used instead.
        noll_indices : Array
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus.
        coefficients : Array
            The coefficients of the Zernike polynomials.
        m1_diameter : float
            The outer diameter of the primary mirror in metres.
        m2_diameter : float
            The diameter of the secondary mirror in metres.
        n_struts : int
            The number of uniformly spaced struts holding the secondary mirror.
        strut_width : float
            The width of the struts in metres.
        strut_rotation : float
            The angular rotation of the struts in radians.
        """

        # Diameter
        self.grating_depth = grating_depth
        self.grating_period = grating_period
        self.spike_npixels = spike_npixels

        super().__init__(
            wf_npixels=wf_npixels,
            psf_npixels=psf_npixels,
            psf_oversample=psf_oversample,
            psf_pixel_scale=psf_pixel_scale,
            mask=mask,
            radial_orders=radial_orders,
            noll_indices=noll_indices,
            coefficients=coefficients,
            m1_diameter=m1_diameter,
            m2_diameter=m2_diameter,
            n_struts=n_struts,
            strut_width=strut_width,
            strut_rotation=strut_rotation
        )

    def model_spike(self, wavelengths, offset, weights, angles, sign, center):
        """
        Model a Toliman diffraction spike.
        """
        propagator = vmap(self.model_spike_mono, (0, None, 0, None, None))
        psfs = propagator(wavelengths, offset, angles, sign, center)
        psfs *= weights[..., None, None]
        return psfs.sum(0)

    def model_spike_mono(self, wavelength, offset, angle, sign, centre):
        """
        Model a monochromatic Toliman diffraction spike.
        "Yeah most of that code was hacked together trying to not burn my legs running it." - L. Desdoigts, 2023.

        Parameters
        ----------
        wavelength : float
            The wavelength of the monochromatic PSF.
        offset : float
            Stellar positional offset from the optical axis in arcseconds. TODO CHECK
        angle : float
            The diffraction angle in radians between the spike and the star, determined by the grating period.
        sign : tuple
            Determine which corner of the detector the spike is in. E.g. [-1, 1].
        centre : tuple
            The central location of the diffraction spike in pixels.
        """

        # Construct and tilt
        wf = dLux.wavefronts.Wavefront(self.aperture.shape[-1], self.diameter, wavelength)

        # Addd offset and tilt
        wf = wf.tilt_wavefront(offset - sign * angle)

        # Apply aperture and normalise
        wf *= self.aperture
        wf = wf.normalise()

        # Apply aberrations
        wf *= self.aberrations

        # Propagate
        shift = sign * centre
        true_pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale = dlu.arcseconds_to_radians(true_pixel_scale)
        wf = wf.shifted_MFT(self.spike_npixels, pixel_scale, shift=shift)

        # Return PSF
        return wf.psf

    def get_diffraction_angles(self, wavelengths):
        """
        Method to get the diffraction angles for a given wavelength set.
        """
        period = self.grating_period * 1e-6  # Convert to meters
        angles = np.arcsin(wavelengths / period) / np.sqrt(2)  # Radians
        return dlu.radians_to_arcseconds(angles)

    def model_spikes(self, wavelengths, offset, weights):
        """
        
        """
        # Get center shift values
        period = self.grating_period * 1e-6  # Convert to meters
        angles = np.arcsin(wavelengths / period) / np.sqrt(2)  # Radians
        # angles = get_diffraction_angles(wavelengths)
        true_pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale = dlu.arcseconds_to_radians(true_pixel_scale)
        center = angles.mean(0) // pixel_scale

        # Model
        signs = np.array([[-1, +1], [+1, +1], [-1, -1], [+1, -1]])
        propagator = vmap(self.model_spike, (None, None, None, None, 0, None))
        return propagator(wavelengths, offset, weights, angles, signs, center)

    def full_model(self, source, cent_nwavels=5):
        """
        Returns the diffraction spikes of the PSF

        source should be an MixedAplhaCen object
        """
        if not isinstance(source, MixedAlphaCen()):
            raise TypeError("source must be a MixedAlphaCen object")

        # Get Values
        wavelengths = source.wavelengths
        weights = source.norm_weights
        fluxes = source.raw_fluxes
        positions = source.xy_positions
        fratio = source.mixing

        # Calculate relative fluxes
        # TODO: Translate grating depth to central vs corner flux
        # Requires some experimental  mathematics
        # Probably requires both period and depth
        central_flux = 0.8
        corner_flux = 0.2

        # Model Central
        # TODO: Downsample central wavelengths and weights
        central_wavelegths = wavelengths
        central_weights = weights
        propagator = vmap(self.propagate, in_axes=(None, 0, 0))
        central_psfs = propagator(
            central_wavelegths,
            positions,
            central_weights)
        central_psfs *= central_flux * fluxes[:, None, None]

        # Model spikes
        propagator = vmap(self.model_spikes, in_axes=(None, 0, 0))
        spikes = propagator(wavelengths, positions, weights)
        spikes *= corner_flux * fluxes[:, None, None, None] / 4

        # Return
        return central_psfs.sum(0), spikes.sum(0)

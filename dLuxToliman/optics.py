from __future__ import annotations
import jax.numpy as np
import jax.random as jr
import dLux.utils as dlu
from jax import Array, vmap
import dLux


Optics = lambda : dLux.core.BaseOptics
MixedAlphaCen = lambda : dLux.models.MixedAlphaCen

__all__ = ["TolimanOptics"]

OpticalLayer = lambda : dLux.optical_layers.OpticalLayer
AngularOptics = lambda : dLux.optics.AngularOptics


class TolimanOptics(AngularOptics()):
    """
    A model of the Toliman optical system.

    Its default parameters are:

    """

    def __init__(self, 

        wf_npixels = 256,
        psf_npixels = 256,
        psf_oversample = 2,
        psf_pixel_scale = 0.375, # arcsec

        mask = None,

        radial_orders    : Array = None,
        noll_indices     : Array = None,
        coefficients = None,
        # amplitude : float = 0.,
        # seed : int = 0,

        m1_diameter = 0.125,
        m2_diameter = 0.032,
        
        nstruts = 3,
        strut_width = 0.002,
        strut_rotation=-np.pi/2

        ) -> TolimanOptics:
        """
        Constructs a simple model of the Toliman Optical Systems

        In this class units are different:
        - psf_pixel_scale is in unit of arcseconds
        """

        # Diameter
        diameter = m1_diameter

        # Generate Aperture
        aperture = dLux.apertures.ApertureFactory(
            npixels         = wf_npixels,
            radial_orders   = radial_orders,
            noll_indices    = noll_indices,
            coefficients    = coefficients,
            secondary_ratio = m2_diameter/m1_diameter,
            nstruts         = nstruts,
            strut_ratio     = strut_width/m1_diameter)

        # Generate Mask
        if mask is None:
            import os
            path = "./dLuxToliman/diffractive_pupil.npy"
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
    A model of the Toliman optical system.

    Its default parameters are:

    """
    grating_depth  : float
    grating_period : float
    spike_npixels  : int

    def __init__(self, 

        wf_npixels = 256,
        psf_npixels = 256,
        psf_oversample = 2,
        psf_pixel_scale = 0.375, # arcsec
        spike_npixels = 512,

        mask = None,
        zernikes = None,
        amplitude : float = 0.,
        seed : int = 0,

        m1_diameter = 0.13, # Double check this
        m2_diameter = 0.032,
        
        nstruts = 3,
        strut_width = 0.002,
        strut_rotation=-np.pi/2,

        grating_depth = 100., # nm
        grating_period = 300, # um

        ) -> TolimanOptics:
        """
        Constructs a simple model of the Toliman Optical Systems

        In this class units are different:
        - psf_pixel_scale is in unit of arcseconds
        grating depth is in nm
        grating period is in um
        """

        # Diameter
        self.grating_depth = grating_depth
        self.grating_period = grating_period
        self.spike_npixels = spike_npixels

        super().__init__(
            wf_npixels      = wf_npixels, 
            psf_npixels     = psf_npixels, 
            psf_oversample  = psf_oversample, 
            psf_pixel_scale = psf_pixel_scale, 
            mask            = mask, 
            zernikes        = zernikes, 
            amplitude       = amplitude, 
            seed            = seed, 
            m1_diameter     = m1_diameter, 
            m2_diameter     = m2_diameter, 
            nstruts         = nstruts, 
            strut_width     = strut_width, 
            strut_rotation  = strut_rotation
            )


    def model_spike(self, wavelengths, offset, weights, angles, sign, center):
        """
        
        """
        propagator = vmap(self.model_spike_mono, (0, None, 0, None, None))
        psfs = propagator(wavelengths, offset, angles, sign, center)
        psfs *= weights[..., None, None]
        return psfs.sum(0)


    def model_spike_mono(self, wavelength, offset, angle, sign, center):
        """
        
        """
        # Construct and tilt
        wf = dLux.wavefronts.Wavefront(self.aperture.shape[-1], self.diameter, 
            wavelength)

        # Addd offset and tilt
        wf = wf.tilt_wavefront(offset - sign * angle)

        # Apply aperture and normalise
        wf *= self.aperture
        wf = wf.normalise()

        # Apply aberrations
        wf *= self.aberrations

        # Propagate
        shift = sign * center
        true_pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale = dlu.arcseconds_to_radians(true_pixel_scale)
        wf = wf.shifted_MFT(self.spike_npixels, pixel_scale, shift=shift)

        # Return PSF
        return wf.psf


    def get_diffraction_angles(self, wavelenghts):
        """
        
        """
        period = self.grating_period * 1e-6 # Convert to meters
        angles = np.arcsin(wavelengths / period) / np.sqrt(2) # Radians
        return dlu.radians_to_arcseconds(angles)
    

    def model_spikes(self, wavelengths, offset, weights):
        """
        
        """
        # Get center shift values
        period = self.grating_period * 1e-6 # Convert to meters
        angles = np.arcsin(wavelengths / period) / np.sqrt(2) # Radians
        # angles = get_diffraction_angles(wavelengths)
        true_pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale = dlu.arcseconds_to_radians(true_pixel_scale)
        center = angles.mean(0)//pixel_scale

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
from __future__ import annotations
from jax import Array, vmap
import jax.numpy as np
from zodiax import filter_vmap
import dLux.utils as dlu
import dLux

Source = lambda: dLux.BaseSource
Optics = lambda: dLux.BaseOpticalSystem

__all__ = ["AlphaCen"]  # , "MixedAlphaCen"]


class AlphaCen(Source()):
    """
    A parametrised model of the Alpha Centauri binary pair.

    Properties
    ----------
    xy_positions : Array
        The x and y positions of the two stars in radians.
    raw_fluxes : Array
        The raw fluxes of the two stars in photons.
    norm_weights : Array
        The normalised spectral weights of each modelled wavelength.

    Methods
    -------
    normalise()
        Normalises the source flux to 1.
    model()
        Models the PSF by propagating the AlphaCen source through the given optics.
    """

    separation: float
    position_angle: float
    x_position: float
    y_position: float
    log_flux: float
    contrast: float
    bandpass: tuple
    weights: Array
    wavelengths: Array

    def __init__(
        self,
        n_wavels=5,
        separation=10.0,  # arcseconds
        position_angle=90,  # degrees
        x_position=0.0,  # arcseconds
        y_position=0.0,  # arcseconds
        log_flux=6.832,  # Photons
        contrast=3.37,
        bandpass=(530, 640),  # nm
        weights=None,
    ):
        """
        Parameters
        ----------
        n_wavels : int
            The number of wavelengths to model.
        separation : float
            The binary separation of the two stars in arcseconds.
        position_angle : float
            The position angle of the binary pair in degrees.
        x_position : float
            The horizontal offset of the image in arcseconds.
        y_position : float
            The vertical offset of the image in arcseconds.
        log_flux : float
            The log10 of the total number of photons in the image.
        contrast : float
            The flux ratio of Alpha Cen A / Alpha Cen B.
        bandpass : tuple
            The wavelength range of the image in nanometers, with syntax (min, max).
        weights : Array
            Spectral weights for each wavelength modelled. If None, the weights are uniform.
        """

        # Positional Parameters
        self.x_position = x_position
        self.y_position = y_position
        self.separation = separation
        self.position_angle = position_angle

        # Flux & Contrast
        self.log_flux = log_flux
        self.contrast = contrast

        # Spectrum (Uniform)  # TODO : Phoenix Models?
        self.bandpass = bandpass
        self.wavelengths = np.linspace(bandpass[0], bandpass[1], n_wavels)
        if weights is None:
            self.weights = np.ones((2, n_wavels)) / n_wavels
        else:
            self.weights = weights

    def normalise(self):
        """
        Normalises the source flux to 1.
        """
        return self.multiply("weights", 1 / self.weights.sum(1)[:, None])

    @property
    def xy_positions(self):
        """
        Returns the x and y positions of the two stars in units of radians.
        """
        # Calculate
        r = self.separation / 2
        phi = dlu.deg2rad(self.position_angle)
        sep_vec = np.array([r * np.sin(phi), r * np.cos(phi)])

        # Add to Position vectors
        pos_vec = np.array([self.x_position, self.y_position])
        output_vec = np.array([pos_vec + sep_vec, pos_vec - sep_vec])
        return dlu.arcsec2rad(output_vec)

    @property
    def raw_fluxes(self):
        """
        Returns the raw integrated fluxes of Alpha Centauri A and B respectively, in units of photons.
        """
        flux = (10**self.log_flux) / 2
        flux_A = 2 * self.contrast * flux / (1 + self.contrast)
        flux_B = 2 * flux / (1 + self.contrast)
        return np.array([flux_A, flux_B])

    @property
    def norm_weights(self):
        """
        Returns the normalised spectral weights of each modelled wavelength.
        """
        return self.weights / self.weights.sum(1)[:, None]

    def model(
        self: Source(),
        optics: Optics(),
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Models the PSF by propagating the AlphaCen source through the given optics.

        Parameters
        ----------
        optics : Optics
            The optics to propagate the source through.
        return_wf
            Whether or not to return the wavefront.
        return_psf
            Whether or not to return the PSF.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        """
        # Get Values
        weights = self.norm_weights
        fluxes = self.raw_fluxes
        positions = self.xy_positions
        wavelengths = 1e-9 * self.wavelengths

        # Model PSF
        input_weights = weights * fluxes[:, None]

        # Return wf case is simple
        prop_fn = lambda position, weight: optics.propagate(
            wavelengths, position, weight, return_wf, return_psf
        )
        output = filter_vmap(prop_fn)(positions, input_weights)

        # Return wf is simple case
        if return_wf:
            return output

        # Return psf just requires constructing object
        if return_psf:
            return dLux.PSF(output.data.sum(0), output.pixel_scale.mean())

        # Return array is simple
        return output.sum(0)


def get_mixed_alpha_cen_spectra(nwavels: int, bandpass: tuple = (530, 640)):
    """
    Under Construction. PySynPhot not currently listed in package requirements. TODO complete.
    Gets the spectra of Alpha Centauri A and B using PySynPhot phoenix models.

    Parameters
    ----------
    nwavels : int
        The number of wavelengths to sample.
    bandpass : tuple
        The wavelength range of the image in nanometers, with syntax (min, max).

    """

    # Importing PySynPhot here to prevent issues with Google colab install, for example
    print("Warning: Method is not fully implemented.")
    import pysynphot as S

    alpha_cen_a_spectrum: float = S.Icat(
        "phoenix",
        5790,  # Surface temp (K)
        0.2,  # Metalicity (Unit?)
        4.0,
    )  # Surface gravity (unit?)
    alpha_cen_a_spectrum.convert("flam")
    alpha_cen_a_spectrum.convert("m")

    alpha_cen_b_spectrum: float = S.Icat(
        "phoenix", 5260, 0.23, 4.37  # Surface temp (K)  # Metalicity (Unit?)
    )  # Surface gravity (unit?)
    alpha_cen_b_spectrum.convert("flam")
    alpha_cen_b_spectrum.convert("m")

    spot_spectrum: float = S.Icat(
        "phoenix", 4000, 0.23, 4.37  # Surface temp (K)  # Metalicity (Unit?)
    )  # Surface gravity (unit?)
    spot_spectrum.convert("flam")
    spot_spectrum.convert("m")

    # Full spectrum
    wavelengths = 1e-9 * np.linspace(bandpass[0], bandpass[1], nwavels)
    Aspec = alpha_cen_a_spectrum.sample(wavelengths)
    Bspec = alpha_cen_b_spectrum.sample(wavelengths)
    Sspec = spot_spectrum.sample(wavelengths)

    Aspec /= Aspec.max()
    Bspec /= Bspec.max()
    Sspec /= Sspec.max()

    return np.array([Aspec, Bspec, Sspec]), wavelengths


class MixedAlphaCen(AlphaCen):
    """
    A class representing an Alpha Centauri class with a mixed spectrum. Each spectrum is generated from the PySynPhot
    Phoenix model for a star of its properties plus one of the same properties but with a much lower temperature.
    This lower temperature spectrum represents the contribution of the star spots to the overall spectrum. The ratio
    of these two different spectra is determined by the 'mixing' parameter.

    Attributes
    ----------
    mixing : float
        The ratio of the spot spectrum to the star spectrum. A value of 0.0 means no spots, 1.0 means only spots.

    Methods
    -------
    model()
        Models the mixed Alpha Cen source through the given optics.
    """

    mixing: float

    def __init__(
        self,
        n_wavels=3,
        separation=10.0,  # arcseconds
        position_angle=90,  # degrees
        x_position=0.0,  # arcseconds
        y_position=0.0,  # arcseconds
        log_flux=6.832,  # Photons
        contrast=3.37,
        mixing=0.05,
        bandpass=(530, 640),  # nm
        weights=None,
    ):
        """ """
        self.mixing = mixing
        weights, wavelengths = get_mixed_alpha_cen_spectra(n_wavels)
        super().__init__(
            n_wavels,
            separation,
            position_angle,
            x_position,
            y_position,
            log_flux,
            contrast,
            bandpass,
            weights,
        )

    @property
    def norm_weights(self):
        """
        Returns the normalised spectral weights of each modelled wavelength.
        """
        # Get Spectra
        Spotted = self.mixing * self.weights[2]
        MixedA = (1 - self.mixing) * self.weights[0] + Spotted
        MixedB = (1 - self.mixing) * self.weights[1] + Spotted
        weights = np.array([MixedA, MixedB])

        # Normalise
        return weights / weights.sum(1)[:, None]

    def model(self: Source, optics: OpticalSystem) -> Array:
        """
        Models the PSF by propagating the AlphaCen source through the given optics.

        Parameters
        ----------
        optics : Optics
            The optics to propagate the source through.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        """

        # Get Values
        weights = self.norm_weights
        fluxes = self.raw_fluxes
        positions = self.xy_positions

        # vmap propagator
        source_propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        propagator = vmap(source_propagator, in_axes=(None, 0))

        # Model PSF
        input_weights = weights * fluxes[:, None]
        psfs = propagator(self.wavelengths, positions)
        psfs *= input_weights[..., None, None]
        return psfs.sum((0, 1))

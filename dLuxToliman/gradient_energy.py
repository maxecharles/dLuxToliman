import jax.numpy as np
from jax import Array
import dLux.utils as dlu

__all__ = ["get_GE", "get_RGE", "get_RWGE", "get_radial_mask"]


def get_GE(array: Array) -> Array:
    """
    Calculates the spatial gradient energy of the array.

    Parameters
    ----------
    array : Array
        The array to calculate the gradient energy for.

    Returns
    -------
    array : Array
        The array of gradient energies.
    """
    grads_vec = np.gradient(array)
    return np.hypot(grads_vec[0], grads_vec[1])


def get_RGE(
    array: Array, epsilon: float = 1e-8
) -> Array:  # TODO : Add epsilon
    """
    Calculates the spatial radial gradient energy of the array.

    Parameters
    ----------
    array : Array
        The array to calculate the radial gradient energy for.
    epsilon : float
        A small value added to the radial values to help with gradient
        stability.

    Returns
    -------
    array : Array
        The array of radial gradient energies.
    """
    npix = array.shape[0]
    positions = dlu.pixel_coords(npix, npix)
    grads_vec = np.gradient(array)

    xnorm = positions[1] * grads_vec[0]
    ynorm = positions[0] * grads_vec[1]

    return np.square(xnorm + ynorm)


def get_RWGE(array: Array, epsilon: float = 1e-8) -> Array:
    """
    Calculates the spatial radially weighted gradient energy of the array.

    Parameters
    ----------
    array : Array
        The array to calculate the radially weighted gradient energy for.
    epsilon : float
        A small value added to the radially weighted values to help with
        gradient stability.

    Returns
    -------
    array : Array
        The array of radial radially weighted energies.
    """
    npix = array.shape[0]
    positions = dlu.pixel_coords(npix, npix)
    radii = dlu.pixel_coords(npix, npix, polar=True)[0]
    radii_norm = positions / (radii + epsilon)
    grads_vec = np.gradient(array)

    xnorm = radii_norm[1] * grads_vec[0]
    ynorm = radii_norm[0] * grads_vec[1]

    return np.square(xnorm + ynorm)


def get_radial_mask(npixels: int, rmin: Array, rmax: Array) -> Array:
    """
    Calculates a binary radial mask, masking out radii below rmin, and above
    rmax.

    Parameters
    ----------
    npixels : int
        The linear size of the array.
    rmin : Array
        The inner radius to mask out.
    rmax : Array
        The outer radius to mask out.

    Returns
    -------
    mask: Array
        A mask with the values below rmin and above rmax masked out.
    """
    radii = dlu.pixel_coords(npixels, npixels, polar=True)[0]
    return np.asarray((radii < rmax) & (radii > rmin), dtype=float)

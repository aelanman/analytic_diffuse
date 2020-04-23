"""Analytic diffuse models with exact solutions."""

import numpy as np
from functools import wraps
import mpmath as mp

_funcnames = []
_projected_funcs = []


def checkinput(func):
    """Enforce input shapes and horizon cutoff."""
    global _funcnames

    _funcnames.append(func.__name__)
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check validity of input shapes
        phi = args[0]
        # Check whether input was scalar.
        scalar = np.isscalar(args[1])

        theta = args[1]

        if not scalar:
            theta = np.array(theta)
            if phi is not None:
                phi = np.array(phi)

        if scalar and theta > np.pi/2:
            return 0

        if phi is not None and not scalar and phi.shape != theta.shape:
            raise ValueError(
                "phi and theta must have the same shape: {}, {}".format(
                    str(phi.shape), str(theta.shape)
                )
            )
        result = func(phi, theta, *args[2:], **kwargs)

        # Set pixels outside the horizon to zero.
        if not scalar:
            result[theta > np.pi / 2] = 0

        # If a scalar was passed in, return a scalar.
        return result
    return wrapper


def projected(func):
    """Enforce input shapes and horizon cutoff."""
    global _projected_funcs

    _projected_funcs.append(func.__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        projected = kwargs.pop('projected', False)

        result = func(*args, **kwargs)

        if not projected:
            try:
                result *= np.cos(args[1])
            except AttributeError:
                result *= mp.cos(args[1])

        return result
    return wrapper


def _angle_to_lmn(phi, theta):
    phi, theta = np.atleast_1d(phi), np.atleast_1d(theta)
    lmn = np.zeros((len(phi), 3), dtype=float)
    lmn[:, 0] = np.sin(phi) * np.sin(theta)
    lmn[:, 1] = np.cos(phi) * np.sin(theta)
    lmn[:, 2] = np.cos(theta)
    return lmn


@checkinput
def monopole(phi, theta):
    """
    Trivially, 1 everywhere.

    Parameters
    ----------
    phi: ndarray of float
        Azimuth in radians, shape (Npixels,)
    theta: ndarray of float
        Zenith angle in radians, shape (Npixels,)
    """
    return np.ones_like(theta)


@projected
@checkinput
def cosza(phi, theta):
    """
    Cosine of zenith angle.

    Parameters
    ----------
    phi: ndarray of float
        Azimuth in radians, shape (Npixels,)
    theta: ndarray of float
        Zenith angle in radians, shape (Npixels,)
    """
    return np.ones_like(theta)


@projected
@checkinput
def polydome(phi, theta, n=2):
    """
    Polynomial function of zenith, weighted by cos(theta).

    Parameters
    ----------
    phi: ndarray of float
        Azimuth in radians, shape (Npixels,)
    theta: ndarray of float
        Zenith angle in radians, shape (Npixels,)
    n: int
        Order of polynomial (Default of 2)
        Must be even.
    """
    if n % 2 != 0:
        raise ValueError("Polynomial order must be even.")
    try:
        return 1 - np.sin(theta) ** n
    except AttributeError:
        return 1 - mp.sin(theta) ** n

@projected
@checkinput
def projgauss(phi, theta, a):
    """
    Gaussian weighted by cos(theta).

    Parameters
    ----------
    phi: ndarray of float
        Azimuth in radians, shape (Npixels,)
    theta: ndarray of float
        Zenith angle in radians, shape (Npixels,)
    a: float
        Gaussian width parameter.
    """
    try:
        return np.exp(-np.sin(theta)**2 / a**2)
    except AttributeError:
        return mp.exp(-mp.sin(theta) ** 2 / a ** 2)

@checkinput
def gauss(phi, theta, a, el0vec=None):
    """
    Gaussian, and off-zenith Gaussian.

    Parameters
    ----------
    phi: ndarray of float
        Azimuth in radians, shape (Npixels,)
    theta: ndarray of float
        Zenith angle in radians, shape (Npixels,)
    a: float
        Gaussian width parameter.
    el0vec: ndarray of float
        Cartesian vector describing lmn displacement of Gaussian center, shape (3,).
    """
    if phi is None:
        phi = np.zeros_like(theta)

    if el0vec is None:
        el0vec = np.array([0, 0, 0])

    sigsq = a**2 / (2 * np.pi)
    lmn = _angle_to_lmn(phi, theta)
    disp = (lmn - el0vec)[:, :2]  # Only l and m
    ang = np.linalg.norm(disp, axis=1)
    try:
        return np.exp(-ang**2 / (2 * sigsq))
    except AttributeError:
        return mp.exp(-ang ** 2 / (2 * sigsq))


@checkinput
def xysincs(phi, theta, a, xi=0.0):
    """
    Crossed sinc functions in l,m coordinates.

    Defined by:
        I(x,y) = sinc(ax) sinc(ay) cos(1 - x^2 - y^2), for |x| and |y| < 1/sqrt(2)
               = 0, otherwise
    where (x,y) is rotated from (l,m) by an angle xi.

    Parameters
    ----------
    phi: ndarray of float
        Azimuth in radians, shape (Npixels,)
    theta: ndarray of float
        Zenith angle in radians, shape (Npixels,)
    a: float
        Sinc parameter, giving width of the "box" in uv space.
        box width in UV is a/(2 pi)
    xi: float
        Rotation of (x,y) coordinates from (l,m) in radians.
    """
    cx, sx = np.cos(xi), np.sin(xi)
    rot = np.array([[cx, sx, 0], [-sx, cx, 0], [0, 0, 1]])

    lmn = _angle_to_lmn(phi, theta)
    xyvecs = np.dot(lmn, rot)

    result = np.sinc(a * xyvecs[:, 0] / np.pi) * np.sinc(a * xyvecs[:, 1] / np.pi) * np.cos(theta)
    outside = np.where((np.abs(xyvecs[:, 0]) > 1 / np.sqrt(2))
                       | (np.abs(xyvecs[:, 1]) > 1 / np.sqrt(2)))
    result[outside] = 0.0
    return result

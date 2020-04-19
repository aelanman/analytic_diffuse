"""Analytic diffuse models with exact solutions."""

import numpy as np
from functools import wraps


def checkinput(func):
    """Enforce input shapes and horizon cutoff."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check validity of input shapes
        phi = args[0]
        theta = args[1]
        if not phi.shape == theta.shape:
            raise ValueError(
                "phi and theta must have the same shape: {}, {}".format(
                    str(phi.shape), str(theta.shape)
                )
            )
        result = func(*args, **kwargs)
        # Set pixels outside the horizon to zero.
        result[theta > np.pi / 2] *= 0
        return result
    return wrapper


def _angle_to_lmn(phi, theta):
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
    return np.ones_like(phi)


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
    return np.cos(theta)


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
    if not n % 2 == 0:
        raise ValueError("Polynomial order must be even.")
    return np.cos(theta) * (1 - np.sin(theta)**n)


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
    return np.exp(-np.sin(theta)**2 / a**2) * np.cos(theta)


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
    if el0vec is None:
        el0vec = np.array([0, 0, 0])

    sigsq = a**2 / (2 * np.pi)
    lmn = _angle_to_lmn(phi, theta)
    disp = (lmn - el0vec)[:, :2]  # Only l an m
    ang = np.linalg.norm(disp, axis=1)
    return np.exp(-ang**2 / (2 * sigsq))


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

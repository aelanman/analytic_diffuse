"""Analytic diffuse models with exact solutions."""

import numpy as np
from functools import wraps
import mpmath as mp

sky_models = []
_projected_models = []   # A list of models whose functions define projected models
circular_models = []  # A list of models which are circularly symmetric.

def sky_model(func):
    """Enforce input shapes and horizon cutoff."""
    sky_models.append(func.__name__)

    param_doc = """Parameters
    ----------
    az: float or ndarray of float
        Azimuth in radians, shape (Npixels,)
    za: float or ndarray of float
        Zenith angle in radians, shape (Npixels,)"""

    other_doc = """Other Parameters
    ----------------
    projected: bool, optional
        Whether to return the projected sky model ``I / cos(za)``."""

    func.__doc__ = func.__doc__.format(params=param_doc, other=other_doc)
    func.__doc__ += """
    Returns
    -------
    float or np.ndarray
        The intensity of the sky model at the input co-ordinates. Same shape
        as ``za``."""

    @wraps(func)
    def wrapper(az, za, *args, projected: bool=False, **kwargs):
        # Check validity of input shapes
        # Check whether input was scalar.
        scalar = np.isscalar(za)

        if not scalar:
            za = np.array(za)
            if az is not None:
                az = np.array(az)

        if scalar and za > np.pi/2:
            return 0

        if not ((az is None or scalar) or az.shape == za.shape):
            raise ValueError(
                "az and za must have the same shape: {}, {}".format(
                    str(az.shape), str(za.shape)
                )
            )
        result = func(az, za, *args, **kwargs)

        # Set pixels outside the horizon to zero.
        if not scalar:
            result[za > np.pi / 2] = 0

        if func.__name__ in _projected_models and not projected:
            try:
                result *= np.cos(za)
            except AttributeError:
                result *= mp.cos(za)
        elif func.__name__ not in _projected_models and projected:
            try:
                result /= np.cos(za)
            except AttributeError:
                result /= mp.cos(za)

            # If a scalar was passed in, return a scalar.
        return result
    return wrapper


def projected(func):
    """Mark a model as defining itself in projected co-ordinates."""
    _projected_models.append(func.__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def circsym(func):
    """Mark a model as circularly (azimuthally) symmetric."""
    circular_models.append(func.__name__)

    @wraps(func)
    def wrapper(az, za, *args, **kwargs):
        return func(None, za, *args, **kwargs)
    return wrapper


def _angle_to_lmn(phi, theta):
    phi, theta = np.atleast_1d(phi), np.atleast_1d(theta)
    lmn = np.zeros((len(phi), 3), dtype=float)
    lmn[:, 0] = np.sin(phi) * np.sin(theta)
    lmn[:, 1] = np.cos(phi) * np.sin(theta)
    lmn[:, 2] = np.cos(theta)
    return lmn


@circsym
@sky_model
def monopole(az: [None, float, np.ndarray], za: [float, np.ndarray]):
    """
    Trivially, 1 everywhere.

    {params}

    {other}
    """
    return np.ones_like(za)


@circsym
@projected
@sky_model
def cosza(az: [None, float, np.ndarray], za: [float, np.ndarray]):
    """
    Cosine of zenith angle.

    {params}

    {other}
    """
    return np.ones_like(za)


@circsym
@projected
@sky_model
def polydome(az: [None, float, np.ndarray], za: [float, np.ndarray], n: int=2):
    """
    Polynomial function of zenith, weighted by cos(theta).

    {params}
    n: int
        Order of polynomial (Default of 2)
        Must be even.

    {other}
    """
    if n % 2:
        raise ValueError("Polynomial order must be even.")
    try:
        return 1 - np.sin(za) ** n
    except AttributeError:
        return 1 - mp.sin(za) ** n

@circsym
@projected
@sky_model
def projgauss(az: [None, float, np.ndarray], za: [float, np.ndarray], a: float):
    """
    Gaussian weighted by cos(theta).

    {params}
    a: float
        Gaussian width parameter.

    {other}

    Notes
    -----
    If called with `projected=True`, output is very similar to :func:`gauss` with
    `projected=False`.
    """
    try:
        return np.exp(-np.sin(za)**2 / a**2)
    except AttributeError:
        return mp.exp(-mp.sin(za) ** 2 / a ** 2)


@circsym
@sky_model
def gauss_zenith(az: [None, float, np.ndarray], za: [float, np.ndarray], a: float):
    """
    On-zenith Gaussian.

    {params}
    a: float
        Gaussian width parameter.

    {other}
    """
    return gauss(az, za, a)


@sky_model
def gauss(az: [None, float, np.ndarray], za: [float, np.ndarray], a: float, el0vec: [None, np.ndarray]=None):
    """
    Gaussian, and off-zenith Gaussian.

    {params}
    a: float
        Gaussian width parameter.
    el0vec: ndarray of float
        Cartesian vector describing lmn displacement of Gaussian center, shape (3,).

    {other}
    """
    if az is None:
        az = np.zeros_like(za)

    if el0vec is None:
        el0vec = np.array([0, 0, 0])

    sigsq = a**2 / (2 * np.pi)
    lmn = _angle_to_lmn(az, za)
    disp = (lmn - el0vec)[:, :2]  # Only l and m
    ang = np.linalg.norm(disp, axis=1)
    try:
        return np.exp(-ang**2 / (2 * sigsq))
    except AttributeError:
        return mp.exp(-ang ** 2 / (2 * sigsq))


@sky_model
def xysincs(az: [None, float, np.ndarray], za: [float, np.ndarray], a: float, xi: float=0.0):
    """
    Crossed sinc functions in l,m coordinates.

    Defined by:
        I(x,y) = sinc(ax) sinc(ay) cos(1 - x^2 - y^2), for |x| and |y| < 1/sqrt(2)
        = 0, otherwise
    where (x,y) is rotated from (l,m) by an angle xi.

    {params}
    a: float
        Sinc parameter, giving width of the "box" in uv space.
        box width in UV is a/(2 pi)
    xi: float
        Rotation of (x,y) coordinates from (l,m) in radians.

    {other}
    """
    cx, sx = np.cos(xi), np.sin(xi)
    rot = np.array([[cx, sx, 0], [-sx, cx, 0], [0, 0, 1]])

    lmn = _angle_to_lmn(az, za)
    xyvecs = np.dot(lmn, rot)

    result = np.sinc(a * xyvecs[:, 0] / np.pi) * np.sinc(a * xyvecs[:, 1] / np.pi) * np.cos(za)
    outside = np.where((np.abs(xyvecs[:, 0]) > 1 / np.sqrt(2))
                       | (np.abs(xyvecs[:, 1]) > 1 / np.sqrt(2)))
    result[outside] = 0.0
    return result

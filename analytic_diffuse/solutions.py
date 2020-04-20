"""Library for analytic solutions."""

import numpy as np
import re
from scipy.special import jv, factorial as fac, gammaincc, gamma, sici, hyp0f1
from mpmath import hyp1f2


@np.vectorize
def vhyp1f2(a0, b0, b1, x):
    """Vectorized hyp1f2 function."""
    return complex(hyp1f2(a0, b0, b1, x))


def approx_hyp1f1(a, b, x, order=5):
    """Asymptotic approximation for large -|x|."""
    assert x < 0

    x = np.abs(x)  # expansion is valid for this case.
    outshape = a.shape
    a = a.flatten()
    first_term = gamma(b) * x**(a - b) * np.exp(-x) / gamma(a)
    ens = np.arange(order)[:, None]
    second_term = (gamma(b) * x**(-a) / gamma(b - a))\
        * np.sum(gamma(a + ens) * gamma(1 + a - b + ens)
                 / (x**ens * gamma(ens + 1) * gamma(a) * gamma(1 + a - b)), axis=0)
    return (first_term + second_term).reshape(outshape)


def monopole(uvecs, order=3):
    """
    Solution for I(r) = 1.

    Also handles nonzero-w case.

    Parameters
    ----------
    uvecs: ndarray of float
        cartesian baseline vectors in wavelengths, shape (Nbls, 3)
    order: int
        Expansion order to use for nonflat array case (w != 0).

    Returns
    -------
    ndarray of complex
        Visibilities, shape (Nbls,)
    """
    if np.allclose(uvecs[:, 2], 0):
        uamps = np.linalg.norm(uvecs, axis=1)
        return 2 * np.pi * np.sinc(2 * uamps)

    uvecs = uvecs[..., None]

    ks = np.arange(order)[None, :]
    fac0 = (2 * np.pi * 1j * uvecs[:, 2, :])**ks / (gamma(ks + 2))
    fac1 = hyp0f1((3 + ks) / 2, -np.pi**2 * (uvecs[:, 0, :]**2 + uvecs[:, 1, :]**2))
    return 2 * np.pi * np.sum(fac0 * fac1, axis=-1)


def cosza(uvecs):
    """
    Solution for I(r) = cos(r).

    Parameters
    ----------
    uvecs: ndarray of float
        cartesian baseline vectors in wavelengths, shape (Nbls, 3)

    Returns
    -------
    ndarray of complex
        Visibilities, shape (Nbls,)
    """
    uamps = np.linalg.norm(uvecs, axis=1)
    return jv(1, 2 * np.pi * uamps) * 1 / uamps


def polydome(uvecs, n=2):
    """
    Solution for I(r) = (1-r^n) * cos(zenith angle).

    Parameters
    ----------
    uvecs: ndarray of float
        Cartesian baseline vectors in wavelengths, shape (Nbls, 3)
    n: int
        Order of polynomial (Default of 2)
        Must be even.

    Returns
    -------
    ndarray of complex
        Visibilities, shape (Nbls,)
    """
    if not n % 2 == 0:
        raise ValueError("Polynomial order must be even.")
    uamps = np.linalg.norm(uvecs, axis=1)

    # Special case:
    if n == 2:
        return (jv(1, 2 * np.pi * uamps) * 1 / uamps - (jv(2, 2 * np.pi * uamps)
                - np.pi * uamps * jv(3, 2 * np.pi * uamps)) * (1 / (np.pi * uamps**2)))
    # General
    res = (1 / (n + 1)) * vhyp1f2(n + 1, 1, n + 2, -np.pi**2 * uamps**2)
    res[uamps == 0] = 2 * np.pi * n / (2 * n + 2)
    res = (jv(1, 2 * np.pi * uamps) * 1 / uamps) - res  # V_cosza - result
    return res


def projgauss(uvecs, a, order=50, usesmall=False, uselarge=False):
    """
    Solution for I(r) = exp(-r^2/2 a^2) * cos(r).

    Parameters
    ----------
    uvecs: ndarray of float
        Cartesian baseline vectors in wavelengths, shape (Nbls, 3)
    a: float
        Gaussian width parameter.
    order: int
        Expansion order.
    usesmall: bool, optional
        Use small-a approximation, regardless of a.
        Default is False
    uselarge: bool, optional
        Use large-a approximation, regardless of a.
        Default is False

    Returns
    -------
    ndarray of complex
        Visibilities, shape (Nbls,)
    """
    uamps = np.linalg.norm(uvecs, axis=1)
    ks = np.arange(order)[None, :]
    uamps = uamps[:, None]
    if uselarge and usesmall:
        raise ValueError("Cannot use both small and large approximations at once")
    if (a < 0.25 and not uselarge) or usesmall:
        return np.pi * a**2 * (
            np.exp(- np.pi**2 * a**2 * uamps**2)
            - np.sum(
                (-1)**ks * (np.pi * uamps * a)**(2 * ks) * gammaincc(ks + 1, 1 / a**2)
                / (fac(ks))**2, axis=1
            )
        )
    else:
        return np.sum(np.pi * (-1)**ks * vhyp1f2(1 + ks, 1, 2 + ks, -np.pi**2 * uamps**2)
                      / (a**(2 * ks) * fac(ks + 1)), axis=1)


def gauss(uvecs, a, el0vec=None, order=10, usesmall=False, uselarge=False, hyporder=5):
    """
    Solution for I(r) = exp(-r^2/2 a^2).

    Parameters
    ----------
    uvecs: ndarray of float
        Cartesian baseline vectors in wavelengths, shape (Nbls, 3)
    a: float
        Gaussian width parameter.
    el0vec: ndarray of float
        Cartesian vector describing lmn displacement of Gaussian center, shape (3,).
    order: int
        Expansion order.
    usesmall: bool, optional
        Use small-a approximation, regardless of a.
        Default is False
    uselarge: bool, optional
        Use large-a approximation, regardless of a.
        Default is False
    hyporder: int, optional
        Expansion order for hypergeometric 1F1 function evaluation.
        Default is 5.

    Returns
    -------
    ndarray of complex
        Visibilities, shape (Nbls,)
    """
    uamps = np.linalg.norm(uvecs, axis=1)
    if el0vec is not None:
        udotel0 = np.dot(uvecs, el0vec)
        el0 = np.linalg.norm(el0vec)
        el0_x = (udotel0.T / uamps).T
    else:
        udotel0 = 0.0
        el0_x = 0
        el0 = 0

    u_in_series = np.sqrt(uamps**2 - el0**2 / a**4 + 2j * uamps * el0_x / a**2)

    ks = np.arange(order)
    v = u_in_series
    if (a < np.pi / 8 and not uselarge) or usesmall:
        phasor = np.exp(-2 * np.pi * 1j * udotel0) * np.exp(-np.pi * a**2 * uamps**2)
        hypterms = approx_hyp1f1(ks + 1, 3 / 2, -np.pi / a**2, order=hyporder)
        ks = ks[None, :]
        v = v[:, None]
        hypterms = hypterms[None, :]
        series = (np.sqrt(np.pi) * v * a)**(2 * ks) / gamma(ks + 1)
        return (2 * np.pi * phasor * np.sum(series * hypterms, axis=1)).squeeze()
    else:
        # order >= 40
        phasor = np.exp(-np.pi * el0**2 / a**2)
        v = v[:, None]
        ks = ks[None, :]
        hypterms = vhyp1f2(ks + 1, 1, ks + 3 / 2, -np.pi**2 * v**2)
        ksum = np.sum((-1)**ks * (np.pi / a**2)**ks * hypterms / gamma(ks + 3 / 2), axis=1)
        res = phasor * np.pi**(3 / 2) * ksum
        return res.squeeze()


def xysincs(uvecs, a, xi=0.0):
    """
    Solution for the xysincs model.

    Defined as:
        I(x,y) = sinc(ax) sinc(ay) cos(1 - x^2 - y^2), for |x| and |y| < 1/sqrt(2)
               = 0, otherwise

        where (x,y) is rotated from (l,m) by an angle xi.

    In UV space, this resembles a rotated square.

    Parameters
    ----------
    uvecs: ndarray of float
        Cartesian baseline vectors in wavelengths, shape (Nbls, 3)
    a: float
        Sinc parameter, giving width of the "box" in uv space.
        box width in UV is a/(2 pi)
    xi: float
        Rotation of (x,y) coordinates from (l,m) in radians.

    Returns
    -------
    ndarray of complex
        Visibilities, shape (Nbls,)
    """
    assert np.allclose(uvecs[:, 2], 0)
    cx, sx = np.cos(xi), np.sin(xi)
    rot = np.array([[cx, sx, 0], [-sx, cx, 0], [0, 0, 1]])

    xyvecs = np.dot(uvecs, rot)

    x = 2 * np.pi * xyvecs[:, 0]
    y = 2 * np.pi * xyvecs[:, 1]

    x = x.astype(complex)
    y = y.astype(complex)

    b = 1 / np.sqrt(2.0)

    xfac = (np.sign(a - x) - np.sign(a - x)) * np.pi / 2
    yfac = (np.sign(a - y) - np.sign(a - y)) * np.pi / 2
    xpart = (sici(b * (a - x))[0] + sici(b * (a + x))[0] + xfac) / a
    ypart = (sici(b * (a - y))[0] + sici(b * (a + y))[0] + yfac) / a

    return ypart * xpart


def parse_filename(fname):
    """
    Interpret model and other parameters from filename.

    Current healvis simulations of these models follow a particular
    naming convention. See example files in data directory.
    """
    params = {}
    if 'monopole' in fname:
        sky = 'monopole'
    elif 'cosza' in fname:
        sky = 'cosza'
    elif 'quaddome' in fname:
        sky = 'polydome'
        params['n'] = 2
    elif 'projgauss' in fname:
        sky = 'projgauss'
        params['a'] = float(re.search(r'(?<=gauss-)\d*\.?\d*', fname).group(0))
    elif 'fullgauss' in fname:
        sky = 'gauss'
        params['a'] = float(re.search(r'(?<=gauss-)\d*\.?\d*', fname).group(0))
        if 'offzen' in fname:
            zenang = np.radians(5.0)
            if 'small-offzen' in fname:
                zenang = np.radians(0.5)
            params['el0vec'] = np.array([np.sin(zenang), 0, 0])
    elif 'xysincs' in fname:
        sky = 'xysincs'
        params['xi'] = np.pi / 4
        params['a'] = float(re.search(r'(?<=-a)\d*\.?\d*', fname).group(0))
    else:
        raise ValueError("No existing model found for {}".format(fname))

    return sky, params

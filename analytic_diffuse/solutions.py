"""Library for analytic solutions."""

import numpy as np
import re
from scipy.special import jv, factorial as fac, gammaincc, gamma, sici, hyp0f1
from mpmath import hyp1f2
import warnings

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


def vec_to_amp(vec):
    vec = np.atleast_1d(vec)
    assert vec.ndim in (1, 2)
    if vec.ndim == 2:
        assert vec.shape[1] <= 3
    return vec if vec.ndim == 1 else np.linalg.norm(vec, axis=1)


def _perform_convergent_sum(fnc, u, order, chunk_order, atol, rtol, ret_cumsum, complex, *args):
    # Set the default order (100 if we're going to convergence, 30 otherwise)
    order = order or (100 if chunk_order else 30)

    # If we're not going to convergence, set chunk_order to order, so we just do the
    # whole sum at once.
    chunk_order = chunk_order or order

    # Find out how many chunks we'll need (accounting for the bit at the end if
    # they don't divide evenly)
    n_chunks = order // chunk_order
    if order % chunk_order:
        n_chunks += 1

    # Now set the actual order (>= original order)
    order = n_chunks * chunk_order

    sm = np.zeros(u.shape + (order, ), dtype=np.complex if complex else np.float)
    u = u[..., None]

    counter = 0
    while (
        counter < 2 or not np.allclose(sm[..., counter-1], sm[..., counter-2], atol=atol, rtol=rtol)) and counter < order:
        ks = np.arange(counter, counter + chunk_order)
        sm[..., ks] = sm[..., counter-1][..., None] + np.cumsum(fnc(ks[None, ...], u, *args), axis=-1)
        counter += chunk_order

    if counter==order and order >= 2 and not np.allclose(sm[..., -1], sm[..., -2]):
        warnings.warn("Desired tolerance not reached. Try setting order higher.")

    # Restrict to actual calculated terms
    sm = sm[..., :counter]

    if ret_cumsum:
        return sm.squeeze()
    else:
        return sm[..., -1].squeeze()


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

    if uvecs.ndim == 1 or uvecs.shape[1] == 2 or np.allclose(uvecs[:, 2], 0):
        # w is zero.
        uamps = vec_to_amp(uvecs)
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
    uamps = vec_to_amp(uvecs)
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
    if n % 2:
        raise ValueError("Polynomial order must be even.")
    uamps = vec_to_amp(uvecs)

    # Special case:
    if n == 2:
        return (jv(1, 2 * np.pi * uamps) * 1 / uamps - (jv(2, 2 * np.pi * uamps)
                - np.pi * uamps * jv(3, 2 * np.pi * uamps)) * (1 / (np.pi * uamps**2)))
    # General
    res = (1 / (n + 1)) * vhyp1f2(n + 1, 1, n + 2, -np.pi**2 * uamps**2)
    res[uamps == 0] = 2 * np.pi * n / (2 * n + 2)
    res = (jv(1, 2 * np.pi * uamps) * 1 / uamps) - res  # V_cosza - result
    return res


def projgauss(uvecs, a, order=None, chunk_order=0, usesmall=False, uselarge=False, atol=1e-10, rtol=1e-8, ret_cumsum=False):
    """
    Solution for I(r) = exp(-r^2/2 a^2) * cos(r).

    Parameters
    ----------
    uvecs: ndarray of float
        Cartesian baseline vectors in wavelengths, shape (Nbls, 3)
    a: float
        Gaussian width parameter.
    order: int
        If not `chunk_order`, the expansion order. Otherwise, this is the *maximum*
        order of the expansion.
    chunk_order: int
        If non-zero, the expansion will be summed until convergence (or max order is
        reached).
    usesmall: bool, optional
        Use small-a approximation, regardless of a.
        Default is False
    uselarge: bool, optional
        Use large-a approximation, regardless of a.
        Default is False
    ret_cumsum : bool, optional
        Whether to return the full cumulative sum of the expansion.

    Returns
    -------
    ndarray of complex
        Visibilities, shape (Nbls,) (or (Nbls, nk) if `ret_cumsum` is True.)
    """
    u_amps = vec_to_amp(uvecs)

    if uselarge and usesmall:
        raise ValueError("Cannot use both small and large approximations at once")

    usesmall = (a < 0.25 and not uselarge) or usesmall

    if usesmall:
        fnc = lambda ks, u: (
            (-1) ** ks * (np.pi * u * a) ** (2 * ks) * gammaincc(ks + 1, 1 / a ** 2)
            / (fac(ks)) ** 2
        )
    else:
        fnc = lambda ks, u: (
            np.pi * (-1)**ks * vhyp1f2(1 + ks, 1, 2 + ks, -np.pi**2 * u**2)
            / (a**(2 * ks) * fac(ks + 1))
        )

    result = _perform_convergent_sum(fnc, u_amps, order, chunk_order, atol, rtol, ret_cumsum, complex=False)

    if usesmall:
        exp = np.exp(-(np.pi * a * u_amps) ** 2)
        if np.shape(result) != np.shape(exp):
            exp = exp[..., None]
        result = np.pi * a ** 2 * (exp - result).squeeze()
    return result


def gauss(uvecs, a, el0vec=None, order=None, chunk_order=0, usesmall=False, uselarge=False, hyporder=5, atol=1e-10, rtol=1e-8, ret_cumsum=False):
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
    u_amps = vec_to_amp(uvecs)

    if el0vec is not None:
        udotel0 = np.dot(uvecs, el0vec)
        el0 = np.linalg.norm(el0vec)
        el0_x = (udotel0.T / u_amps).T
    else:
        udotel0 = 0.0
        el0_x = 0
        el0 = 0

    usesmall = (a < np.pi / 8 and not uselarge) or usesmall

    u_in_series = np.sqrt(u_amps**2 - el0**2 / a**4 + 2j * u_amps * el0_x / a**2)

    if usesmall:
        def fnc(ks, v):
            hypterms = approx_hyp1f1(ks + 1, 3 / 2, -np.pi / a**2, order=hyporder)
            hypterms = hypterms[None, :]
            series = (np.sqrt(np.pi) * v * a)**(2 * ks) / gamma(ks + 1)
            return series * hypterms
    else:
        def fnc(ks, v):
            hypterms = vhyp1f2(ks + 1, 1, ks + 3 / 2, -np.pi**2 * v**2)
            return (-1)**ks * (np.pi / a**2)**ks * hypterms / gamma(ks + 3 / 2)

    result = _perform_convergent_sum(fnc, u_in_series, order, chunk_order, atol, rtol, ret_cumsum, complex=True)

    if usesmall:
        phasor = np.exp(-2 * np.pi * 1j * udotel0) * np.exp(-np.pi * a ** 2 * u_amps ** 2)
        return (2 * np.pi * phasor * result).squeeze()
    else:
        phasor = np.exp(-np.pi * el0 ** 2 / a ** 2)
        return (phasor * np.pi ** (3 / 2) * result).squeeze()


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
    assert uvecs.ndim == 2 and uvecs.shape[1] == 3
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

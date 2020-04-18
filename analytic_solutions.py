#!/bin/env python

# Library for analytic solutions.

import numpy as np
from functools import update_wrapper
from scipy.special import jv, struve, factorial as fac, gammaincc, hyp1f1, gamma
from mpmath import hyp1f2

@np.vectorize
def vhyp1f2(a0,b0,b1,x):
    return complex(hyp1f2(a0,b0,b1,x))


def approx_hyp1f1(a, b, x, order=5):
    """Asymptotic approximation for large -|x|."""
    assert x < 0

    x = np.abs(x)  # expansion is valid for this case.
    outshape = a.shape
    a = a.flatten()
    first_term = gamma(b) * x**(a - b) * np.exp(-x) / gamma(a)
    ens = np.arange(order)[:, None]
    second_term = (gamma(b) * x**(-a) / gamma(b-a))\
        * np.sum(gamma(a+ens) * gamma(1 + a - b + ens)
                 / (x**ens * gamma(ens + 1) * gamma(a) * gamma(1 + a - b))
        , axis=0)
    return (first_term + second_term).reshape(outshape)


def monopole(uamps):
    """
    Solution for I(r) = 1.

    Parameters
    ----------
    uamps: ndarray of float
        Baseline lengths in wavelengths.

    Returns
    -------
    ndarray of complex
        Visibilities corresponding with uamps.
    """
    return 2*np.pi *  np.sinc(2*u)

def cosza(uamps):
    """
    Solution for I(r) = cos(r).

    Parameters
    ----------
    uamps: ndarray of float
        Baseline lengths in wavelengths.

    Returns
    -------
    ndarray of complex
        Visibilities corresponding with uamps.
    """
    return jv(1, 2 * np.pi * u) * 1 / u

def quaddome(uamps):
    """
    Solution for I(r) = 1-r^2.

    Parameters
    ----------
    uamps: ndarray of float
        Baseline lengths in wavelengths.

    Returns
    -------
    ndarray of complex
        Visibilities corresponding with uamps.
    """
    return (jv(1, 2 * np.pi * u) * 1 / u  - (jv(2, 2 * np.pi * u) - np.pi * u * jv(3, 2*np.pi*u)) * (1 / (np.pi * u**2)))

def projgauss(uamps, a, order=50, usesmall=False, uselarge=False):
    """
    Solution for I(r) = exp(-r^2/2 a^2) * cos(r).

    Parameters
    ----------
    uamps: ndarray of float
        Baseline lengths in wavelengths.
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
        Visibilities corresponding with uamps.
    """
    ks = np.arange(order)[None,:]
    u = u[:, None]
    if uselarge and usesmall:
        raise ValueError("Cannot use both small and large approximations at once")
    if (a < 0.25 and not uselarge) or usesmall:
        return  np.pi * a**2 * (np.exp(- np.pi**2 * a**2 * u**2)
               - np.sum( (-1)**ks * (np.pi * u * a)**(2*ks) * gammaincc(ks+1, 1/a**2)
                         / (fac(ks))**2, axis=1))
    else:
        return  np.sum(np.pi * (-1)**ks * vhyp1f2(1+ks, 1, 2+ks, -np.pi**2 * u**2)
                           / (a**(2*ks) * fac(ks+1)), axis=1)

def gauss(uamps=None, uvecs=None, a=None, el0vec=None, order=10, usesmall=False, uselarge=False):
    """
    Solution for I(r) = exp(-r^2/2 a^2).

    Parameters
    ----------
    uamps: ndarray of float
        Baseline lengths in wavelengths., shape (Nbls,) for Nbls baselines.
    uvecs: ndarray of float
        Cartesian baseline vectors in wavelengths, shape (Nbls, 3)
        Overrides uamps if set.
        Required if el0vec is not None.
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

    Returns
    -------
    ndarray of complex
        Visibilities corresponding with uamps.
    """
    if el0vec is not None:
        assert uvecs is not None
        udotel0 = np.dot(uvecs, el0vec)
        el0 = np.linalg.norm(el0vec)
        el0_x = (udotel0.T / uamps).T
    else:
        el0_x = 0
        el0 = 0

    u_in_series = np.sqrt(uamps**2  - el0**2/a**4 + 2j * uamps * el0_x / a**2)
    u = uamps

    ks = np.arange(order)[None, :]
    v = u_in_series[:, None]
    u = u[:, None]

    if (a < np.pi/8 and not uselarge) or usesmall:
        phasor = np.exp(-2 * np.pi * 1j * udotel0) * np.exp(-np.pi * a**2 * u**2)
        hypterms = approx_hyp1f1(ks + 1, 3/2, -np.pi / a**2)
        series = (np.sqrt(np.pi) * v * a)**(2*ks) / gamma(ks + 1)
        return (phasor *  2*np.pi * np.sum(series * hypterms, axis=1)).squeeze()
    else:
        # order >= 40
        phasor = np.exp(-np.pi * el0**2/a**2)
        ks = np.arange(order)[:,None]
        v = u_in_series[None,:]
        hypterms = vhyp1f2(ks + 1, 1, ks + 3/2, -np.pi**2 * v**2)
        ksum = np.sum((-1)**ks * (np.pi/a**2)**ks * hypterms / gamma(ks + 3/2), axis=1)
        res = phasor *  np.pi**(3/2) * ksum
        return res.squeeze()

from scipy import integrate
import numpy as np
from . import models
from scipy.special import j0, jn_zeros
import mpmath as mp
from typing import Optional


def hankel_solver(model: [callable, str], u: [float, np.ndarray], quad_kwargs: Optional[dict]=None,
                  high_precision: bool=False, use_points: bool=False, *args, **kwargs):
    """
    Perform transform to determine visibility solutions for circularly symmetric
    sky models.

    Performs the integral

    .. math:: V(u) = \int_0^1 r I'(r) J_0(2\pi u r) dr,

    where ``r`` is ``sin(za)`` and ``I'(r)`` is the projected sky model, ``I(r)/cos(za)``.

    Parameters
    ----------
    model : callable or str
        The sky model to integrate. Must be a model decorated with `circsym` (whether
        user-defined or built-in). If a string, just the function name.
    u : float or ndarray
        The magnitude(s) of the baseline vectors at which to evaluate the visibility,
        in units of wavelengths.
    quad_kwargs : dict, optional
        Keyword arguments to pass to the `quad` function to control its behaviour.
        Note that if ``high_precision`` is True, the `quad` function is from mpmath,
        otherwise it is from scipy.
    high_precision : bool, optional
        Whether to use the high-precision library mpmath for evaluating the model and
        the integral.
    use_points : bool, optional
        Whether to specify zeros of the Bessel function to pass to the quad routine, to
        aid evaluation of highly-oscillatory integrals.
    args :
        Extra arguments passed to the model
    kwargs :
        Extra arguments passed to the model

    Returns
    -------
    float or ndarray
        The real visibility solution at the given baseline magnitudes, ``u``.
        Will be a scalar float if u is a float, ndarray otherwise.

    Notes
    -----
    This function is valid for any circularly symmetric sky model (whether defined in
    projected co-ordinates or not), even though analytically it is not the easiest way
    to integrate any particular model.
    """
    quad_kwargs = quad_kwargs or {}
    if callable(model):
        assert model.__name__ in models.circular_models
    else:
        assert model in models.circular_models
        model = getattr(models, model)

    # Define functions depending on whether using high precision or not.
    # Faster if defined outside the integrand function.
    if high_precision:
        asin = mp.asin
        J0 = mp.j0
        cos = mp.cos
    else:
        asin = np.arcsin
        J0 = j0
        cos = np.cos

    tau = 2 * np.pi

    # Define the integrand. We get some speed up in models that are defined in projected
    # co-ordinates by not multiplying then dividing by cos(za).
    def integrand(r, uu):
        return tau * r * model(None, asin(r), projected=True, *args, **kwargs) * J0(tau * uu * r)

    scalar = np.isscalar(u)
    u = np.atleast_1d(u)

    # If the user wants to specify integration points, get the list of zeros of the
    # Bessel function, out to r=1. Only use the points if there's more than 5.
    if use_points:
        points = [jn_zeros(0, int(2*uu))/(tau*uu) if uu > 5 else [] for uu in u]
        points = [p[p < 1].tolist() if len(p) else [] for p in points]
    else:
        points = [[]]*len(u)

    if high_precision:
        res = [float(mp.quad(lambda r: integrand(r, uu), [0] + p + [1], **quad_kwargs)) for uu, p in zip(u, points)]
    else:
        res = [integrate.quad(integrand, 0, 1, args=(uu,), points=p, **quad_kwargs)[0] for uu, p in zip(u, points)]

    if scalar:
        return res[0]
    else:
        return res

from scipy import integrate
import numpy as np
from . import models
from scipy.special import j0, jn_zeros
import mpmath as mp

def hankel_solver(model: [callable, str], u: [float, np.ndarray], quad_kwargs={}, high_precision=False, use_points=False, *args, **kwargs):
    if callable(model):
        assert model.__name__ in models._funcnames
    else:
        model = getattr(models, model)

    if model.__name__ in models._projected_funcs:
        # using phi here because that's what we use in the paper, though in models.py it's theta
        def integrand(r, uu):
            if high_precision:
                bj0 = mp.j0(2 * np.pi * uu * r) if high_precision else j0(2 * np.pi * uu * r)
                phi = mp.asin(r)
            else:
                bj0 = j0(2 * np.pi * uu * r)
                phi = np.arcsin(r)

            return 2*np.pi * r * model(None, phi, projected=True, *args, **kwargs) * bj0
    else:
        def integrand(r, uu):
            if high_precision:
                phi = mp.asin(r)
                n = mp.cos(phi)
                bj0 = mp.j0(2 * np.pi * uu * r)
            else:
                phi = np.arcsin(r)
                bj0 = j0(2 * np.pi * uu * r)
                n = np.cos(phi)

            return 2*np.pi * r * model(None, phi, *args, **kwargs) / n * bj0

    scalar = np.isscalar(u)
    u = np.atleast_1d(u)

    if use_points:
        points = [jn_zeros(0, int(2*uu))/(2*np.pi*uu) if uu > 5 else [] for uu in u]
        points = [p[p<1].tolist() if len(p) else [] for p in points]
    else:
        points = [[]]*len(u)

    quad = mp.quad if high_precision else integrate.quad
    if high_precision:
        res = [float(mp.quad(lambda r: integrand(r, uu), [0] + p + [1], **quad_kwargs)) for uu, p in zip(u, points)]
    else:
        res = [quad(integrand, 0, 1, args=(uu), points=p, **quad_kwargs)[0] for uu, p in zip(u, points)]

    if scalar:
        return res[0]
    else:
        return res

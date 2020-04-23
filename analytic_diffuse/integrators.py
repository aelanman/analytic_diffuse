from scipy.integrate import quad
import numpy as np
from . import models
from scipy.special import j0


def hankel_solver(model: [callable, str], u: [float, np.ndarray], quad_kwargs={}, *args, **kwargs):
    if callable(model):
        assert model.__name__ in models._funcnames
    else:
        model = getattr(models, model)

    def integrand(r, uu):
        phi = np.arcsin(r)  # using phi here because that's what we use in the paper, though in models.py it's theta
        return 2*np.pi * r * model(None, phi, *args, **kwargs) / np.cos(phi) * j0(2*np.pi* uu *r)

    scalar = np.isscalar(u)
    u = np.atleast_1d(u)
    res = [quad(integrand, 0, 1, args=(uu), **quad_kwargs) for uu in u]

    res, info = [r[0] for r in res], [r[1:] for r in res]

    if scalar:
        return res[0], info[0]
    else:
        return res, info
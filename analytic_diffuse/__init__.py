from . import models
from . import solutions

from .models import _funcnames as available_models


def get_model(name):
    """
    Get a given model function.

    Parameters
    ----------
    name: str
        Name of the model.

    Returns
    -------
    function
        Callable function.
    """
    if name not in available_models:
        raise ValueError("Model {} is not available".format(name))

    return getattr(models, name)


def get_solution(name):
    """
    Get a given solution function.

    Parameters
    ----------
    name: str
        Name of the model.

    Returns
    -------
    function
        Callable function.
    """
    if name not in available_models:
        raise ValueError("Solution {} is not available".format(name))

    return getattr(solutions, name)

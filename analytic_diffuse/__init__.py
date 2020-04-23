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
        raise ValueError(f"Model {name} is not available")

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
        raise ValueError(f"Solution {name} is not available")

    return getattr(solutions, name)

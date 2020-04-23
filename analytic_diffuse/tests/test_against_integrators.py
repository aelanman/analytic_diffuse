from analytic_diffuse import integrators, models, get_model, get_solution
import pytest
import numpy as np


@pytest.mark.parametrize(
    'model,kwargs',
    [('cosza', {}),
     ('polydome', {}),
     ('projgauss', {'a':0.2})]
)
def test_against_hankel(model, kwargs):
    u = np.array([0.1, 1])

    anl = get_solution(model)(u, **kwargs)
    num = integrators.hankel_solver(model, u, quad_kwargs=dict(epsabs=0, epsrel=1e-8), **kwargs)[0]

    assert np.allclose(anl, num, rtol=1e-8)
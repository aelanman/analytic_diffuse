from analytic_diffuse import integrators, models, get_model, get_solution
import pytest
import numpy as np


@pytest.mark.parametrize(
    'model,kwargs',
    [('cosza', {}),
     ('polydome', {}),
     ('projgauss', {'a': 0.01}),
     ('projgauss', {'a': 2}),
     ]
)
def test_against_hankel(model, kwargs):
    u = np.array([0.1, 1, 10, 100])

    anl = get_solution(model)(u, **kwargs)
    num = integrators.hankel_solver(model, u, quad_kwargs=dict(epsabs=0, limit=100, epsrel=1e-8),  **kwargs)[0]

    assert np.allclose(anl, num, rtol=1e-8)


@pytest.mark.parametrize('a', (0.2, 0.25, 0.5))
def test_projgauss_mid_a(a):
    u = np.array([0.1, 1, 10, 100])

    anl = get_solution('projgauss')(u, a=a)
    num = integrators.hankel_solver('projgauss', u, quad_kwargs=dict(epsabs=0, limit=100, epsrel=1e-8),  a=a)[0]

    assert np.allclose(anl, num, rtol=1e-8)
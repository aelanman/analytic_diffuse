from analytic_diffuse import integrators, models, get_model, get_solution
import pytest
import numpy as np


@pytest.mark.parametrize(
    'model,kwargs',
    [('cosza', {}),
     ('polydome', {}),
     ('projgauss', {'a': 0.01}),
     ('projgauss', {'a': 2}),
     ('gauss_zenith', {'a': 0.01}),
     ('gauss_zenith', {'a': 2})
     ]
)
def test_against_hankel(model, kwargs):
    u = np.array([0.1, 1, 10, 100])

    anl = get_solution(model)(u, **kwargs)
    num = integrators.hankel_solver(model, u, quad_kwargs=dict(epsabs=0, limit=100, epsrel=1e-8), **kwargs)

    assert np.allclose(anl, num, rtol=1e-8)


@pytest.mark.parametrize('a', (0.1, 0.2, 0.25, 0.5))
def test_projgauss_mid_a(a):
    if a < 0.25:
        u = np.array([0.01, 0.1, 1]) / a
    else:
        u = np.array([0.01, 0.1, 1, 10, 100]) / a

    anl = get_solution('projgauss')(u, a=a, order=1000, chunk_order=20)
    num = integrators.hankel_solver('projgauss', u, quad_kwargs=dict(epsabs=0, limit=1000, epsrel=1e-8), use_points=True, a=a)

    assert np.allclose(anl, num, rtol=1e-8)


def test_gauss_offzen_small():
    #a = 2.05
    a = 5e-3
    el0 = np.array([np.sin(np.radians(89)),0,0])
    kwargs = {"a": a, "el0vec": el0}

    u = np.linspace(1, 10, 5) 
    v = u.copy()

    num = np.asarray(integrators.dbl_int_solver('gauss', u, v, **kwargs))
    anl = get_solution('gauss')(np.vstack((u, v, np.zeros_like(v))).T, **kwargs)
    import IPython; IPython.embed()
    print(anl, num)

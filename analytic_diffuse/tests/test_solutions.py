from analytic_diffuse.solutions import projgauss
import numpy as np


def test_return_cumsum():
    res = projgauss(1, 0.01, order=30, ret_cumsum=True)
    assert res.shape == (30, )

    res = projgauss(1, 0.01, order=30, ret_cumsum=False)
    assert np.isscalar(res)

    res = projgauss(np.linspace(1,2, 10), 0.01, order=30, ret_cumsum=True)
    assert res.shape == (10, 30)

    res = projgauss(np.linspace(1,2, 10), 0.01, order=30, ret_cumsum=False)
    assert res.shape == (10, )


def test_chunking():
    res = projgauss(1, 0.01, order=30)
    res2 = projgauss(1, 0.01, order=30, chunk_order=30)
    assert res == res2

    res3= projgauss(1, 0.01, order=30, chunk_order=15, ret_cumsum=True)
    assert np.isclose(res3[-1], res2)
    assert res3.shape[0] <= 30  # could be smaller, since it may converge.

    res4 = projgauss(1, 0.01, chunk_order=10, ret_cumsum=True)
    assert np.isclose(res4[-1], res3[-1])
    assert res4.shape[0] <= 100  # default maximum order.

    res5 = projgauss(1, 0.01, chunk_order=1, ret_cumsum=True)
    assert np.isclose(res5[-1], res4[-1])
    assert res5.shape[0] <= res4.shape[0]   # convergence should be stable.


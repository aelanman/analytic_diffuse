import numpy as np
import h5py
import os
import pytest

from analytic_diffuse import models
from analytic_diffuse.data import DATA_PATH

from healvis.observatory import Observatory


obs = Observatory(0, 0)
obs.set_fov(360.0)
Nfreqs = 10

nside = 128
theta, phi, inds = obs.calc_azza(nside, [0, 0], return_inds=True)
inhorizon = theta <= np.pi / 2

# Amplitude of test maps
amp = 2.0 / (4 * np.pi / (12 * nside**2))

names = ['cosza', 'polydome', 'projgauss', 'gauss', 'xysincs']
params = [{}, {'n': 2}, {'a': 0.5}, {'a': 0.5}, {'a': 64, 'xi': np.pi / 4}]
keys = ['cosza', 'quaddome', 'projgauss-0.5', 'fullgauss-0.50', 'xysincs-a64']

@pytest.mark.parametrize('mod, param, key', zip(names, params, keys))
def test_against_data(mod, param, key):
    analytic = getattr(models, mod)
    test0 = analytic(phi, theta, **param)
    fname = 'analytic_test_model_' + key + '_nside' + str(nside) + '.hdf5'
    f0 = h5py.File(os.path.join(DATA_PATH, fname), 'r')
    map0 = f0['data'][0, :, 0]
    f0.close()
    print(mod)
    assert np.allclose(map0[inhorizon], amp * test0[inhorizon])

# script to evaluate a given visibility solution and save to npz.
# optionally -- parse a pyuvdata-compatible file to get baseline vectors etc. and save alongside evaluated data.
#   give model parameters (and name) as arguments. Option to parse a filename for that info too.

import numpy as np
import os
import argparse
from pyuvdata import UVData
from healvis.cosmology import c_ms
from healvis.utils import jy2Tsr

import analytic_diffuse as andiff

parser = argparse.ArgumentParser(
    description="Evaluate a given visibility solution and save results to an npz file."
)

parser.add_argument('-v', '--visfile', type=str, help='pyuvdata-compatible visibility data file')
parser.add_argument('--infile', type=str, help='Input npz file containing uvw vectors')
helpstr = 'Model name. Available: ' + ', '.join(andiff.available_models)
parser.add_argument('--model', type=str, help=helpstr, default=None)
parser.add_argument('-a', type=float, help='a parameter for gaussian and xysincs models.', default=None)
parser.add_argument('--el0vec', type=str, help='(l,m,n) coordinate vector for center displacement from zenith. Comma-delimited string.', default=None, required=False)
parser.add_argument('--el0ang', type=str, help='(azimuth, zenith angle) coordinate vector for center position.', default=None)
parser.add_argument('--xi', type=float, help='Xi parameter for xysincs model.', default=None)
parser.add_argument('-n', type=int, help='Polynomial order for polydome model.', default=None)
parser.add_argument('--order', type=int, help='Expansion order, for series solutions.')
parser.add_argument('--amp', type=float, help='Amplitude of the model in. Default is 2 K.', default=2)
parser.add_argument('--maxu', type=float, help='Maximum baseline length in wavelengths.')
parser.add_argument('-o', '--ofname', type=str, help='Output npz file name', default=None)

args = parser.parse_args()


if args.el0vec is not None:
    args.el0vec = list(map(float, args.el0vec.split(',')))
    assert len(args.el0vec) == 3

if args.el0ang is not None:
    args.el0ang = list(map(float, args.el0ang.split(',')))
    assert len(args.el0ang) == 2
    if args.el0vec is None:
        args.el0vec = andiff.models._angle_to_lmn(*args.el0ang).squeeze()
        args.el0vec[2] = 0   # Scrap w component

outdict = {}

sel = slice(None)
if args.maxu is not None:
    sel = np.linalg.norm(uvw, axis=1) < args.maxu

uv = UVData()
if args.visfile is None:
    if args.model is None:
        raise ValueError("Model type needed.")
    if args.infile is None:
        raise ValueError("Input npz file needed.")
    f = np.load(args.infile)
    uvw = f['uvws']
    uvw = uvw[sel]
elif args.visfile is not None and args.infile is not None:
    raise ValueError("Cannot do both npz and uv input files at once.")
else:
    uv.read(args.visfile)
    print("Using visibility file {}".format(args.visfile))
    uv.select(times=uv.time_array[0])    # Select first time only.
    dat = uv.data_array[:,0,:,0]
    lam = c_ms/uv.freq_array[0]
    
    dat_Tsr = dat * jy2Tsr(uv.freq_array[0], bm=1.0)
    uvw = np.repeat(uv.uvw_array[:,:,None], uv.Nfreqs, axis=2)
    uvw = np.swapaxes((uvw/lam), 1,2)
    uvw = uvw.reshape((uv.Nbls * uv.Nfreqs, 3))
    dat_Tsr = dat_Tsr.flatten()
    uvw = uvw[sel]
    dat_Tsr = dat_Tsr[sel]
    if args.model is None:
        model, params = andiff.solutions.parse_filename(os.path.basename(args.visfile))
        args.model = model
        for k, v in params.items():
            if (hasattr(args, k)) and (getattr(args, k) is None):
                setattr(args, k, v)
    outdict['dat_Tsr'] = dat_Tsr


outdict['uvws'] = uvw

for key, val in vars(args).items():
    if val is not None:
        outdict[key] = val


# Next -- evaluate function, then save things to npz file.

analytic = andiff.get_solution(args.model)

params = {}
keys = ['a', 'n', 'xi', 'el0vec', 'order']

for k in keys:
    val = getattr(args, k)
    if val is not None:
        params[k] = val

if ('gauss' in args.model) or ('xysincs' in args.model):
    a = params.pop('a')
    if a is None:
        raise ValueError("Missing parameter 'a' for {}.".format(args.model))
    outdict['result'] = args.amp * analytic(uvw, a, **params)
    params['a'] = a
else:
    outdict['result'] = args.amp * analytic(uvw, **params)

if args.ofname is None:
    if args.infile is None:
        args.ofname = "ana_comp_"+args.model
    else:
        args.ofname = "ana_eval_"+args.model    # Comp for data comp, eval for just evaluation
    for k, val in params.items():
        if isinstance(val, int):
            args.ofname += '_{}{:d}'.format(k, val)
        elif k == 'el0vec':
            zenang = np.degrees(np.arcsin(np.linalg.norm(val)))
            args.ofname += '_offzen{:.1f}'.format(zenang)
        else:
            args.ofname += '_{}{:.2f}'.format(k, val)
    if 'nside' in uv.extra_keywords.keys():
        args.ofname += '_nside{}'.format(uv.extra_keywords['nside'])
    args.ofname += ".npz"

print("Saving results to {}".format(args.ofname))
np.savez(args.ofname, **outdict)

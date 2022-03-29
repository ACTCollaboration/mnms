from mnms import noise_models as nm
from soapack import interfaces as sints
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qid', dest='qid', nargs='+', type=str, required=True,
                    help='list of soapack array "qids"')

parser.add_argument('--mask-version', dest='mask_version', type=str, 
                    default=None, help='Look in mnms:mask_path/mask_version/ for mask')

parser.add_argument('--mask-name', dest='mask_name', type=str, default=None,
                    help='Load mnms:mask_path/mask_version/mask_name.fits')

parser.add_argument('--downgrade', dest='downgrade', type=int, default=1,
                    help='downgrade all data in pixel space by square of this many pixels per side')

parser.add_argument('--lmax', dest='lmax', type=int, default=None,
                    help='Bandlimit of covariance matrix.')

parser.add_argument('--lmin', dest='lmin', type=int, default=10)

parser.add_argument('--lambda', dest='lamb', type=float, required=False, default=1.8,
                    help='Parameter specifying width of wavelets kernels in log(ell).')

parser.add_argument('--n', dest='n', type=int, default=24,
                    help='Bandlimit (in radians per azimuthal radian) of the directional kernels.')

parser.add_argument('--fwhm-fact', dest='fwhm_fact', type=float, default=2., 
                    help='Factor determining smoothing scale at each wavelet scale: '
                    'FWHM = fact * pi / lmax, where lmax is the max wavelet ell.')

parser.add_argument('--union-sources', dest='union_sources', type=str, default=None,
                    help="Version string for soapack's union sources. E.g. " 
                    "'20210209_sncut_10_aggressive'. Will be used for inpainting.")

parser.add_argument('--kfilt-lbounds', dest='kfilt_lbounds', nargs='+', type=float, default=None,
                    help="The ly, lx scale for an ivar-weighted Gaussian kspace filter. E.g. " 
                    "'4000 5'. Will be used for kspace filtering.")

parser.add_argument('--notes', dest='notes', type=str, default=None, 
                    help='a simple notes string to manually distinguish this set of sims ')

parser.add_argument('--data-model', dest='data_model', type=str, default=None, 
                    help='soapack DataModel class to use')

parser.add_argument('--split', nargs='+', dest='split', type=int, 
                    help='if --no-auto-split, simulate this list of splits '
                    '(0-indexed) (default: %(default)s)')

parser.add_argument('--no-auto-split', dest='auto_split', default=True, 
                    action='store_false', help='if passed, do not simulate every '
                    'split for this array')

parser.add_argument('--maps', dest='maps', nargs='+', type=str, default=None,
                    help='simulate exactly these map_ids, overwriting if preexisting; '
                    'overriden by --nsims (default: %(default)s)')

parser.add_argument('--maps-start', dest='maps_start', type=int, default=None, 
                    help='like --maps, except iterate starting at this map_id '
                    '(default: %(default)s)')

parser.add_argument('--maps-end', dest='maps_end', type=int, default=None, 
                    help='like --maps, except end iteration with this map_id '
                    '(default: %(default)s)')

parser.add_argument('--maps-step', dest='maps_step', type=int, default=1,
                    help='like --maps, except step iteration over map_ids by '
                    'this number (default: %(default)s)')

parser.add_argument('--map', dest='alm', default=True, 
                    action='store_false', help='Generate simulated maps instead of alms.')

args = parser.parse_args()

if args.data_model:
    data_model = getattr(sints,args.data_model)()
else:
    data_model = None
    
model = nm.FSAWNoiseModel(
    *args.qid, data_model=data_model, downgrade=args.downgrade, lmax=args.lmax, mask_version=args.mask_version,
    mask_name=args.mask_name, union_sources=args.union_sources, kfilt_lbounds=args.kfilt_lbounds, 
    notes=args.notes, lamb=args.lamb, n=args.n, fwhm_fact=args.fwhm_fact, lmin=args.lmin)

# get split nums
if args.auto_split:
    splits = np.arange(model.num_splits)
else:
    splits = np.atleast_1d(args.split)
assert np.all(splits >= 0)

# get map nums
if args.maps is not None:
    assert args.maps_start is None and args.maps_end is None
    maps = np.atleast_1d(args.maps).astype(int)
else:
    assert args.maps_start is not None and args.maps_end is not None
    maps = np.arange(args.maps_start, args.maps_end+args.maps_step, args.maps_step)
assert np.all(maps >= 0)

# Iterate over sims
for i, s in enumerate(splits):
    for j, m in enumerate(maps):
        model.get_sim(s, m, alm=args.alm, check_on_disk=True, write=True, verbose=True)
    model.delete_model(s)

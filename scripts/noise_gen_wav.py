from mnms import noise_models as nm
from soapack import interfaces as sints
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qid', dest='qid', nargs='+', type=str, required=True,
                    help='list of soapack array "qids"')

parser.add_argument('--mask-version', dest='mask_version', type=str, 
                    default=None, help='Look in mnms:mask_path/mask_version/ '
                    'for mask (default: %(default)s)')

parser.add_argument('--mask-name', dest='mask_name', type=str, default=None,
                    help='Load mnms:mask_path/mask_version/mask_name.fits '
                    '(default: %(default)s)')

parser.add_argument('--downgrade', dest='downgrade', type=int, default=1,
                    help='downgrade all data in pixel space by square of this many '
                    'pixels per side (default: %(default)s)')

parser.add_argument('--lmax', dest='lmax', type=int, default=None,
                    help='Bandlimit of covariance matrix.')

parser.add_argument('--lambda', dest='lamb', type=float, required=False, default=1.3,
                    help='Parameter specifying width of wavelets kernels in log(ell).')

parser.add_argument('--smooth-loc', dest='smooth_loc', default=False, action='store_true',
                    help='If passed, use smoothing kernel that varies over the map, smaller along edge of mask.')

parser.add_argument('--union-sources', dest='union_sources', type=str, default=None,
                    help="Version string for soapack's union sources. E.g. " 
                    "'20210209_sncut_10_aggressive'. Will be used for inpainting.")

parser.add_argument('--kfilt-lbounds', dest='kfilt_lbounds', nargs='+', type=float, default=None,
                    help="The ly, lx scale for an ivar-weighted Gaussian kspace filter. E.g. " 
                    "'4000 5'. Will be used for kspace filtering.")

parser.add_argument('--notes', dest='notes', type=str, default=None, 
                    help='a simple notes string to manually distinguish this set of '
                    'sims (default: %(default)s)')

parser.add_argument('--data-model', dest='data_model', type=str, default=None, 
                    help='soapack DataModel class to use (default: %(default)s)')
args = parser.parse_args()

if args.data_model:
    data_model = getattr(sints,args.data_model)()
else:
    data_model = None
    
model = nm.WaveletNoiseModel(
    *args.qid, data_model=data_model, downgrade=args.downgrade, lmax=args.lmax, mask_version=args.mask_version,
    mask_name=args.mask_name, union_sources=args.union_sources, kfilt_lbounds=args.kfilt_lbounds, 
    notes=args.notes, lamb=args.lamb, smooth_loc=args.smooth_loc)
model.get_model(check_on_disk=True, verbose=True)

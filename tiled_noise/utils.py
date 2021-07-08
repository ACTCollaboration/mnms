from pixell import enmap, enplot, curvedsky
from soapack import interfaces as dmint
import numpy as np
from scipy.interpolate import interp1d

# Utility functions to support tiling classes and functions. Just keeping code organized so I don't get whelmed.

def eshow(x, *args, title=None, write=False, fname='',**kwargs): 
    plots = enplot.plot(x, **kwargs)
    if write:
        enplot.write(fname, plots)
    enplot.show(plots, title=title)

def slice_geometry_by_pixbox(ishape, iwcs, pixbox):
    pb = np.asarray(pixbox)
    return enmap.slice_geometry(ishape[-2:], iwcs, (slice(*pb[:, -2]), slice(*pb[:, -1])), nowrap=True)

# users must be careful with indices shape. it gets promoted to 2D array by
# prepending a dimension if necessary. afterward, indices[i] (ie, indexing along
# axis=0 of 2D indices array) corresponds to axis[i]
def get_take_indexing_obj(arr, indices, axis=None):
    if axis is not None:
        shape = arr.shape
        slicelist = [slice(dim) for dim in shape]
        axis = np.atleast_1d(axis)
        indices = atleast_nd(indices, 2)
        for i in range(len(axis)):
            slicelist[axis[i]] = tuple(indices[i])
        return tuple(slicelist)
    else:
        return np.array(indices)

# this is like enmap.partial_flatten except you give the axes you *want* to flatten
def flatten_axis(imap, axis=None, pos=0):
    if axis is None:
        return imap.reshape(-1)
    else:
        imap = np.moveaxis(imap, axis, range(len(axis))) # put the axes to be flattened in front
        imap = imap.reshape((-1,) + imap.shape[len(axis):]) # flatten the front dimensions
        imap = np.moveaxis(imap, 0, pos) # put front dimension in pos
        return imap

# this is like enmap.partial_expand except you give the axes you *want* to restore
# shape is the shape of the restored axes or the full original array with restored axes
def unflatten_axis(imap, shape, axis=None, pos=0):
    if axis is None:
        return imap
    else:
        imap = np.moveaxis(imap, pos, 0) # put pos in front dimension
        
        shape = np.atleast_1d(shape)
        axis = np.atleast_1d(axis)
        if len(shape) == len(axis):
            shape = tuple(shape)
        elif len(shape) == len(axis) + len(imap.shape[1:]):
            shape = tuple(shape[axis])
        else:
            raise ValueError('Shape arg must either be the same length as axis or the restored array shape')
        imap = imap.reshape(shape + imap.shape[1:]) # expand the front dimension into restored shape

        imap = np.moveaxis(imap, range(len(axis)), axis) # put the front restored axes into correct positions
        return imap

def atleast_nd(arr, n, axis=None):
    arr = np.asanyarray(arr)
    if (axis is None) or (arr.ndim >= n):
        oaxis=tuple(range(n - arr.ndim)) # prepend the dims or do nothing in n < arr.ndim
    else:
        axis = np.atleast_1d(axis)
        assert (n - arr.ndim) >= len(axis), 'More axes than dimensions to add'
        oaxis = tuple(range(n - arr.ndim - len(axis))) + tuple(axis) # prepend the extra dims
    return np.expand_dims(arr, oaxis)

def triu_indices(N):
    """Gives the upper triangular indices of an NxN matrix in diagonal-major order
    (compatible with healpy spectra ordering with new=True)

    Parameters
    ----------
    N : int
        Side-length of matrix

    Returns
    -------
    tuple of ndarray
        A row array and column array of ints, such that if arr is NxN, then
        arr[triu_indices(N)] gives the 1D elements of the upper triangle of arr
        in diagonal-major order.
    """
    num_idxs = N*(N+1)//2
    rows = []
    cols = []
    rowoffset = 0
    coloffset = 0
    for i in range(num_idxs):
        rownum = i - rowoffset
        colnum = rownum + coloffset
        rows.append(rownum)
        cols.append(colnum)
        if colnum >= N-1: # N-1 is last column
            rowoffset = i+1 # restart rows next iteration
            coloffset += 1 # cols start one higher
    return np.array(rows), np.array(cols)

def triu_indices_1d(N):
    idxs = np.arange(N**2).reshape(N, N)
    return idxs[triu_indices(N)]

def is_triangular(i):
    return np.roots((1, 1, -2*i))[1].is_integer()

def triangular(N):
    return N*(N+1)//2

def triangular_idx(i):
    assert is_triangular(i), 'Arg must be a triangular number'
    return np.roots((1, 1, -2*i)).astype(int)[1]

def triu_pos(i, N):
    return np.stack(triu_indices(N))[:, i]

def triu_to_symm(arr, copy=False, axis1=0, axis2=1):
    assert arr.shape[axis1] == arr.shape[axis2], 'Axes of matrix must at axis1, axis2 dimension must have same size'
    if copy:
        arr = arr.copy()
    ncomp = arr.shape[axis1]
    for i in range(ncomp):
        for j in range(i+1, ncomp): # i+1 to only do upper triangle, not diagonal
            getslice = get_take_indexing_obj(arr, [[i], [j]], axis=(axis1, axis2))
            setslice = get_take_indexing_obj(arr, [[j], [i]], axis=(axis1, axis2))
            arr[setslice] = arr[getslice]
    return arr

# we can collapse a symmetric 2D matrix whose first axis is passed as the arg
# into a 1D list of the upper triangular components.
# this will clobber if axis1 or axis2 is in map axes, see __array_wrap__
def to_flat_triu(arr, axis1=0, axis2=None, flat_triu_axis=None):
    # force axes to be positive
    axis1 = axis1%arr.ndim
    if axis2 is None:
        axis2 = axis1+1
    axis2 = axis2%arr.ndim

    assert axis1 < axis2
    assert arr.shape[axis1] == arr.shape[axis2], 'Axes of matrix at axis1, axis2 dimension must have same size'
    ncomp = arr.shape[axis1]

    # put flattened dims at flat_triu_axis position
    if flat_triu_axis is None:
        flat_triu_axis = axis1
    arr = flatten_axis(arr, axis=(axis1, axis2), pos=flat_triu_axis)

    # get triangular indices for axis to be expanded
    slice_obj = get_take_indexing_obj(arr, triu_indices_1d(ncomp), axis=flat_triu_axis)
    return arr[slice_obj]

# this clobbers ndarray subclasses because of call to append!
# axis2 is position in expanded array, while flat_triu_axis is position in passed array
def from_flat_triu(arr, axis1=0, axis2=None, flat_triu_axis=None):
    if flat_triu_axis is None:
        flat_triu_axis = axis1
    
    # force axes to be positive
    axis1 = axis1%arr.ndim
    if axis2 is None:
        axis2 = axis1+1
    axis2 = axis2%(arr.ndim+1)

    assert axis1 < axis2
    assert is_triangular(arr.shape[flat_triu_axis]), 'flat_triu_axis length must be a triangular number'

    # place in new array with ncomp**2 instead of n_triu at flat_triu_axis
    ncomp = triangular_idx(arr.shape[flat_triu_axis])
    ishape = list(arr.shape)
    ishape[flat_triu_axis] = ncomp**2 - triangular(ncomp) # this will be concatenated to arr
    arr = np.append(arr, np.zeros(ishape), axis=flat_triu_axis).astype(arr.dtype)

    # get triangular indices for axis to be expanded
    getslice = get_take_indexing_obj(arr, np.arange(triangular(ncomp)), axis=flat_triu_axis)
    setslice = get_take_indexing_obj(arr, triu_indices_1d(ncomp), axis=flat_triu_axis)
    arr[setslice] = arr[getslice]

    # reshape into correct shape
    arr = unflatten_axis(arr, (ncomp, ncomp), axis=(axis1, axis2), pos=flat_triu_axis)
    return triu_to_symm(arr, axis1=axis1, axis2=axis2)

# get a logical mask that can be used to index an array, eg to build a mask
# out of conditions.
# if not keep_prepend_dims, perform op over all dims up to map dims; else
# perform over specified axes
def get_logical_mask(cond, op=np.logical_or, keep_prepend_dims=False, axis=0):
    if not keep_prepend_dims:
        axis = tuple(range(cond.ndim - 2))
    return op.reduce(cond, axis=axis)

def get_coadd_map(imap, ivar):
    imap = atleast_nd(imap, 4) # make 4d by prepending
    ivar = atleast_nd(ivar, 4)

    # due to floating point precision, the coadd is not exactly the same
    # as a split where that split is the only non-zero ivar in that pixel
    coadd = np.nan_to_num( 
        np.sum(imap * ivar, axis=-4, keepdims=True) / np.sum(ivar, axis=-4, keepdims=True) # splits along -4 axis
        )

    # find pixels where exactly one split has a nonzero ivar
    single_nonzero_ivar_mask = np.broadcast_to(np.sum(ivar!=0, axis=-4) == 1, coadd.shape)
    
    # set the coadd in those pixels to be equal to the imap value of that split (ie, avoid floating
    # point errors in naive coadd calculation)
    coadd[single_nonzero_ivar_mask] = np.sum(imap * (ivar!=0), axis=-4, keepdims=True)[single_nonzero_ivar_mask]
    return coadd

def get_ivar_eff(ivar, use_inf=False):
    """
    Return ivar_eff = 1 / (1 / ivar - 1 / sum_ivar), 
    where sum_ivar is the sum over splits.
    
    Parameters
    ----------
    ivar : (..., nsplit, 1, ny, nx) enmap
        Inverse variance maps for N splits.
    use_inf : bool, optional
        If set, use np.inf for values that approach infinity, 
        instead of large numerical values.
    
    Returns
    -------
    ivar_eff : (..., nsplit, 1, ny, nx) enmap
        Ivar_eff for each split.
    """

    # Make 4d by prepending splits along -4 axis.
    ivar = atleast_nd(ivar, 4) 

    # We want to calculate 1 / (1/ivar - 1/sum(ivar). It easier to do 
    # ivar * sum(ivar) / (sum(ivar) - ivar) to avoid (some) divisions by zero.
    sum_ivar = np.sum(ivar, axis=-4, keepdims=True)
    num = sum_ivar * ivar # Numerator.
    den = sum_ivar - ivar # Denominator.

    # In pixels were ivar == sum_ivar we get inf.
    mask = den != 0 
    out = np.divide(num, den, where=mask, out=num)

    if use_inf:
        out[~mask] = np.inf
    else:
        # Fill with largest value allowed by dtype to mimic np.nan_to_num.
        out[~mask] = np.finfo(out.dtype).max

    return out

def get_corr_fact(ivar):
    '''
    Get correction factor sqrt(ivar_eff / ivar) that converts a draw from 
    split difference d_i to a draw from split noise n_i.

    Parameters
    ----------
    ivar : (..., nsplit, 1, ny, nx) enmap
        Inverse variance maps for N splits.

    Returns
    -------
    corr_fact : (..., nsplit, 1, ny, nx) enmap
        Correction factor for each split.
    '''

    corr_fact = get_ivar_eff(ivar, use_inf=True)
    corr_fact[~np.isfinite(corr_fact)] = 0
    np.divide(corr_fact, ivar, out=corr_fact, where=ivar!=0)
    corr_fact[ivar==0] = 0
    corr_fact **= 0.5

    return corr_fact

def get_noise_map(imap, ivar):
    return imap - get_coadd_map(imap, ivar)

def get_whitened_noise_map(imap, ivar):
    return get_noise_map(imap, ivar)*np.sqrt(get_ivar_eff(ivar))

def rolling_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def linear_crossfade(cNy,cNx,npix_y,npix_x=None, dtype=np.float32):
    if npix_x is None:
        npix_x = npix_y
    fys = np.ones(cNy, dtype=dtype)
    fxs = np.ones(cNx, dtype=dtype)
    
    fys[cNy-npix_y:] = np.linspace(0.,1.,npix_y)[::-1]
    fys[:npix_y] = np.linspace(0.,1.,npix_y)
        
    fxs[:npix_x] = np.linspace(0.,1.,npix_x)
    fxs[cNx-npix_x:] = np.linspace(0.,1.,npix_x)[::-1]
    return fys[:,None] * fxs[None,:]

# based on enmap.lbin but handles arbitrary ell bins and weights
def radial_bin(smap, rmap, bins, weights=None):
    """Go from a set of 2D maps to a their radially-binned sum, with bin edges
    given by the sequence "bins".

    Parameters
    ----------
    smap : array-like
        A maps to bin radially along last two axes, with any prepended shape
    rmap : array-like
        A map of of shape smap.shape[-2:] that gives radial positions
    bins : iterable
        A sequence of bin edges
    weights : array-like or callable
        An array of weights to apply to the smap, if we want to calculate 
        a weighted sum. Must be broadcastable with smap's shape. The default is
        None, in which case weights of 1 are applied to each pixel. 

        If callable, will be applied to the rmap before the result is used
        in the sum.

    Returns
    -------
    array-like
        The radially-binned sum of smap, with the same prepended shape as smap,
        and whose last dimension is the number of bins.

    Notes
    -----
    bins must be monotonically increasing and positive semi-definite. The 
    r=0 mode is never included even if bins[0]=0, since this function uses
    np.bincount(..., right=True)
    """
    # make sure there is at least one prepended dim to smap
    smap = atleast_nd(smap, 3)
    assert rmap.ndim == 2, 'rmap must have two dimensions'

    # get prepended smap shape
    preshape = smap.shape[:-2]

    # prepare bin edges
    bins = np.atleast_1d(bins)
    assert len(bins) > 1
    assert bins.ndim == 1
    nbins = len(bins)-1
    assert np.min(bins) >= 0, 'Do not include r=0 pixel in any bin'

    # prepare weights
    if weights is None:
        weights = np.array([1]) # same prepended shape as smap but just a singleton one
    elif callable(weights):
        weights = weights(rmap)
    weights = np.broadcast_to(weights, preshape + weights.shape, subok=True)

    # "flatten" rmap, smap, and weights; we will reshape everything at the end
    rmap = rmap.reshape(-1)
    bin_rmaps = np.digitize(rmap, bins, right=True) # the bin of each pixel
    smap = smap.reshape(np.prod(preshape), -1)
    weights = weights.reshape(np.prod(preshape), -1)

    # iterate through all smaps to be ginned and do a weighted sum by 
    # number of pixels within ell bins, with an optional weight map.
    # [1:nbins+1] index because 0'th bin is always empty or not needed with right=True in 
    # digitize and don't need data beyond last bin, if any
    omap = enmap.zeros((len(smap), nbins), wcs=smap.wcs)
    for i in range(len(smap)):
        bin_count = np.bincount(bin_rmaps, weights=np.broadcast_to(weights[i], bin_rmaps.shape), minlength=nbins+1)[1:nbins+1] 
        omap[i] = np.bincount(bin_rmaps, weights=weights[i]*smap[i], minlength=nbins+1)[1:nbins+1]
        omap[i] = np.nan_to_num(omap[i] / bin_count)

    # same shape as spectra, with full 2D power replaced with binned spectra
    return omap.reshape(*preshape, -1)

def interp1d_bins(bins, y, return_vals=False, **interp1d_kwargs):
    # prepare x values
    bins = np.atleast_1d(bins)
    assert len(bins) > 1
    assert bins.ndim == 1
    x = (bins[1:] + bins[:-1])/2
    fill_value = (y[0], y[-1])

    if return_vals:
        return interp1d(x, y, fill_value=fill_value, **interp1d_kwargs), y
    else:
        return interp1d(x, y, fill_value=fill_value, **interp1d_kwargs)

# forces shape to (num_arrays, num_splits, num_pol, ny, nx) and sums over num_splits
def ell_flatten(imap, mask=None, return_cov=False, mode='fft', lmax=6000, ledges=None, weights=None, nthread=0):
    imap = atleast_nd(imap, 5)
    num_arrays, num_splits, num_pol = imap.shape[:3]
    if mask is None:
        mask = 1
    if mode == 'fft':
        kmap = enmap.fft(imap*mask, normalize='phys', nthread=nthread)
        smap = enmap.enmap(np.einsum('...miayx,...miayx->...mayx', kmap, np.conj(kmap)).real, wcs=kmap.wcs)
        w2 = np.mean(mask**2)
        smap /= num_splits*w2

        modlmap = smap.modlmap().astype(imap.dtype) # modlmap is np.float64 always...
        smap = radial_bin(smap, modlmap, ledges, weights=weights)
        ys = []
        for i in range(num_arrays):
            for j in range(num_pol):
                lfunc, y = interp1d_bins(ledges, smap[i, j], return_vals=True, kind='cubic', bounds_error=False)
                ys.append(y)
                lfilter = 1/np.sqrt(lfunc(modlmap))
                assert np.all(np.isfinite(lfilter)), f'{i,j}'
                assert np.all(lfilter > 0)
                kmap[i, :, j] *= lfilter
        ys = np.array(ys)
        ys = ys.reshape(num_arrays, num_pol, -1)

        if return_cov:
            return enmap.ifft(kmap, normalize='phys', nthread=nthread).real, ys
        else:
            return enmap.ifft(kmap, normalize='phys', nthread=nthread).real
    else:
        raise NotImplementedError('mode=curvedsky not yet implemented')

def lmax_from_wcs(imap):
    return int(180/imap.wcs.wcs.cdelt[1])

# copied from pixell online, not in tigress pixell version.
# further extended here for ffts
def ell_filter(imap, lfunc, mode='fft', ainfo=None, lmax=None, nthread=0):
    """Filter a map isotropically by a function.
    Returns alm2map(map2alm(alm * lfunc(ell),lmax))
    Args:
        imap: (...,Ny,Nx) ndmap stack of enmaps.
        lmax: integer specifying maximum multipole beyond which the alms are zeroed
        lfunc: a function mapping multipole ell to the filtering expression
        ainfo: 	If ainfo is provided, it is an alm_info describing the layout 
    of the input alm. Otherwise it will be inferred from the alm itself.
    Returns:
        omap: (...,Ny,Nx) ndmap stack of filtered enmaps
    """
    if mode == 'fft':
        kmap = enmap.fft(imap, nthread=nthread)
        filterlmap = lfunc(imap.modlmap().astype(imap.dtype)) # modlmap is np.float64 always...
        return enmap.ifft(kmap * filterlmap, nthread=nthread).real
    elif mode == 'curvedsky':
        if lmax is None:
            lmax = lmax_from_wcs(imap)
        alm = curvedsky.map2alm(imap, ainfo=ainfo, lmax=lmax, spin=0)
        alm = curvedsky.almxfl(alm, lfunc=lfunc, ainfo=ainfo)
        return curvedsky.alm2map(alm, enmap.empty(imap.shape, imap.wcs, dtype=imap.dtype), spin=0, ainfo=ainfo)

def get_ell_linear_transition_funcs(center, width, dtype=np.float32):
    lmin = center - width/2
    lmax = center + width/2
    def lfunc1(ell):
        ell = np.asarray(ell, dtype=dtype)
        condlist = [ell < lmin, np.logical_and(lmin <= ell, ell < lmax), lmax <= ell]
        funclist = [lambda x: 1, lambda x: -(x - lmin)/(lmax - lmin) + 1, lambda x: 0]
        return np.piecewise(ell, condlist, funclist)
    def lfunc2(ell):
        ell = np.asarray(ell, dtype=dtype)
        condlist = [ell < lmin, np.logical_and(lmin <= ell, ell < lmax), lmax <= ell]
        funclist = [lambda x: 0, lambda x: (x - lmin)/(lmax - lmin), lambda x: 1]
        return np.piecewise(ell, condlist, funclist)
    return lfunc1, lfunc2
    
class _SeedTracker(object):
    def __init__(self):
        self.CMB     = 0
        self.FG      = 1
        self.PHI     = 2
        self.NOISE   = 3
        self.POISSON = 4
        self.COMPTONY = 5
        self.TILED_NOISE = 6

        #quick-srcfree is maxmally correlated with 15mJy sims
        self.fgdict  = {'15mjy': 0, '100mjy': 1, 'srcfree': 2, 'quick-srcfree':0,'comptony': 3}

        self.dmdict  = {'act_mr3':0,'act_c7v5':1,'planck_hybrid':2,'dr5':3}

    def get_cmb_seed(self, set_idx, sim_idx):
        return (set_idx, 0, self.CMB, sim_idx)

    def get_fg_seed(self, set_idx, sim_idx, fg_type):
        assert(fg_type in self.fgdict.keys())
        return (set_idx, 0, self.FG, sim_idx, self.fgdict[fg_type])

    def get_phi_seed(self, set_idx, sim_idx):
        return (set_idx, 0, self.PHI, sim_idx)

    def get_noise_seed(self, set_idx, sim_idx, data_model, season, patch, array, patch_id=None):
        ret = (set_idx, 0, self.NOISE, sim_idx)
        dm  = data_model
        
        assert(dm.name in self.dmdict.keys())
        sid =  dm.seasons.index(season) if dm.seasons is not None else 0
        pid =  dm.patches.index(patch) if dm.patches is not None else 0
        aid = list(dm.array_freqs.keys()).index(array)
        ret = ret + (self.dmdict[dm.name],sid,pid,aid)
        if patch_id is not None:
            ret = ret + (patch_id,)
        return ret

    def get_poisson_seed(self, set_idx, sim_idx):
        return (set_idx, 0, self.POISSON, sim_idx)

    def get_tiled_noise_seed(self, set_idx, sim_idx, data_model, qid, tile_idx, lowell_seed=False):
        """Return a seed for a tile in a tiled noise simulation scheme. Allows consistent
        seeding for a given simulation set, map number, data model, qid (array), and tile
        number, across users and platforms.

        Parameters
        ----------
        set_idx : int
        sim_idx : int
        data_model : object
            A soapack.interfaces DataModel object
        qid : str or iterable of str
            If simulating 1 array, can pass 1 string or iterable of type string
            and length 1. If simulating correlation between 2 arrays, pass iterable of
            type str and length 2. Iterables are sorted, so order does not matter. Cannot
            correlate more than 2 arrays. 
        tile_idx : int
        lowell_seed : bool
            If two tiling schemes are building one sim, you don't want correlated tiles.
            The second integer in the seed tuple will be 1 if True.

        Returns
        -------
        tuple of int
            Seed to be passed to np.random.seed

        Example
        -------
        >>> from actsims import util as u
        >>> u.seed_tracker.get_tiled_noise_seed(3,963,u.dmint.DR5(),'s18_03',7_034)
        >>> (3, 0, 6, 963, 3, 8326, 0, 7034)
        >>> u.seed_tracker.get_tiled_noise_seed(3,963,u.dmint.DR5(),['s18_04','s18_03'],7_034)
        >>> (3, 0, 6, 963, 3, 8326, 2839, 7034)

        """
        ret = (set_idx, int(lowell_seed), self.TILED_NOISE, sim_idx)
        dm = data_model
        qid = np.sort(np.atleast_1d(qid)) # sorted qids
        assert len(qid) <= 2, f'Can only seed for correlation of up to 2 arrays; {len(qid)} passed'

        assert(dm.name in self.dmdict.keys())
        dm_idx = self.dmdict[dm.name]
        
        if len(qid) == 1:
            # qid_idx = (dmint.arrays(qid[0], 'hash'), 0)
            qid_idx = (dmint.get_all_dr5_qids().index(qid[0]), 0)
        else:
            # qid_idx = tuple(dmint.arrays(q, 'hash') for q in qid)
            qid_idx = tuple(dmint.get_all_dr5_qids().index(q) for q in qid)
        return ret + (dm_idx,) + qid_idx + (tile_idx,)
seed_tracker = _SeedTracker()

### OLD ###

def bin(data, modlmap, bin_edges=25):
    """
    A function used to the bin 2d power spectra to 1d ones.
    """
    digitized = np.digitize(np.ndarray.flatten(modlmap), bin_edges, right=True)
    return np.bincount(digitized, (data).reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]

def gen_coadd_map_old(map_list, ivar_list, a):
    """return coadded map from splits, the map in maplist contains I,Q,U 
    a=0,1,2 selects one of I Q U """
    map_list = np.array(map_list)
    ivar_list = np.array(ivar_list)
    coadd_map = np.sum(map_list[:, a] * ivar_list, axis=0)
    coadd_map /= np.sum(ivar_list, axis=0)
    coadd_map[~np.isfinite(coadd_map)] = 0

    return enmap.samewcs(coadd_map, map_list[0])

def get_ivar_eff_old(split, ivar_list):
    """return effective invers variance map for split i and a list of inverse variance maps.
    Inputs
    splits:integer
    ivar_list:list
    Output
    ndmap with same shape and wcs as individual inverse variance maps.
    """
    ivar_list = np.array(ivar_list)
    h_c = np.sum(ivar_list, axis=0)
    # w=h_c-ivar_list[split]
    weight = 1/(1/ivar_list[split]-1/h_c)
    weight[~np.isfinite(weight)] = 0
    weight[weight < 0] = 0
    return enmap.samewcs(weight, ivar_list[0])

def get_noise_maps(map_list, ivar_list, N=20):
    """ 
    Get the noise splits from the total map.
    """

    n = len(map_list)
    # calculate the coadd maps
    noise = enmap.samewcs(np.zeros(np.shape(map_list)), map_list[0][0])

    # iterate over IQU
    for index in range(len(map_list[0])):
        coadd = gen_coadd_map_old(map_list, ivar_list, index)

        # iterate over split
        for i in range(n):
            print(i, index)
            noise[i][index] = map_list[i][index]-coadd  # [i]
    return noise

def get_whitened_noise_maps(map_list, ivar_list, N=20):
    """
    Go from splits to a whitened noise map

    """
    # pmap=enmap.pixsizemap(map_list[0].shape,map_list[0].wcs)

    n = len(map_list)
    # calculate the coadd maps

    noise = enmap.samewcs(np.zeros(np.shape(map_list)), map_list[0][0])

    # iterate over IQU
    for index in range(len(map_list[0])):
        coadd = gen_coadd_map_old(map_list, ivar_list, index)

        # iterate over split
        for i in range(n):
            print(i, index)
            d = map_list[i][index]-coadd  # [i]
            noise[i][index] = d*np.sqrt(get_ivar_eff_old(i, ivar_list))
    return noise

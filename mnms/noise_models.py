from mnms import simio, tiled_ndmap, utils, soapack_utils as s_utils, tiled_noise, wav_noise, fsaw_noise, inpaint
from pixell import enmap, wcsutils
from enlib import bench
from optweight import wavtrans

import numpy as np

from abc import ABC, abstractmethod

# expose only concrete noise models, helpful for namespace management in client
# package development. NOTE: this design pattern inspired by the super-helpful
# registry trick here: https://numpy.org/doc/stable/user/basics.dispatch.html

REGISTERED_NOISE_MODELS = {}

def register(registry=REGISTERED_NOISE_MODELS):
    """Add a concrete NoiseModel implementation to the specified registry (dictionary)."""
    def decorator(noise_model_class):
        registry[noise_model_class.__name__] = noise_model_class
        return noise_model_class
    return decorator


# NoiseModel API and concrete NoiseModel classes. 
class NoiseModel(ABC):

    def __init__(self, *qids, data_model=None, preload=True, ivar=None, mask_est=None,
                calibrated=True, downgrade=1, lmax=None, mask_version=None, mask_name=None,
                union_sources=None, kfilt_lbounds=None, fwhm_ivar=None, notes=None, dtype=None,
                **kwargs):
        """Base class for all NoiseModel subclasses. Supports loading raw data necessary for all 
        subclasses, such as masks and ivars. Also defines some class methods usable in subclasses.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model : soapack.DataModel, optional
            DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
        preload: bool, optional
            If ivar is None, load them ivar maps for each qid, by default True. Setting
            to False can expedite access to helper methods of this class without waiting
            for ivar to load from disk.
        ivar : array-like, optional
            Data inverse-variance maps, by default None. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations. If None
            and preload, will be loaded via DataModel according to 'downgrade' and
            'calibrated' kwargs.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            to whiten the data before estimating its variance, by default None. If
            provided, assumed properly downgraded into compatible wcs with internal
            NoiseModel operations. If None and preload, will load a mask according to 
            the 'mask_version' and 'mask_name' kwargs.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default True.
        downgrade : int, optional
            The factor to downgrade map pixels by, by default 1.
        lmax : int, optional
            The bandlimit of the maps, by default None.
            If None, will be set to twice the theoretical CAR limit, ie 180/wcs.wcs.cdelt[1].
        mask_version : str, optional
           The mask version folder name, by default None.
           If None, will first look in config 'mnms' block, then block of default data model.
        mask_name : str, optional
            Name of mask file, by default None.
            If None, a default mask will be loaded from disk.
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.
        """

        # if mask and ivar are provided, there is no way of checking
        # whether calibrated, what downgrade, etc; they are assumed to be 'correct'

        # store basic set of instance properties
        self._qids = qids
        if data_model is None:
            data_model = utils.get_default_data_model()
        self._data_model = data_model
        self._calibrated = calibrated
        self._downgrade = downgrade
        if mask_version is None:
            mask_version = utils.get_default_mask_version()
        self._mask_version = mask_version
        self._mask_name = mask_name
        self._notes = notes
        self._union_sources = union_sources
        if kfilt_lbounds is not None:
            kfilt_lbounds = np.array(kfilt_lbounds).reshape(2)
        self._kfilt_lbounds = kfilt_lbounds
        self._fwhm_ivar = fwhm_ivar
        self._kwargs = kwargs
        self._dtype = dtype if dtype is not None else self._data_model.dtype

        # get derived instance properties
        self._num_arrays = len(self._qids)
        self._num_splits = utils.get_nsplits_by_qid(self._qids[0], self._data_model)
        self._use_default_mask = mask_name is None

        # Get shape, wcs, ivars, and mask_observed -- need these for every operation.
        full_shape, full_wcs = self._check_geometry()
        self._shape, self._wcs = utils.downgrade_geometry_cc_quad(
            full_shape, full_wcs, self._downgrade
            )

        if ivar is not None:
            self._ivar = ivar
            self._mask_observed = utils.get_bool_mask_from_ivar(self._ivar)
        elif preload:            
            self._ivar, self._mask_observed = self._get_ivar_and_mask_observed()
        else:
            raise ValueError('Models require ivar, please preload or supply manually')
        
        # Possibly store input data
        self._mask_est = mask_est

        # initialize unloaded difference maps
        self._dmap = None

        # initialize unloaded noise model dictionary, holds noise model variables for each split
        self._nm_dict = {}

        # get lmax
        if lmax is None:
            lmax = utils.lmax_from_wcs(self._wcs)
        self._lmax = lmax

        # sanity checks
        if self._ivar is not None:
            assert self._num_splits == self._ivar.shape[-4], \
                'Num_splits inferred from ivar shape != num_splits from data model table'

    def _check_geometry(self, return_geometry=True):
        """Check that each qid in this instance's qids has compatible shape and wcs."""
        for i, qid in enumerate(self._qids):
            # Load up first geometry of first split's ivar. Only need pixel shape.
            shape, wcs = s_utils.read_map_geometry(self._data_model, qid, 0, ivar=True)
            shape = shape[-2:]
            assert len(shape) == 2, 'shape must have only 2 dimensions'

            # Check that we are using the geometry for each qid -- this is required!
            if i == 0:
                main_shape, main_wcs = shape, wcs
            else:
                with bench.show(f'Checking geometry compatibility between {qid} and {self._qids[0]}'):
                    assert(
                        shape == main_shape), 'qids do not share pixel shape -- this is required!'
                    assert wcsutils.is_compatible(
                        wcs, main_wcs), 'qids do not share a common wcs -- this is required!'
        
        if return_geometry:
            return main_shape, main_wcs
        else:
            return None

    def _get_ivar_and_mask_observed(self):
        """Load the inverse-variance maps according to instance attributes, and use it
        to construct and observed-by-all-splits pixel map.

        Returns
        -------
        ivars, mask_observed : (nmaps, nsplits, npol, ny, nx) enmap, (ny, nx) enmap
            Inverse-variance maps, possibly downgraded. Observed pixel map,
            possibly downgraded.
        """

        # first check for ivar compatibility and get map geometry
        full_shape, full_wcs = self._check_geometry()

        # allocate a buffer to accumulate all ivar maps in.
        # this has shape (nmaps, nsplits, 1, ny, nx).
        ivars = self._empty(ivar=True)
        mask_observed = np.ones(full_shape, dtype=bool)

        for i, qid in enumerate(self._qids):
            with bench.show(self._action_str(qid, ivar=True)):
                if self._calibrated:
                    mul = s_utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                for j in range(self._num_splits):
                    ivar = s_utils.read_map(self._data_model, qid, j, ivar=True)
                    ivar *= mul

                    # iteratively build the mask_observed at full resolution, 
                    # loop over leading dims
                    for idx in np.ndindex(*ivar.shape[:-2]):
                        mask_observed *= ivar[idx].astype(bool)
                    
                    if self._downgrade != 1:
                        # use harmonic instead of interpolated downgrade because it is 
                        # 10x faster
                        ivar = utils.fourier_downgrade_cc_quad(
                            ivar, self._downgrade, area_pow=1
                            )
                    
                    # zero-out any numerical negative ivar
                    ivar[ivar < 0] = 0                    
                    
                    if self._fwhm_ivar:
                        mask_good = ivar != 0
                        inpaint.inpaint_median(ivar, mask_good, inplace=True)
                        utils.smooth_gauss(ivar, np.radians(self._fwhm_ivar), inplace=True)
                        ivar *= mask_good

                    ivars[i,j] = ivar

        with bench.show('Generating observed-pixels mask'):
            mask_observed = enmap.enmap(mask_observed, wcs=full_wcs, copy=False)
            mask_observed = utils.interpol_downgrade_cc_quad(
                mask_observed, self._downgrade, dtype=self._dtype
                )

            # define downgraded mask_observed to be True only where the interpolated 
            # downgrade is all 1 -- this is the most conservative route in terms of 
            # excluding pixels that may not actually have nonzero ivar or data
            mask_observed = utils.get_mask_bool(mask_observed, threshold=1.)

            # finally, need to layer on any ivars that may still be 0 that aren't yet
            # masked
            mask_observed *= utils.get_bool_mask_from_ivar(ivars)
        
        return ivars*mask_observed, mask_observed

    def _empty(self, ivar=False, num_arrays=None, num_splits=None,
                shape=None, wcs=None):
        """Allocate an empty buffer that will broadcast against the Noise Model 
        number of arrays, number of splits, and the map (or ivar) shape.

        Parameters
        ----------
        ivar : bool, optional
            If True, load the inverse-variance map shape for the qid and
            split. If False, load the source-free map shape for the same,
            by default False.
        num_arrays : int, optional
            The number of arrays (axis -5) in the empty ndmap, by default None.
            If None, inferred from the number of qids in the NoiseModel.
        num_splits : int, optional
            The number of splits (axis -4) in the empty ndmap, by default None.
            If None, inferred from the number of splits on disk.
        shape : tuple, optional
            A geometry footprint shape to use to build the empty ndmap, by
            default None. If None, will use the downgraded geometry inferred
            from the data on-disk.
        wcs : astropy.wcs.WCS
            A geometry wcs to use to build the empty ndmap, by default None. 
            If None, will use the downgraded geometry inferred from the data
            on-disk.

        Returns
        -------
        enmap.ndmap
            An empty ndmap with shape (num_arrays, num_splits, num_pol, ny, nx),
            with dtype of the instance soapack.DataModel. If ivar is True, num_pol
            likely is 1. If ivar is False, num_pol likely is 3.
        """
        # read geometry from the map to be loaded. we really just need the first component,
        # a.k.a "npol", which varies depending on if ivar is True or False
        if shape is not None:
            footprint_shape = shape[-2:]
        else:
            footprint_shape = self._shape

        if wcs is not None:
            footprint_wcs = wcs
        else:
            footprint_wcs = self._wcs

        shape, _ = s_utils.read_map_geometry(self._data_model, self._qids[0], 0, ivar=ivar)
        shape = (shape[0], *footprint_shape)

        if num_arrays is None:
            num_arrays = self._num_arrays
        if num_splits is None:
            num_splits = self._num_splits

        shape = (num_arrays, num_splits, *shape)
        return enmap.empty(shape, wcs=footprint_wcs, dtype=self._dtype)

    def _action_str(self, qid, ivar=False):
        """Get a string for benchmarking the loading step of a map product.

        Parameters
        ----------
        qid : str
            Map identification string
        ivar : bool, optional
            If True, print 'ivar' where appropriate. If False, print 'imap'
            where appropriate, by default False.

        Returns
        -------
        str
            Benchmarking action string.

        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> tnm = nm.TiledNoiseModel('s18_03', downgrade=2, union_sources='20210209_sncut_10_aggressive', notes='my_model')
        >>> tnm._action_str('s18_03')
        >>> 'Loading, inpainting, downgrading imap for s18_03'
        """
        ostr = 'Loading'
        if ivar:
            if self._downgrade != 1:
                ostr += ', downgrading'
            ostr += f' ivar for {qid}'
        else:
            if self._union_sources:
                ostr += ', inpainting'
            if self._kfilt_lbounds is not None:
                ostr += ', kspace filtering'
            if self._downgrade != 1:
                ostr += ', downgrading'
            ostr += f' imap for {qid}'
        return ostr

    def get_model(self, check_on_disk=True, write=True, keep_model=False, keep_data=False, 
                  verbose=False, **kwargs):
        """Generate (or load) a sqrt-covariance matrix for this NoiseModel instance.

        Parameters
        ----------
        check_on_disk : bool, optional
            If True, first check if an identical model (including by 'notes') exists
            on-disk for each split. If it does, do nothing or store it in the object
            attributes, depending on the 'keep_model' kwarg. If it does not, generate the model
            for the missing splits instead. By default True.
        write : bool, optional
            Save the generated model to disk, by default True.
        keep_model : bool, optional
            Store the generated (or loaded) model in the instance attributes, by 
            default False.
        keep_data: bool, optional
            Store the loaded, possibly downgraded, data split differences in the
            instance attributes, by default False.
        verbose : bool, optional
            Print possibly helpful messages, by default False.
        """
        if check_on_disk:
            # build a list of splits that don't have models on-disk
            does_not_exist = []
            for s in range(self._num_splits):
                res = self._check_model_on_disk(s, keep_model=keep_model)
                if not res:
                    does_not_exist.append(s)
            if not does_not_exist:
                # if all models exist on-disk, exit this function
                return
        else:
            does_not_exist = range(self._num_splits)

        # models need data split differences as inputs
        if self._dmap is None:
            dmap = self._get_data_split_diffs()
            if keep_data:
                self._dmap = dmap
        else:
            dmap = self._dmap

        # particular model subclasses may further modify data split differences,
        # e.g. with correction factors or ivar weighting 
        dmap = self._get_dmap(dmap)

        # get the conservative mask for estimating the harmonic filter used to whiten
        # the difference maps
        if self._mask_est is None:
            self._mask_est = self._get_mask_est(min_threshold=1e-4)

        for s in does_not_exist:
            with bench.show(f'Generating noise model for split {s}'):
                nm_dict = self._get_model(s, dmap, verbose=verbose)

            if keep_model:
                self._keep_model(s, nm_dict)

            if write:
                fn = self._get_model_fn(s)
                self._write_model(fn, **nm_dict)

            # remove nm_dict from memory before moving onto next split
            nm_dict = None

    def _get_data_split_diffs(self):
        """Load the raw data split differences according to instance attributes.

        Returns
        -------
        dmaps : (nmaps, nsplits, npol, ny, nx) enmap
            Data split difference maps, possibly downgraded.
        """
        # first check for mask compatibility and get map geometry
        full_shape, full_wcs = self._check_geometry()

        # allocate a buffer to accumulate all difference maps in.
        # this has shape (nmaps, nsplits, npol, ny, nx).
        dmaps = self._empty(ivar=False)

        # all filtering operations use the same filter
        if self._kfilt_lbounds is not None:
            filt = utils.build_filter(
                full_shape, full_wcs, self._kfilt_lbounds, self._dtype
                )
    
        for i, qid in enumerate(self._qids):
            with bench.show(self._action_str(qid, ivar=False)):
                if self._calibrated:
                    mul_imap = s_utils.get_mult_fact(self._data_model, qid, ivar=False)
                    mul_ivar = s_utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul_imap = 1
                    mul_ivar = 1

                # get the coadd from disk, this is the same for all splits
                cmap = s_utils.read_map(self._data_model, qid, coadd=True, ivar=False)
                cmap = enmap.extract(cmap, full_shape, full_wcs) 
                cmap *= mul_imap

                # need full-res coadd ivar if kspace filtering
                if self._kfilt_lbounds is not None:
                    cvar = s_utils.read_map(self._data_model, qid, coadd=True, ivar=True)
                    cvar = enmap.extract(cvar, full_shape, full_wcs)
                    cvar *= mul_ivar

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                for j in range(self._num_splits):
                    imap = s_utils.read_map(self._data_model, qid, j, ivar=False)
                    imap = enmap.extract(imap, full_shape, full_wcs) 
                    imap *= mul_imap

                    # need to reload ivar at full res if inpainting or kspace filtering
                    if self._union_sources or self._kfilt_lbounds:
                        if self._downgrade != 1:
                            ivar = s_utils.read_map(self._data_model, qid, j, ivar=True)
                            ivar = enmap.extract(ivar, full_shape, full_wcs)
                            ivar *= mul_ivar
                        else:
                            ivar = self._ivar[i, j]

                    if self._union_sources:
                        # the boolean mask for this array, split, is non-zero ivar.
                        # iteratively build the boolean mask at full resolution, 
                        # loop over leading dims. this really should be a singleton
                        # leading dim!
                        mask_bool = np.ones(full_shape, dtype=bool)
                        for idx in np.ndindex(*ivar.shape[:-2]):
                            mask_bool *= ivar[idx].astype(bool)
                            
                        self._inpaint(imap, ivar, mask_bool, qid=qid, split_num=j) 

                    dmap = imap - cmap

                    if self._kfilt_lbounds is not None:
                        ivar_eff = utils.get_ivar_eff(ivar, sum_ivar=cvar, use_zero=True)
                        dmap = utils.filter_weighted(dmap, ivar_eff, filt)

                    if self._downgrade != 1:
                        dmaps[i, j] = utils.fourier_downgrade_cc_quad(
                            dmap, self._downgrade
                        )
                    else:
                        dmaps[i, j] = dmap
    
        return dmaps*self._mask_observed

    def _inpaint(self, imap, ivar, mask, inplace=True, qid=None, split_num=None):
        """Inpaint point sources given by the union catalog in input map.

        Parameters
        ---------
        imap : (..., 3, Ny, Nx) enmap
            Maps to be inpainted.
        ivar : (Ny, Nx) or (..., 1, Ny, Nx) enmap
            Inverse variance map. If not 2d, shape[:-3] must match imap.
        mask : (Ny, Nx) bool array
            Mask, True in observed regions.
        inplace : bool, optional
            Modify input map.
        qid : str, optional
            Array identifier, used to determine seed for inpainting.
        split_num : int, optional
            The 0-based index of the split that is inpainted, used to get unique seeds 
            per split if this function is called per split. Otherwise defaults to 0.
        """

        assert self._union_sources is not None, f'Inpainting needs union-sources, got {self._union_sources}'

        catalog = utils.get_catalog(self._union_sources)
        mask_bool = utils.get_mask_bool(mask)

        if qid:
            # This makes sure each qid gets a unique seed. The sim index is fixed.
            split_idx = 0 if split_num is None else split_num
            seed = utils.get_seed(*(split_idx, 999_999_999, self._data_model, qid))
        else:
            seed = None

        return inpaint.inpaint_noise_catalog(imap, ivar, mask_bool, catalog, inplace=inplace, 
                                             seed=seed)

    def _get_mask_est(self, min_threshold=1e-3, max_threshold=1.):
        """Load the data mask from disk according to instance attributes.

        Returns
        -------
        mask : (ny, nx) enmap
            Sky mask. Dowgraded if requested.
        """
        with bench.show('Generating harmonic-filter-estimate mask'):

            # first check for ivar compatibility and get map geometry
            full_shape, full_wcs = self._check_geometry()

            for i, qid in enumerate(self._qids):
                fn = simio.get_sim_mask_fn(
                    qid, self._data_model, use_default_mask=self._use_default_mask,
                    mask_version=self._mask_version, mask_name=self._mask_name, **self._kwargs
                )
                mask = enmap.read_map(fn).astype(self._dtype, copy=False)

                # check that we are using the same mask for each qid -- this is required!
                if i == 0:
                    mask_est = mask
                else:
                    with bench.show(f'Checking mask compatibility between {qid} and {self._qids[0]}'):
                        assert np.allclose(
                            mask, mask_est), 'qids do not share a common mask -- this is required!'
                        assert wcsutils.is_compatible(
                            mask.wcs, mask_est.wcs), 'qids do not share a common mask wcs -- this is required!'

            # Extract mask onto geometry specified by the ivar map.
            mask_est = enmap.extract(mask, full_shape, full_wcs)                                    
            
            if self._downgrade != 1:
                mask_est = utils.interpol_downgrade_cc_quad(mask_est, self._downgrade)

                # to prevent numerical error, cut below a threshold
                mask_est[mask_est < min_threshold] = 0.

                # to prevent numerical error, cut above a maximum
                mask_est[mask_est > max_threshold] = 1.

        return mask_est

    def _check_model_on_disk(self, split_num, keep_model=False, generate=True):
        """Check if this NoiseModel's model for a given split exists on disk. 
        If it does, return True. Depending on the 'keep_model' kwarg, possibly store
        the model in memory. Depending on the 'generate' kwarg, return either 
        False or raise a FileNotFoundError if it does not exist on-disk.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split model to look for.
        keep_model : bool, optional
            Store the generated (or loaded) model in the instance attributes if it
            exists on-disk, by default False.
        generate : bool, optional
            If the model does not exist on-disk and 'generate' is True, then return
            False. If the model does not exist on-disk and 'generate' is False, then
            raise a FileNotFoundError. By default True.

        Returns
        -------
        bool
            If the model exists on-disk, return True. If 'generate' is True and the 
            model does not exist on-disk, return False.

        Raises
        ------
        FileNotFoundError
            If 'generate' is False and the model does not exist on-disk.
        """
        try:
            self._get_model_from_disk(split_num, keep_model=keep_model)
            return True
        except (FileNotFoundError, OSError) as e:
            fn = self._get_model_fn(split_num)
            if generate:
                print(f'Model for split {split_num} not found on-disk, generating instead')
                return False
            else:
                print(f'Model for split {split_num} not found on-disk, please generate it first')
                raise FileNotFoundError(fn) from e

    def _get_model_from_disk(self, split_num, keep_model=True):
        """Load a sqrt-covariance matrix from disk. If keep_model, store it in instance attributes."""
        fn = self._get_model_fn(split_num)
        nm_dict = self._read_model(fn)
        if keep_model:
            self._keep_model(split_num, nm_dict)

    @abstractmethod
    def _get_model_fn(self, split_num):
        """Get a noise model filename for split split_num; return as <str>"""
        return ''

    @abstractmethod
    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        return {}
    
    def _keep_model(self, split_num, nm_dict):
        """Store a dictionary of noise model variables in instance attributes under key split_num"""
        print(f'Loading model for split {split_num} into memory')
        self._nm_dict[split_num] = nm_dict

    # for memory management
    def _delete_data(self):
        """Remove data stored in memory under instance attribute self._imap"""
        self._imap = None

    @abstractmethod
    def _get_dmap(self, imap):
        """Return the required input difference map for a NoiseModel subclass, from split data imap"""
        return enmap.ndmap

    @abstractmethod
    def _get_model(self, split_num, dmap, verbose=False):
        """Return a dictionary of noise model variables for this NoiseModel subclass, for split split_num and from difference maps dmap"""
        return {}

    @abstractmethod
    def _write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, **kwargs):
        """Write sqrt_cov_mat, sqrt_cov_ell, and possibly more noise model variables to filename fn"""
        pass

    def get_sim(self, split_num, sim_num, alm=True, check_on_disk=True, write=False,
                keep_model=True, do_mask_observed=True, verbose=False):
        """Generate a sim from this NoiseModel.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split to simulate.
        sim_num : int
            The map index, used in setting the random seed. Must be non-negative. If the sim
            is written to disk, this will be recorded in the filename. There is a maximum of
            9999, ie, one cannot have more than 10_000 of the same sim, of the same split, 
            from the same noise model (including the 'notes').
        alm : bool, optional
            Generate simulated alms instead of a simulated map, by default True.
        check_on_disk : bool, optional
            If True, first check if the exact sim (including the noise model 'notes'), 
            exists on disk, and if it does, load and return it. If it does not,
            generate the sim on-the-fly instead, by default True.
        write : bool, optional
            Save the generated sim to disk, by default False.
        keep_model : bool, optional
            Store the loaded model for this split in instance attributes, by default True.
            This spends memory to avoid spending time loading the model from disk
            for each call to this method.
        do_mask_observed : bool, optional
            Apply the mask determined by the observed patch to the sim, by default True. 
            If not applied, the sim will bleed into unobserved pixels, but this can
            potentially avoid intermediate calculation if the user will be applying
            their own analysis mask to sims before processing them.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        enmap.ndmap
            A sim of this noise model with the specified sim num, with shape
            (num_arrays, num_splits, num_pol, ny, nx), even if some of these
            axes have dimension 1. As implemented, num_splits is always 1. 
        """

        assert sim_num <= 9999, 'Cannot use a map index greater than 9999'

        if check_on_disk:
            res = self._check_sim_on_disk(split_num, sim_num, alm=alm, mask_obs=do_mask_observed)
            if res is not None:
                return res

        if split_num not in self._nm_dict:
            self._check_model_on_disk(split_num, keep_model=True, generate=False)

        seed = self._get_seed(split_num, sim_num)

        with bench.show(f'Generating noise sim for split {split_num}, map {sim_num}'):

            mask = self._mask_observed if do_mask_observed else None
            if alm:
                sim = self._get_sim_alm(split_num, seed, verbose=verbose, mask=mask)
            else:
                sim = self._get_sim(split_num, seed, verbose=verbose, mask=mask)

        if not keep_model:
            self.delete_model(split_num)
        
        if write:
            fn = self._get_sim_fn(split_num, sim_num, alm=alm, mask_obs=do_mask_observed)
            if alm:
                utils.write_alm(fn, sim, dtype=self._dtype)
            else:
                enmap.write_map(fn, sim)
        return sim

    def _check_sim_on_disk(self, split_num, sim_num, alm=False, mask_obs=True):
        """Check if sim with split_num, sim_num exists on-disk; if so return it, else return None."""
        fn = self._get_sim_fn(split_num, sim_num, alm=alm, mask_obs=mask_obs)
        try:
            if alm:
                # we know the preshape is (num_arrays, num_splits=1)
                return utils.read_alm(fn, preshape=(self._num_arrays, 1))
            else:
                return enmap.read_map(fn)
        except FileNotFoundError:
            print(f'Sim for split {split_num}, map {sim_num} not found on disk, generating instead')
            return None

    @abstractmethod
    def _get_sim_fn(self, split_num, sim_num, alm=False, mask_obs=True):
        """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
        pass

    def _get_seed(self, split_num, sim_num):
        """Return seed for sim with split_num, sim_num."""
        return utils.get_seed(
            *(split_num, sim_num, self._data_model, *self._qids)
            )

    @abstractmethod
    def _get_sim(self, split_num, seed, mask=None, verbose=False):
        """Return a masked enmap.ndmap sim of split split_num, with seed <sequence of ints>"""
        return enmap.ndmap

    @abstractmethod
    def _get_sim_alm(self, split_num, seed, mask=None, verbose=False):
        """Return a masked alm sim of split split_num, with seed <sequence of ints>"""
        pass

    def delete_model(self, split_num):
        """Delete a dictionary entry of noise model variables from instance attributes under key split_num"""
        try:
            del self._nm_dict[split_num] 
        except KeyError:
            print(f'Nothing to delete, no model in memory for split {split_num}')

    @property
    def num_splits(self):
        return self._num_splits

    @property
    def mask_est(self):
        return self._mask_est

    @property
    def mask_observed(self):
        return self._mask_observed


@register()
class TiledNoiseModel(NoiseModel):

    def __init__(self, *qids, data_model=None, preload=True, ivar=None, mask_est=None,
                calibrated=True, downgrade=1, lmax=None, mask_version=None, mask_name=None,
                union_sources=None, kfilt_lbounds=None, fwhm_ivar=None, notes=None, dtype=None,
                width_deg=4., height_deg=4., delta_ell_smooth=400, **kwargs):
        """A TiledNoiseModel object supports drawing simulations which capture spatially-varying
        noise correlation directions in map-domain data. They also capture the total noise power
        spectrum, spatially-varying map depth, and array-array correlations.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model : soapack.DataModel, optional
            DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
        preload: bool, optional
            If ivar is None, load them ivar maps for each qid, by default True. Setting
            to False can expedite access to helper methods of this class without waiting
            for ivar to load from disk.
        mask : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            to whiten the data before estimating its variance, by default None.
            If None and preload, will load a mask according to the 'mask_version' and
            'mask_name' kwargs.
        ivar : array-like, optional
            Data inverse-variance maps, by default None. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations. If None
            and preload, will be loaded via DataModel according to 'downgrade' and
            'calibrated' kwargs.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            to whiten the data before estimating its variance, by default None. If
            provided, assumed properly downgraded into compatible wcs with internal
            NoiseModel operations. If None and preload, will load a mask according to 
            the 'mask_version' and 'mask_name' kwargs.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default True.
        downgrade : int, optional
            The factor to downgrade map pixels by, by default 1.
        lmax : int, optional
            The bandlimit of the maps, by default None.
            If None, will be set to twice the theoretical CAR limit, ie 180/wcs.wcs.cdelt[1].
        mask_version : str, optional
           The mask version folder name, by default None.
           If None, will first look in config 'mnms' block, then block of default data model.
        mask_name : str, optional
            Name of mask file, by default None.
            If None, a default mask will be loaded from disk.
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
        width_deg : scalar, optional
            The characteristic tile width in degrees, by default 4.
        height_deg : scalar, optional
            The characteristic tile height in degrees,, by default 4.
        delta_ell_smooth : int, optional
            The smoothing scale in Fourier space to mitigate bias in the noise model
            from a small number of data splits, by default 400.
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.

        Notes
        -----
        Unless passed explicitly, the mask and ivar will be loaded at object instantiation time, 
        and stored as instance attributes.

        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> tnm = nm.TiledNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
        >>> tnm.get_model() # will take several minutes and require a lot of memory
                            # if running this exact model for the first time, otherwise
                            # will return None if model exists on-disk already
        >>> imap = tnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
                                       # the map will have "index" 123, which is used in making
                                       # the random seed whether or not the sim is saved to disk,
                                       # and will be recorded in the filename if saved to disk.
        >>> print(imap.shape)
        >>> (2, 1, 3, 5600, 21600)
        """
        super().__init__(
            *qids, data_model=data_model, preload=preload, ivar=ivar, mask_est=mask_est,
            calibrated=calibrated, downgrade=downgrade, lmax=lmax, mask_version=mask_version,
            mask_name=mask_name, union_sources=union_sources, kfilt_lbounds=kfilt_lbounds,
            fwhm_ivar=fwhm_ivar, notes=notes, dtype=dtype, **kwargs
        )

        # save model-specific info
        self._width_deg = width_deg
        self._height_deg = height_deg
        self._delta_ell_smooth = delta_ell_smooth

    def _get_model_fn(self, split_num):
        """Get a noise model filename for split split_num; return as <str>"""
        return simio.get_tiled_model_fn(
            self._qids, split_num, self._width_deg, self._height_deg, self._delta_ell_smooth, self._lmax, notes=self._notes,
            data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, union_sources=self._union_sources,
            kfilt_lbounds=self._kfilt_lbounds, fwhm_ivar=self._fwhm_ivar, **self._kwargs
        )

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        # read from disk
        sqrt_cov_mat, extra_hdu = tiled_ndmap.read_tiled_ndmap(
            fn, extra_hdu=['SQRT_COV_ELL']
        )
        sqrt_cov_ell = extra_hdu['SQRT_COV_ELL']
    
        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_ell': sqrt_cov_ell
            }

    def _get_dmap(self, imap):
        """Return the required input difference map for a NoiseModel subclass, from split data imap"""
        # model needs whitened noise maps as input
        return imap * np.sqrt(utils.get_ivar_eff(self._ivar))

    def _get_model(self, split_num, dmap, verbose=False):
        """Return a dictionary of noise model variables for this NoiseModel subclass, for split split_num and from difference maps dmap"""
        sqrt_cov_mat, sqrt_cov_ell = tiled_noise.get_tiled_noise_covsqrt(
            dmap, split_num, mask_observed=self._mask_observed, mask_est=self._mask_est, 
            width_deg=self._width_deg, height_deg=self._height_deg,
            delta_ell_smooth=self._delta_ell_smooth, lmax=self._lmax, rfft=True, nthread=0, verbose=verbose
        )
        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_ell': sqrt_cov_ell
            }

    def _write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None):
        """Write sqrt_cov_mat, sqrt_cov_ell, and possibly more noise model variables to filename fn"""
        tiled_ndmap.write_tiled_ndmap(
            fn, sqrt_cov_mat, extra_hdu={'SQRT_COV_ELL': sqrt_cov_ell}
        )

    def _get_sim_fn(self, split_num, sim_num, alm=False, mask_obs=True):
        """Get a sim filename for split split_num, sim sim_num; return as <str>"""
        return simio.get_tiled_sim_fn(
            self._qids, self._width_deg, self._height_deg, self._delta_ell_smooth, self._lmax, split_num, sim_num, alm=alm, notes=self._notes,
            data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, union_sources=self._union_sources, kfilt_lbounds=self._kfilt_lbounds, fwhm_ivar=self._fwhm_ivar,
            mask_obs=mask_obs, **self._kwargs
        )

    def _get_sim(self, split_num, seed, mask=None, verbose=False):
        """Return a masked enmap.ndmap sim of split split_num, with seed <sequence of ints>"""
        
        # Get noise model variables 
        sqrt_cov_mat = self._nm_dict[split_num]['sqrt_cov_mat']
        sqrt_cov_ell = self._nm_dict[split_num]['sqrt_cov_ell']
        
        sim = tiled_noise.get_tiled_noise_sim(
            sqrt_cov_mat, split_num, ivar=self._ivar,
            sqrt_cov_ell=sqrt_cov_ell, num_arrays=self._num_arrays,
            num_splits=self._num_splits, nthread=0, seed=seed, verbose=verbose
        )
        if mask is not None:
            sim *= mask
        return sim

    def _get_sim_alm(self, split_num, seed, mask=None, verbose=False):    
        """Return a masked alm sim of split split_num, with seed <sequence of ints>"""

        sim = self._get_sim(split_num, seed, mask=mask, verbose=verbose)
        return utils.map2alm(sim, lmax=self._lmax)


@register()
class WaveletNoiseModel(NoiseModel):

    def __init__(self, *qids, data_model=None, preload=True, ivar=None, mask_est=None,
                 calibrated=True, downgrade=1, lmax=None, mask_version=None, mask_name=None,
                 union_sources=None, kfilt_lbounds=None, fwhm_ivar=None, notes=None, dtype=None,
                 lamb=1.3, smooth_loc=False, fwhm_fact=2, **kwargs):
        """A WaveletNoiseModel object supports drawing simulations which capture scale-dependent, 
        spatially-varying map depth. They also capture the total noise power spectrum, and 
        array-array correlations.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model : soapack.DataModel, optional
            DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
        preload: bool, optional
            If mask and/or ivar are respectively None, load them, by default True. Setting
            to False can expedite access to helper methods of this class without waiting
            for data to load.
        mask : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            to whiten the data before estimating its variance, by default None.
            If None and preload, will load a mask according to the 'mask_version' and
            'mask_name' kwargs.
        ivar : array-like, optional
            Data inverse-variance maps, by default None.
            If None, will be loaded via DataModel according to 'downgrade' and 'calibrated' kwargs.
        imap : array-like, optional
            Data maps, by default None.
            If None, will be loaded in call to NoiseModel.get_model(...), and may be retained in 
            memory if keep_data=True is passed to that function.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default True.
        downgrade : int, optional
            The factor to downgrade map pixels by, by default 1.
        lmax : int, optional
            The bandlimit of the maps, by default None.
            If None, will be set to twice the theoretical CAR limit, ie 180/wcs.wcs.cdelt[1].
        mask_version : str, optional
           The mask version folder name, by default None.
           If None, will first look in config 'mnms' block, then block of default data model.
        mask_name : str, optional
            Name of mask file, by default None.
            If None, a default mask will be loaded from disk.
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
        lamb : float, optional
            Parameter specifying width of wavelets kernels in log(ell), by default 1.3
        smooth_loc : bool, optional
            If passed, use smoothing kernel that varies over the map, smaller along edge of 
            mask, by default False.
        fwhm_fact : float, optional
            Factor determining smoothing scale at each wavelet scale:
            FWHM = fact * pi / lmax, where lmax is the max wavelet ell.

        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.

        Notes
        -----
        Unless passed explicitly, the mask and ivar will be loaded at object instantiation time, 
        and stored as instance attributes.

        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> wnm = nm.WaveletNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
        >>> wnm.get_model() # will take several minutes and require a lot of memory
                            # if running this exact model for the first time, otherwise
                            # will return None if model exists on-disk already
        >>> imap = wnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
                                       # the map will have "index" 123, which is used in making
                                       # the random seed whether or not the sim is saved to disk,
                                       # and will be recorded in the filename if saved to disk.
        >>> print(imap.shape)
        >>> (2, 1, 3, 5600, 21600)
        """
        super().__init__(
            *qids, data_model=data_model, preload=preload, ivar=ivar, mask_est=mask_est,
            calibrated=calibrated, downgrade=downgrade, lmax=lmax, mask_version=mask_version,
            mask_name=mask_name, union_sources=union_sources, kfilt_lbounds=kfilt_lbounds,
            fwhm_ivar=fwhm_ivar, notes=notes, dtype=dtype, **kwargs
        )

        # save model-specific info
        self._lamb = lamb
        self._smooth_loc = smooth_loc
        self._fwhm_fact = fwhm_fact

    def _get_model_fn(self, split_num):
        """Get a noise model filename for split split_num; return as <str>"""
        return simio.get_wav_model_fn(
            self._qids, split_num, self._lamb, self._lmax, self._smooth_loc, self._fwhm_fact, 
            notes=self._notes, data_model=self._data_model, mask_version=self._mask_version,
            bin_apod=self._use_default_mask, mask_name=self._mask_name, calibrated=self._calibrated,
            downgrade=self._downgrade, union_sources=self._union_sources, kfilt_lbounds=self._kfilt_lbounds,
            fwhm_ivar=self._fwhm_ivar, **self._kwargs
        )

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        # read from disk
        sqrt_cov_mat, nm_dict = wavtrans.read_wav(
            fn, extra=['sqrt_cov_ell', 'w_ell']
        )
        nm_dict['sqrt_cov_mat'] = sqrt_cov_mat
        
        return nm_dict

    def _get_dmap(self, imap):
        """Return the required input difference map for a NoiseModel subclass, from split data imap"""
        # Correction factor turns split difference d_i to split noise n_i 
        # (which is the quantity we want to simulate in the end).
        return imap * utils.get_corr_fact(self._ivar)

    def _get_model(self, split_num, dmap, verbose=False):
        """
        Return a dictionary of noise model variables for this NoiseModel subclass,
        for split split_num and from difference maps dmap
        """
        
        sqrt_cov_mat, sqrt_cov_ell, w_ell = wav_noise.estimate_sqrt_cov_wav_from_enmap(
            dmap[:, split_num], self._mask_observed, self._lmax, self._mask_est, lamb=self._lamb,
            smooth_loc=self._smooth_loc, fwhm_fact=self._fwhm_fact
        )

        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_ell': sqrt_cov_ell,
            'w_ell': w_ell
            }

    def _write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, w_ell=None):
        """Write sqrt_cov_mat, sqrt_cov_ell, and possibly more noise model variables to filename fn"""
        wavtrans.write_wav(
            fn, sqrt_cov_mat, symm_axes=[[0, 1], [2, 3]],
            extra={'sqrt_cov_ell': sqrt_cov_ell, 'w_ell': w_ell}
        )

    def _get_sim_fn(self, split_num, sim_num, alm=False, mask_obs=True):
        """Get a sim filename for split split_num, sim sim_num; return as <str>"""
        return simio.get_wav_sim_fn(
            self._qids, split_num, self._lamb, self._lmax, self._smooth_loc, self._fwhm_fact,
            sim_num, alm=alm, notes=self._notes, data_model=self._data_model,
            mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, union_sources=self._union_sources,
            kfilt_lbounds=self._kfilt_lbounds, fwhm_ivar=self._fwhm_ivar, mask_obs=mask_obs, **self._kwargs
        )

    def _get_sim(self, split_num, seed, mask=None, verbose=False):
        """Return a masked enmap.ndmap sim of split split_num, with seed <sequence of ints>"""

        alm, ainfo = self._get_sim_alm(split_num, seed, mask=None, return_ainfo=True, verbose=verbose)
        sim = utils.alm2map(alm, shape=self._shape, wcs=self._wcs,
                            dtype=np.float32, ainfo=ainfo)
        if mask is not None:
            sim *= mask
        return sim

    def _get_sim_alm(self, split_num, seed, mask=None, return_ainfo=False, verbose=False):
        """Return a masked alm sim of split split_num, with seed <sequence of ints>"""

        # Get noise model variables. 
        sqrt_cov_mat = self._nm_dict[split_num]['sqrt_cov_mat']
        sqrt_cov_ell = self._nm_dict[split_num]['sqrt_cov_ell']
        w_ell = self._nm_dict[split_num]['w_ell']

        alm, ainfo = wav_noise.rand_alm_from_sqrt_cov_wav(
            sqrt_cov_mat, sqrt_cov_ell, self._lmax,
            w_ell, dtype=np.complex64, seed=seed)

        # We always want shape (num_arrays, num_splits=1, num_pol, nelem).
        assert alm.ndim == 3, 'Alm must have shape (num_arrays, num_pol, nelem)'
        alm = alm.reshape(alm.shape[0], 1, *alm.shape[1:])

        if mask is not None:
            sim = utils.alm2map(alm, shape=self._shape, wcs=self._wcs,
                                dtype=np.float32, ainfo=ainfo)
            sim *= mask
            utils.map2alm(sim, alm=alm, ainfo=ainfo)
        if return_ainfo:
            return alm, ainfo
        else:
            return alm      


@register()
class FSAWNoiseModel(NoiseModel):

    def __init__(self, *qids, data_model=None, preload=True, ivar=None, mask_est=None,
                 calibrated=True, downgrade=1, lmax=None, mask_version=None, mask_name=None,
                 union_sources=None, kfilt_lbounds=None, fwhm_ivar=None, notes=None, 
                 dtype=None, lamb=1.8, n=24, fwhm_fact=2., lmin=10, **kwargs):
        """An FSAWNoiseModel object supports drawing simulations which capture direction- 
        and scale-dependent, spatially-varying map depth. The simultaneous direction- and
        scale-sensitivity is achieved through steerable wavelet kernels in Fourier space.
        They also capture the total noise power spectrum, and array-array correlations.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model : soapack.DataModel, optional
            DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
        preload: bool, optional
            If mask and/or ivar are respectively None, load them, by default True. Setting
            to False can expedite access to helper methods of this class without waiting
            for data to load.
        mask : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            to whiten the data before estimating its variance, by default None.
            If None and preload, will load a mask according to the 'mask_version' and
            'mask_name' kwargs.
        ivar : array-like, optional
            Data inverse-variance maps, by default None.
            If None, will be loaded via DataModel according to 'downgrade' and 'calibrated' kwargs.
        imap : array-like, optional
            Data maps, by default None.
            If None, will be loaded in call to NoiseModel.get_model(...), and may be retained in 
            memory if keep_data=True is passed to that function.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default True.
        downgrade : int, optional
            The factor to downgrade map pixels by, by default 1.
        lmax : int, optional
            The bandlimit of the maps, by default None.
            If None, will be set to twice the theoretical CAR limit, ie 180/wcs.wcs.cdelt[1].
        mask_version : str, optional
           The mask version folder name, by default None.
           If None, will first look in config 'mnms' block, then block of default data model.
        mask_name : str, optional
            Name of mask file, by default None.
            If None, a default mask will be loaded from disk.
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
        lamb : float, optional
            Parameter specifying width of wavelets kernels in log(ell), by default 1.8
        n : int, optional
            Azimithul bandlimit (in radians per azimuthal radian) of the
            directional kernels. In other words, there are n+1 azimuthal 
            kernels of the form cos^n(phi), by default 24. For speed, the lowest- and
            highest-ell kernels will override to 0, and the next-lowest- and
            next-highest-ell kernels will override to n//2. 
        fwhm_fact : float, optional
            Factor determining smoothing scale at each wavelet scale:
            FWHM = fact * pi / lmax, where lmax is the max wavelet ell.

        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.

        Notes
        -----
        Unless passed explicitly, the mask and ivar will be loaded at object instantiation time, 
        and stored as instance attributes.

        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> fsawnm = nm.FSAWNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
        >>> fsawnm.get_model() # will take several minutes and require a lot of memory
                            # if running this exact model for the first time, otherwise
                            # will return None if model exists on-disk already
        >>> fsawnm = wnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
                                       # the map will have "index" 123, which is used in making
                                       # the random seed whether or not the sim is saved to disk,
                                       # and will be recorded in the filename if saved to disk.
        >>> print(imap.shape)
        >>> (2, 1, 3, 5600, 21600)
        """
        super().__init__(
            *qids, data_model=data_model, preload=preload, ivar=ivar, mask_est=mask_est,
            calibrated=calibrated, downgrade=downgrade, lmax=lmax, mask_version=mask_version,
            mask_name=mask_name, union_sources=union_sources, kfilt_lbounds=kfilt_lbounds,
            fwhm_ivar=fwhm_ivar, notes=notes, dtype=dtype, **kwargs
        )

        # save model-specific info
        self._lamb = lamb
        self._n = n
        self._fwhm_fact = fwhm_fact      

        # build the kernels.
        # there is no real significance to lmax=10_800 here. it will just be
        # used to build the kernel generating functions, specifically, to 
        # check that the last kernel is not "clipped"
        self._fk = fsaw_noise.FSAWKernels(
            self._lamb, 10_800, lmin, 5300, self._n, self._shape, self._wcs,
            dtype=self._dtype, nforw=[0, n//2], nback=[0, n//2]
        )

    def _get_model_fn(self, split_num):
        """Get a noise model filename for split split_num; return as <str>"""
        return simio.get_fsaw_model_fn(
            self._qids, split_num, self._lamb, self._n, self._fwhm_fact,
            self._lmax, notes=self._notes, data_model=self._data_model, 
            mask_version=self._mask_version, bin_apod=self._use_default_mask,
            mask_name=self._mask_name, calibrated=self._calibrated,
            downgrade=self._downgrade, union_sources=self._union_sources,
            kfilt_lbounds=self._kfilt_lbounds, fwhm_ivar=self._fwhm_ivar, **self._kwargs
        )

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        sqrt_cov_mat, extra_datasets = fsaw_noise.read_wavs(
            fn, extra_datasets=['sqrt_cov_k']
        )
        sqrt_cov_k = extra_datasets['sqrt_cov_k']

        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_k': sqrt_cov_k
            }

    def _get_dmap(self, imap):
        """Return the required input difference map for a NoiseModel subclass, from split data imap"""
        # Correction factor turns split difference d_i to split noise n_i 
        # (which is the quantity we want to simulate in the end).
        return imap * utils.get_corr_fact(self._ivar)

    def _get_model(self, split_num, dmap, verbose=False):
        """Return a dictionary of noise model variables for this NoiseModel subclass, for split split_num and from difference maps dmap"""
        sqrt_cov_mat, sqrt_cov_k = fsaw_noise.get_fsaw_noise_covsqrt(
            self._fk, dmap[:, split_num], mask_observed=self._mask_observed,
            mask_est=self._mask_est, fwhm_fact=self._fwhm_fact, nthread=0,
            verbose=verbose
        )

        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_k': sqrt_cov_k
            }

    def _write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_k=None, **kwargs):
        """Write sqrt_cov_mat, sqrt_cov_k, and possibly more noise model variables to filename fn"""
        fsaw_noise.write_wavs(
            fn, sqrt_cov_mat, extra_datasets={'sqrt_cov_k': sqrt_cov_k}
        )

    def _get_sim_fn(self, split_num, sim_num, alm=False, mask_obs=True):
        """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
        return simio.get_fsaw_sim_fn(
            self._qids, split_num, self._lamb, self._n, self._fwhm_fact,
            self._lmax, sim_num, alm=alm, notes=self._notes, data_model=self._data_model, 
            mask_version=self._mask_version, bin_apod=self._use_default_mask,
            mask_name=self._mask_name, calibrated=self._calibrated,
            downgrade=self._downgrade, union_sources=self._union_sources,
            kfilt_lbounds=self._kfilt_lbounds, mask_obs=mask_obs, fwhm_ivar=self._fwhm_ivar, 
            **self._kwargs
        )

    def _get_sim(self, split_num, seed, mask=None, verbose=False):
        """Return a masked enmap.ndmap sim of split split_num, with seed <sequence of ints>"""
        # Get noise model variables 
        sqrt_cov_mat = self._nm_dict[split_num]['sqrt_cov_mat']
        sqrt_cov_k = self._nm_dict[split_num]['sqrt_cov_k']

        sim = fsaw_noise.get_fsaw_noise_sim(
            self._fk, sqrt_cov_mat, preshape=(self._num_arrays, -1),
            sqrt_cov_k=sqrt_cov_k, seed=seed, nthread=0, verbose=verbose
        )

        # We always want shape (num_arrays, num_splits=1, num_pol, ny, nx).
        assert sim.ndim == 4, 'Sim must have shape (num_arrays, num_pol, ny, nx)'
        sim = sim.reshape(sim.shape[0], 1, *sim.shape[1:])

        if mask is not None:
            sim *= mask
        return sim

    def _get_sim_alm(self, split_num, seed, mask=None, verbose=False):
        """Return a masked alm sim of split split_num, with seed <sequence of ints>"""
        sim = self._get_sim(split_num, seed, mask=mask, verbose=verbose)
        return utils.map2alm(sim, lmax=self._lmax)


class WavFiltTile(NoiseModel):

    def __init__(self):
        pass

    def _get_model(self):
        pass
        # 1. get wavelet noise model including sqrt_cov_ell and wavelet maps
        # 2. use sqrt_cov_ell to flatten noise maps
        # 3. do map->alm->wav on noise maps, flatten using wavelet maps
        # 4. do wav->alm->map to recover full-res flattened noise map
        # 5. make tiled noise model

    def _get_sim(self):
        pass
        # 1. draw tiled noise sim, stitch into flat map
        # 2. do map->alm->wav, unflatten using wavelet maps
        # 3. do wav->alm->map to recover full-res unflattened noise map
        # 4. use sqrt_cov_ell to unflatten
        # 5. use corr_fact to get sim

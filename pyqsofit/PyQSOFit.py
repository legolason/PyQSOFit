#!/usr/bin/python3
# A code for quasar spectrum fitting
# Last modified on 07/12/2022
#v1.1
# Auther: Hengxiao Guo @ UCI
# Email: hengxiaoguo@gmail.com
# Co-Auther Shu Wang, Yue Shen, Wenke Ren
# v2.0
# Auther:  Qiaoya wu @ UIUC
# Email: qiaoyaw2@illinois.edu
# -------------------------------------------------


import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sfdmap
from scipy import interpolate
from scipy import integrate
from scipy.signal import medfilt
from kapteyn import kmpfit
from PyAstronomy import pyasl
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import rc
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.modeling.blackbody import blackbody_lambda
from astropy.table import Table
import warnings

warnings.filterwarnings("ignore")


class QSOFit():

    def __init__(self, lam, flux, err, z, ra=- 999., dec=-999., plateid=None, mjd=None, fiberid=None, \
                 target_info=None, path=None, and_mask=None, or_mask=None):
        """
        Get the input data perpared for the QSO spectral fitting

        Parameters:
        -----------
        lam: 1-D array with Npix
             Observed wavelength in unit of Angstrom

        flux: 1-D array with Npix
             Observed flux density in unit of 10^{-17} erg/s/cm^2/Angstrom

        err: 1-D array with Npix
             1 sigma err with the same unit of flux

        z: float number
            redshift

        ra, dec: float number, optional
            the location of the source, right ascension and declination. The default number is 0

        plateid, mjd, fiberid: integer number, optional
            If the source is SDSS object, they have the plate ID, MJD and Fiber ID in their file herader.

        target_info: str
            additional information about the target

        path: str
            the path of the input data

        and_mask, or_mask: 1-D array with Npix, optional
            the bad pixels defined from SDSS data, which can be got from SDSS datacube.

        """

        self.lam = np.asarray(lam, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64)
        self.z = z
        self.and_mask = and_mask
        self.or_mask = or_mask
        self.ra = ra
        self.dec = dec
        self.plateid = plateid
        self.mjd = mjd
        self.fiberid = fiberid
        self.target_info = target_info
        self.path = path

    def Fit(self, name=None, nsmooth=1, and_or_mask=True, reject_badpix=False, deredden=True, wave_range=None, \
            wave_mask=None, decomposition_host=True, BC03=False, Mi=None, npca_gal=5, npca_qso=20, \
            Fe_uv_op=True, Fe_flux_range=None, poly=False, BC=False, rej_abs=False, \
            initial_guess=None, MC=False, n_trails=25, \
            linefit=True, tie_lambda=True, tie_width=True, tie_flux_1=True, tie_flux_2=True, \
            rej_line_abs=True, rej_line_max_niter=2, save_result=True, \
            if_localfit=False, if_save_localfit=True, if_plot_localfit=False, if_tie_localfit=True, save_localfit_plot_path=None, \
            plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True, dustmap_path=None, save_fig_path=None,
            save_fits_path=None, save_fits_name=None, if_read_line_prop = True, \
            if_dump_MC_result = False, if_save_spec = False):

        """
        Fit the QSO spectrum and get different decomposed components and corresponding parameters

        Parameter:
        ----------
        name: str, optinal
            source name, Default is None. If None, it will use plateid+mjd+fiberid as the name. If there are no
            such parameters, it will be empty.

        nsmooth: integer number, optional
            do n-pixel smoothing to the raw input flux and err spectra. The default is set to 1 (no smooth).
            It will return the same array size. We note that smooth the raw data is not suggested, this function is in case of some fail-fitted low S/N spectra.

        and_or_mask: bool, optional
            If True, and and_mask or or_mask is not None, it will delete the masked pixels, and only return the remained pixels. Default: False

        reject_badpix: bool, optional
            reject 10 most possible outliers by the test of pointDistGESD. One important Caveat here is that this process will also delete narrow emission lines
            in some high SN ratio object e.g., [OIII]. Only use it when you are definitely clear about what you are doing. It will return the remained pixels.

        deredden: bool, optional
            correct the Galactic extinction only if the RA and Dec are available. It will return the corrected flux with the same array size. Default: True.

        wave_range: 2-element array, optional
            trim input wavelength (lam) according to the min and max range of the input 2-element array, e.g.,
            np.array([4000.,6000.]) in Rest frame range. Default: None

        wave_mask: 2-D array
            mask some absorption lines or sky lines in spectrum, e.g., np.array([[2200.,2300.]]), np.array([[5650.,5750.],[5850.,5900.]])

        decomposition_host: bool, optional
            If True, the host galaxy-QSO decomposition will be applied. If no more than 100 pixels are negative, the result will be applied. The Decomposition is
            based on the PCA method of Yip et al. 2004 (AJ, 128, 585) & (128, 2603). Now the template is only available for redshift < 1.16 in specific absolute
            magnitude bins. For galaxy, the global model has 10 PCA components and first 5 will enough to reproduce 98.37% galaxy spectra. For QSO, the global model
            has 50, and the first 20 will reproduce 96.89% QSOs. If have i-band absolute magnitude, the Luminosity-redshift binned PCA components are available.
            Then the first 10 PCA in each bin is enough to reproduce most QSO spectrum. Default: False
        BC03: bool, optional
            if True, it will use Bruzual1 & Charlot 2003 host model to fit spectrum, high shift host will be low resolution R ~ 300, the rest is R ~ 2000. Default: False

        Mi: float, optional
            the absolute magnitude of i band. It only works when decomposition_host is True. If not None, the Luminosity redshift binned PCA will be used to decompose
            the spectrum. Default: None

        npca_gal: int, optional
            the number of galaxy PCA components applied. It only works when decomposition_host is True. The default is 5,
            which is already account for 98.37% galaxies.

        npca_qso: int, optional
            the number of QSO PCA components applied. It only works when decomposition_host is True. The default is 20,
            No matter the global or luminosity-redshift binned PCA is used, it can reproduce > 92% QSOs. The binned PCA
            is better if have Mi information.

        Fe_uv_op: bool, optional
            if True, fit continuum with UV and optical FeII template. Default: True

        Fe_flux_range: 1-D array, 2-D array or None, optional
            if 1-D array was given, it should contain two parameters contain a range of wavelength. FeII flux within this range would be calculate and documented in the result fits file.
            if 2-D array was given, it should contain a series of ranges. FeII flux within these ranges would be documented respectively.
            if None was given, nothing would be return. Default: None

        poly: bool, optional
            if True, fit continuum with the polynomial component to account for the dust reddening. Default: False

        BC: bool, optional
            if True, fit continuum with Balmer continua from 1000 to 3646A. Default: False

        rej_abs: bool, optional
            if True, it will iterate the continuum fitting for deleting some 3 sigmas out continuum window points
            (< 3500A), which might fall into the broad absorption lines. Default: False

        rej_line_abs: bool, optional
            if True, will iterate the line fitting to remove some 3 sigmas absorption out from complex line window
            Default: True

        if_localfit: bool, optional
            if True, fit narrow lines CaII3934, OII3728, NeV3426 with localfit (fit only PL+gauss). Default: False

        if_plot_localfit: bool, optional
            if True, plot the local fit QA plots for narrow lines CaII3934, OII3728, NeV3426. Default: False

        save_localfit_result_path: str, optional
            the output path of the localfit fits. If None, the default "save_fig_path" is set to "path"

        save_localfit_plot_path: str, optional
            the output path of the localfit fits. If None, the default "save_fig_path" is set to "path"

        initial_gauss: 1*14 array, optional
            better initial value will help find a solution faster. Default initial is np.array([0., 3000., 0., 0.,
            3000., 0., 1., -1.5, 0., 15000., 0.5, 0., 0., 0.]). First six parameters are flux scale, FWHM, small shift for wavelength for UV and optical FeII template,
            respectively. The next two parameters are the power-law slope and intercept. The next three are the norm, Te, tau_BE in Balmer continuum model in
            Dietrich et al. 2002. the last three parameters are a,b,c in polynomial function a*(x-3000)+b*x^2+c*x^3.

        MC: bool, optional
            if True, do the Monte Carlo simulation based on the input error array to produce the MC error array.
            if False, the code will not save the error produced by kmpfit since it is biased and can not be trusted.
            But it can be still output by in kmpfit attribute. Default: False

        n_trails: int, optional
            the number of trails of the MC process to produce the error array. The conservative number should be larger than 20. It only works when MC is True. Default: 20

        linefit: bool, optional
            if True, the emission line will be fitted. Default: True

        tie_lambda: bool, optional
            if True, line center with the same "vindex" will be tied together in the same line complex, this is always used to tie e.g., NII. Default: False

        tie_width: bool, optional
            if True, line sigma with the same "windex" will be tied together in the same line complex, Default: False

        tie_flux_1: bool, optional
            if True, line flux with the flag "findex = 1" will be tied to the ratio of fvalue. To fix the narrow line flux, the user should fix the line width
            "tie_width" first. Default: False

        tie_flux_2: bool, optional
            if True, line flux with the flag "findex = 2" will be tied to the ratio of fvalue. To fix the narrow line flux, the user should fix the line width
            "tie_width" first. Default: False

        save_result: bool, optional
            if True, all the fitting results will be saved to a fits file, Default: True

        plot_fig: bool, optional
            if True, the fitting results will be plotted. Default: True

        save_fig: bool, optional
            if True, the figure will be saved, and the path can be set by "save_fig_path". Default: True

        plot_line_name: bool, optional
            if True, serval main emission lines will be plotted in the first panel of the output figure. Default: False

        plot_legend: bool, optional
            if True, open legend in the first panel of the output figure. Default: False

        dustmap_path: str, optional
            if Deredden is True, the dustmap_path must be set. If None, the default "dustmap_path" is set to "path"

        save_fig_path: str, optional
            the output path of the figure. If None, the default "save_fig_path" is set to "path"

        save_fit_path: str, optional
            the output path of the result fits. If None, the default "save_fits_path" is set to "path"

        save_fit_name: str, optional
            the output name of the result fits. Default: "result.fits"

        if_read_line_prop: bool, optional
            if True, the line_prop would be measured. Default: True

        Return:
        -----------
        .wave: array
            the rest wavelength, some pixels have been removed.

        .flux: array
            the rest flux. Dereddened and *(1+z) flux.

        .err: array
            the error.

        .wave_prereduced: array
            the wavelength after removing bad pixels, masking, deredden, spectral trim, and smoothing.

        .flux_prereduced: array
            the flux after removing bad pixels, masking, deredden, spectral trim, and smoothing.

        .err_prereduced: array
            the error after removing bad pixels, masking, deredden, spectral trim, and smoothing.

        .host: array
            the model of host galaxy from PCA method

        .qso: array
            the model of a quasar from PCA method.

        .SN_ratio_conti: float
            the mean S/N ratio of 1350, 3000 and 5100A.

        .conti_fit.: structure
            all the kmpfit continuum fitting results, including best-fit parameters and Chisquare, etc. For details,
            see https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html

        .f_conti_model: array
            the continuum model including power-law, polynomial, optical/UV FeII, Balmer continuum.

        .f_bc_model: array
            the Balmer continuum model.

        .f_fe_uv_model: array
            the UV FeII model.

        .f_fe_op_model: array
            the optical FeII model.

        .f_pl_model: array
            the power-law model.

        .f_poly_model: array
            the polynomial model.

        .PL_poly_BC: array
            The combination of Powerlaw, polynomial and Balmer continuum model.

        .line_flux: array
            the emission line flux after subtracting the .f_conti_model.

        .line_fit: structrue
            kmpfit line fitting results for last complexes (From Lya to Ha) , including best-fit parameters, errors (kmpfit derived) and Chisquare, etc. For details,
            see https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html

        .gauss_result: array
            3*n Gaussian parameters for all lines in the format of [scale, centerwave, sigma ], n is number of Gaussians for all complexes.

        .conti_result: array
            continuum parameters, including widely used continuum parameters and monochromatic flux at 1350, 3000
            and5100 Angstrom, etc. The corresponding names are listed in .conti_result_name. For all continuum fitting results,
            go to .conti_fit.params.

        .conti_result_name: array
            the names for .conti_result.

        .line_result: array
            emission line parameters, including FWHM, sigma, EW, measured from whole model of each main broad emission line covered,
            and fitting parameters of each Gaussian component. The corresponding names are listed in .line_result_name.


        .line_result_name: array
            the names for .line_result.

        .uniq_linecomp_sort: array
            the sorted complex names.

        .all_comp_range: array
            the start and end wavelength for each complex. e.g., Hb in [4640.  5100.].

        .linelist: array
            the information listed in the qsopar.fits.

        --------
        saved fits output:
            extension=1: basic fitting results
            extension=2: MC results
            extension=3: pre-reduced spectrum, continuum fitting spectrum, line fitting (absorption rejected) spectrum
        """

        self.name = name
        self.wave_range = wave_range
        self.wave_mask = wave_mask
        self.BC03 = BC03
        self.Mi = Mi
        self.npca_gal = npca_gal
        self.npca_qso = npca_qso
        self.initial_guess = initial_guess
        self.Fe_uv_op = Fe_uv_op
        self.Fe_flux_range = Fe_flux_range
        self.poly = poly
        self.BC = BC
        self.rej_abs = rej_abs
        self.rej_line_abs = rej_line_abs
        self.rej_line_max_niter = rej_line_max_niter
        self.if_localfit = if_localfit
        self.MC = MC
        self.n_trails = n_trails
        self.tie_lambda = tie_lambda
        self.tie_width = tie_width
        self.tie_flux_1 = tie_flux_1
        self.tie_flux_2 = tie_flux_2
        self.plot_line_name = plot_line_name
        self.plot_legend = plot_legend
        self.save_fig = save_fig
        self.if_read_line_prop = if_read_line_prop
        self.if_dump_MC_result = if_dump_MC_result
        self.if_save_spec = if_save_spec
        self.if_localfit = if_localfit
        self.if_tie_localfit = if_tie_localfit
        self.if_plot_localfit = if_plot_localfit

        # get the source name in plate-mjd-fiber, if no then None
        if name is None:
            if np.array([self.plateid, self.mjd, self.fiberid]).any() is not None:
                self.sdss_name = '%04d'%self.plateid+'-%05d'%self.mjd+'-%04d'%self.fiberid
            else:
                if self.plateid is None:
                    self.plateid = 0
                if self.mjd is None:
                    self.mjd = 0
                if self.fiberid is None:
                    self.fiberid = 0
                self.sdss_name = ''
        else:
            self.sdss_name = name

        # set default path for figure and fits
        if save_result == True and save_fits_path == None:
            save_fits_path = self.path
        if save_fig == True and save_fig_path == None:
            save_fig_path = self.path
        if save_fits_name == None:
            if self.sdss_name == '':
                save_fits_name = 'result'
            else:
                save_fits_name = self.sdss_name
        else:
            save_fits_name = save_fits_name

        # deal with pixels with error equal 0 or inifity
        lam_good, flux_good, err_good = self._Remove_bad_pixel(self.lam, self.flux, self.err)

        if (self.and_mask is not None) & (self.or_mask is not None):
            and_mask_good = self.and_mask[ind_gooderror]
            or_mask_good = self.or_mask[ind_gooderror]
            del self.and_mask, self.or_mask
            self.and_mask = and_mask_good
            self.or_mask = or_mask_good
        del self.err, self.flux, self.lam
        self.err = err_good
        self.flux = flux_good
        self.lam = lam_good

        if nsmooth is not None:
            self.flux = self.Smooth(self.flux, nsmooth)
            self.err = self.Smooth(self.err, nsmooth)
        if (and_or_mask == True) and (self.and_mask is not None or self.or_mask is not None):
            self._MaskSdssAndOr(self.lam, self.flux, self.err, self.and_mask, self.or_mask)
        if reject_badpix == True:
            self._RejectBadPix(self.lam, self.flux, self.err)
        if wave_range is not None:
            self._WaveTrim(self.lam, self.flux, self.err, self.z)
        if wave_mask is not None:
            self._WaveMsk(self.lam, self.flux, self.err, self.z)
        if deredden == True and self.ra != -999. and self.dec != -999.:
            self._DeRedden(self.lam, self.flux, self.err, self.ra, self.dec, dustmap_path)

        self._RestFrame(self.lam, self.flux, self.err, self.z)
        self._CalculateSN(self.wave, self.flux)
        self._OriginalSpec(self.wave, self.flux, self.err)

        # do host decomposition --------------
        if self.z < 1.16 and decomposition_host == True:
            self._DoDecomposition(self.wave, self.flux, self.err, self.path)
        else:
            self.decomposed = False
            if self.z > 1.16 and decomposition_host == True:
                print('redshift larger than 1.16 is not allowed for host '
                      'decomposion!')

        if self.rej_line_abs:
            self.rej_line_abs_wave = np.array([])
            self.rej_line_abs_flux = np.array([])
            self.rej_line_abs_err = np.array([])
        if self.if_save_spec:
            self.linespec_wave = np.array([])
            self.linespec_lineflux = np.array([])
            self.linespec_err = np.array([])
        # fit continuum --------------------
        self._DoContiFit(self.wave, self.flux, self.err, self.ra, self.dec, self.plateid, self.mjd, self.fiberid)
        # fit line
        if linefit == True:
            self._DoLineFit(self.wave, self.line_flux, self.err, self.conti_fit)
        else:
            self.ncomp = 0

        # narrow line local fit
        if if_localfit == True:
            self._DoLocalFit(self.wave, self.flux, self.err, if_plot_localfit, self.if_tie_localfit, save_localfit_plot_path)

        # save data -------
        if save_result == True:
            if linefit == False:
                self.line_result = np.array([])
                self.line_result_type = np.array([])
                self.line_result_name = np.array([])
            self._SaveResult(self.conti_result, self.conti_result_type, self.conti_result_name, self.line_result,
                             self.line_result_type, self.line_result_name, save_fits_path, save_fits_name, \
                             self.if_dump_MC_result, self.if_save_spec, self.if_localfit)

        # plot fig and save ------
        if plot_fig == True:
            if linefit == False:
                self.gauss_result = np.array([])
                self.all_comp_range = np.array([])
                self.uniq_linecomp_sort = np.array([])
            self._PlotFig(self.ra, self.dec, self.z, self.wave, self.flux, self.err, decomposition_host, linefit,
                          self.tmp_all, self.gauss_result, self.f_conti_model, self.conti_fit, self.all_comp_range,
                          self.uniq_linecomp_sort, self.line_flux, save_fig_path, self.target_info)

    def _MaskSdssAndOr(self, lam, flux, err, and_mask, or_mask):
        """
        Remove SDSS and_mask and or_mask points are not zero
        Parameter:
        ----------
        lam: wavelength
        flux: flux
        err: 1 sigma error
        and_mask: SDSS flag "and_mask", mask out all non-zero pixels
        or_mask: SDSS flag "or_mask", mask out all npn-zero pixels

        Retrun:
        ---------
        return the same size array of wavelength, flux, error
        """
        ind_and_or = np.where((and_mask == 0) & (or_mask == 0), True, False)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = lam[ind_and_or], flux[ind_and_or], err[ind_and_or]

    def _RejectBadPix(self, lam, flux, err):
        """
        Reject 10 most possiable outliers, input wavelength, flux and error. Return a different size wavelength,
        flux, and error.
        """
        # -----remove bad pixels, but not for high SN spectrum------------
        ind_bad = pyasl.pointDistGESD(flux, 10)
        wv = np.asarray([i for j, i in enumerate(lam) if j not in ind_bad[1]], dtype=np.float64)
        fx = np.asarray([i for j, i in enumerate(flux) if j not in ind_bad[1]], dtype=np.float64)
        er = np.asarray([i for j, i in enumerate(err) if j not in ind_bad[1]], dtype=np.float64)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = wv, fx, er
        return self.lam, self.flux, self.err

    def _Remove_bad_pixel(self, lam, flux, err):
        """deal with pixels with error equal 0 or inifity"""
        ind_gooderror = np.where((err != 0) & ~np.isinf(err), True, False)
        return lam[ind_gooderror], flux[ind_gooderror], err[ind_gooderror]

    def _WaveTrim(self, lam, flux, err, z):
        """
        Trim spectrum with a range in the rest frame.
        """
        # trim spectrum e.g., local fit emiision lines
        ind_trim = np.where((lam/(1.+z) > self.wave_range[0]) & (lam/(1.+z) < self.wave_range[1]), True, False)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = lam[ind_trim], flux[ind_trim], err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError("No enough pixels in the input wave_range!")
        return self.lam, self.flux, self.err

    def _WaveMsk(self, lam, flux, err, z):
        """Block the bad pixels or absorption lines in spectrum."""

        for msk in range(len(self.wave_mask)):
            try:
                ind_not_mask = ~np.where((lam/(1.+z) > self.wave_mask[msk, 0]) & (lam/(1.+z) < self.wave_mask[msk, 1]),
                                         True, False)
            except IndexError:
                raise RuntimeError("Wave_mask should be 2D array,e.g., np.array([[2000,3000],[3100,4000]]).")

            del self.lam, self.flux, self.err
            self.lam, self.flux, self.err = lam[ind_not_mask], flux[ind_not_mask], err[ind_not_mask]
            lam, flux, err = self.lam, self.flux, self.err
        return self.lam, self.flux, self.err

    def _DeRedden(self, lam, flux, err, ra, dec, dustmap_path):
        """Correct the Galactic extinction"""
        m = sfdmap.SFDMap(dustmap_path)
        zero_flux = np.where(flux == 0, True, False)
        flux[zero_flux] = 1e-10
        flux_unred = pyasl.unred(lam, flux, m.ebv(ra, dec))
        err_unred = err*flux_unred/flux
        flux_unred[zero_flux] = 0
        del self.flux, self.err
        self.flux = flux_unred
        self.err = err_unred
        self.EBV = m.ebv(ra, dec)
        return self.flux

    def _RestFrame(self, lam, flux, err, z):
        """Move wavelenth and flux to rest frame"""
        self.wave = lam/(1.+z)
        self.flux = flux*(1.+z)
        self.err = err*(1.+z)
        return self.wave, self.flux, self.err

    def _OriginalSpec(self, wave, flux, err):
        """save the orignial spectrum before host galaxy decompsition"""
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _CalculateSN(self, wave, flux):
        """calculate the spectral SN ratio for 1350, 3000, 5100A, return the mean value of Three spots"""
        if ((wave.min() < 1350. and wave.max() > 1350.) or (wave.min() < 3000. and wave.max() > 3000.) or (
                wave.min() < 5100. and wave.max() > 5100.)):

            ind5100 = np.where((wave > 5080.) & (wave < 5130.), True, False)
            ind3000 = np.where((wave > 3000.) & (wave < 3050.), True, False)
            ind1350 = np.where((wave > 1325.) & (wave < 1375.), True, False)

            tmp_SN = np.array([flux[ind5100].mean()/flux[ind5100].std(), flux[ind3000].mean()/flux[ind3000].std(),
                               flux[ind1350].mean()/flux[ind1350].std()])
            tmp_SN = tmp_SN[~np.isnan(tmp_SN)]
            self.SN_ratio_conti = tmp_SN.mean()
        else:
            self.SN_ratio_conti = -1.

        return self.SN_ratio_conti

    def _DoDecomposition(self, wave, flux, err, path):
        """Decompose the host galaxy from QSO"""
        datacube = self._HostDecompose(self.wave, self.flux, self.err, self.z, self.Mi, self.npca_gal, self.npca_qso,
                                       path)

        # for some negtive host templete, we do not do the decomposition
        if np.sum(np.where(datacube[3, :] < 0., True, False)) > 100:
            self.host = np.zeros(len(wave))
            self.decomposed = False
            print('Get negtive host galaxy flux larger than 100 pixels, '
                  'decomposition is not applied!')
        else:
            self.decomposed = True
            del self.wave, self.flux, self.err
            self.wave = datacube[0, :]
            # block OIII, ha,NII,SII,OII,Ha,Hb,Hr,hdelta

            line_mask = np.where(
                (self.wave < 4970.) & (self.wave > 4950.) | (self.wave < 5020.) & (self.wave > 5000.) | (
                        self.wave < 6590.) & (self.wave > 6540.) | (self.wave < 6740.) & (self.wave > 6710.) | (
                        self.wave < 3737.) & (self.wave > 3717.) | (self.wave < 4872.) & (self.wave > 4852.) | (
                        self.wave < 4350.) & (self.wave > 4330.) | (self.wave < 4111.) & (self.wave > 4091.), True,
                False)

            f = interpolate.interp1d(self.wave[~line_mask], datacube[3, :][~line_mask], bounds_error=False,
                                     fill_value=0)
            masked_host = f(self.wave)
            self.flux = datacube[1, :]-masked_host  # QSO flux without host
            self.err = datacube[2, :]
            self.host = datacube[3, :]
            self.qso = datacube[4, :]
            self.host_data = datacube[1, :]-self.qso
        return self.wave, self.flux, self.err

    def _HostDecompose(self, wave, flux, err, z, Mi, npca_gal, npca_qso, path):
        """
        core function to do host decomposition
        #Wave is the obs frame wavelength, n_gal and n_qso are the number of eigenspectra used to fit
        #If Mi is None then the qso use the globle ones to fit. If not then use the redshift-luminoisty binded ones to fit
        #See details:
        #Yip, C. W., Connolly, A. J., Szalay, A. S., et al. 2004a, AJ, 128, 585
        #Yip, C. W., Connolly, A. J., Vanden Berk, D. E., et al. 2004b, AJ, 128, 2603
        """

        # read galaxy and qso eigenspectra -----------------------------------
        if self.BC03 == False:
            galaxy = fits.open(path+'pca/Yip_pca_templates/gal_eigenspec_Yip2004.fits')
            gal = galaxy[1].data
            wave_gal = gal['wave'].flatten()
            flux_gal = gal['pca'].reshape(gal['pca'].shape[1], gal['pca'].shape[2])
        if self.BC03 == True:
            cc = 0
            flux03 = np.array([])
            for i in glob.glob(path+'/bc03/*.gz'):
                cc = cc+1
                gal_temp = np.genfromtxt(i)
                wave_gal = gal_temp[:, 0]
                flux03 = np.concatenate((flux03, gal_temp[:, 1]))
            flux_gal = np.array(flux03).reshape(cc, -1)

        if Mi is None:
            quasar = fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_global.fits')
        else:
            if -24 < Mi <= -22 and 0.08 <= z < 0.53:
                quasar = fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_CZBIN1.fits')
            elif -26 < Mi <= -24 and 0.08 <= z < 0.53:
                quasar = fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_DZBIN1.fits')
            elif -24 < Mi <= -22 and 0.53 <= z < 1.16:
                quasar = fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_BZBIN2.fits')
            elif -26 < Mi <= -24 and 0.53 <= z < 1.16:
                quasar = fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_CZBIN2.fits')
            elif -28 < Mi <= -26 and 0.53 <= z < 1.16:
                quasar = fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_DZBIN2.fits')
            else:
                raise RuntimeError('Host galaxy template is not available for this redshift and Magnitude!')

        qso = quasar[1].data
        wave_qso = qso['wave'].flatten()
        flux_qso = qso['pca'].reshape(qso['pca'].shape[1], qso['pca'].shape[2])

        # get the shortest wavelength range
        wave_min = max(wave.min(), wave_gal.min(), wave_qso.min())
        wave_max = min(wave.max(), wave_gal.max(), wave_qso.max())

        ind_data = np.where((wave > wave_min) & (wave < wave_max), True, False)
        ind_gal = np.where((wave_gal > wave_min-1.) & (wave_gal < wave_max+1.), True, False)
        ind_qso = np.where((wave_qso > wave_min-1.) & (wave_qso < wave_max+1.), True, False)

        flux_gal_new = np.zeros(flux_gal.shape[0]*flux[ind_data].shape[0]).reshape(flux_gal.shape[0],
                                                                                   flux[ind_data].shape[0])
        flux_qso_new = np.zeros(flux_qso.shape[0]*flux[ind_data].shape[0]).reshape(flux_qso.shape[0],
                                                                                   flux[ind_data].shape[0])
        for i in range(flux_gal.shape[0]):
            fgal = interpolate.interp1d(wave_gal[ind_gal], flux_gal[i, ind_gal], bounds_error=False, fill_value=0)
            flux_gal_new[i, :] = fgal(wave[ind_data])
        for i in range(flux_qso.shape[0]):
            fqso = interpolate.interp1d(wave_qso[ind_qso], flux_qso[i, ind_qso], bounds_error=False, fill_value=0)
            flux_qso_new[i, :] = fqso(wave[ind_data])

        wave_new = wave[ind_data]
        flux_new = flux[ind_data]
        err_new = err[ind_data]

        flux_temp = np.vstack((flux_gal_new[0:npca_gal, :], flux_qso_new[0:npca_qso, :]))
        res = np.linalg.lstsq(flux_temp.T, flux_new)[0]

        host_flux = np.dot(res[0:npca_gal], flux_temp[0:npca_gal])
        qso_flux = np.dot(res[npca_gal:], flux_temp[npca_gal:])

        data_cube = np.vstack((wave_new, flux_new, err_new, host_flux, qso_flux))

        ind_f4200 = np.where((wave_new > 4160.) & (wave_new < 4210.), True, False)
        frac_host_4200 = np.sum(host_flux[ind_f4200])/np.sum(flux_new[ind_f4200])
        ind_f5100 = np.where((wave_new > 5080.) & (wave_new < 5130.), True, False)
        frac_host_5100 = np.sum(host_flux[ind_f5100])/np.sum(flux_new[ind_f5100])

        return data_cube  # ,frac_host_4200,frac_host_5100

    def _DoContiFit(self, wave, flux, err, ra, dec, plateid, mjd, fiberid):
        """Fit the continuum with PL, Polynomial, UV/optical FeII, Balmer continuum"""
        global fe_uv, fe_op
        fe_uv = np.genfromtxt(self.path+'fe_uv.txt')
        fe_op = np.genfromtxt(self.path+'fe_optical.txt')

        # do continuum fit--------------------------
        window_all = np.array(
            [[1150., 1170.], [1275., 1290.], [1350., 1360.], [1445., 1465.], [1690., 1705.], [1770., 1810.],
             [1970., 2400.], [2480., 2675.], [2925., 3400.], [3775., 3832.], [4000., 4050.], [4200., 4230.],
             [4435., 4640.], [5100., 5535.], [6005., 6035.], [6110., 6250.], [6800., 7000.], [7160., 7180.],
             [7500., 7800.], [8050., 8150.], ])

        tmp_all = np.array([np.repeat(False, len(wave))]).flatten()
        for jj in range(len(window_all)):
            tmp = np.where((wave > window_all[jj, 0]) & (wave < window_all[jj, 1]), True, False)
            tmp_all = np.any([tmp_all, tmp], axis=0)

        if wave[tmp_all].shape[0] < 10:
            print('Continuum fitting pixel < 10.  ')

        # set initial paramiters for continuum
        if self.initial_guess is not None:
            pp0 = self.initial_guess
        else:
            pp0 = np.array([0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 15000., 0.5, 0., 0., 0.])

        conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(wave[tmp_all], flux[tmp_all], err[tmp_all]))
        tmp_parinfo = [{'limits': (0., 10.**10)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                       {'limits': (0., 10.**10)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                       {'limits': (0., 10.**10)}, {'limits': (-5., 3.)}, {'limits': (0., 10.**10)},
                       {'limits': (10000., 50000.)}, {'limits': (0.1, 2.)}, None, None, None, ]
        conti_fit.parinfo = tmp_parinfo
        conti_fit.fit(params0=pp0)

        # Perform one iteration to remove 3sigma pixel below the first continuum fit
        # to avoid the continuum windows falls within a BAL trough
        if self.rej_abs == True:
            if self.poly == True:
                tmp_conti = (conti_fit.params[6]*(wave[tmp_all]/3000.0)**conti_fit.params[7]+self.F_poly_conti(
                    wave[tmp_all], conti_fit.params[11:]))
            else:
                tmp_conti = (conti_fit.params[6]*(wave[tmp_all]/3000.0)**conti_fit.params[7])
            ind_noBAL = ~np.where(((flux[tmp_all] < tmp_conti-3.*err[tmp_all]) & (wave[tmp_all] < 3500.)), True, False)
            f = kmpfit.Fitter(residuals=self._residuals, data=(
                wave[tmp_all][ind_noBAL], self.Smooth(flux[tmp_all][ind_noBAL], 10), err[tmp_all][ind_noBAL]))
            conti_fit.parinfo = tmp_parinfo
            conti_fit.fit(params0=pp0)

        # calculate continuum luminoisty
        L = self._L_conti(wave, conti_fit.params)

        # calculate the Fe EW
        Fe_EW_val = self.Get_Fe_EW(conti_fit.params[:6], np.append(conti_fit.params[6:8], conti_fit.params[11:]))

        # calculate FeII flux
        Fe_flux_result = np.array([])
        Fe_flux_type = np.array([])
        Fe_flux_name = np.array([])
        if self.Fe_flux_range is not None:
            Fe_flux_result, Fe_flux_type, Fe_flux_name = self.Get_Fe_flux(self.Fe_flux_range, conti_fit.params[:6])

        # get conti result -----------------------------

        if self.if_save_spec:
            if self.rej_abs == True:
                self.wave_conti = self.wave[tmp_all][ind_noBAL]
                self.flux_conti = self.flux[tmp_all][ind_noBAL]
                self.err_conti = self.err[tmp_all][ind_noBAL]
            else:
                self.wave_conti = self.wave[tmp_all]
                self.flux_conti = self.flux[tmp_all]
                self.err_conti = self.err[tmp_all]

        if self.MC == True and self.n_trails > 0:
            # calculate MC err
            if self.if_dump_MC_result==True:
                conti_para_MC, all_L_MC, Fe_flux_MC, Fe_EW_MC, conti_para_std, all_L_std, Fe_flux_std, \
                    conti_para_std2, all_L_std2, Fe_flux_std2, Fe_EW_std2  = self._conti_mc(self.wave[tmp_all], self.flux[tmp_all],
                                                                                            self.err[tmp_all], pp0, conti_fit.parinfo,
                                                                                            self.n_trails, self.if_dump_MC_result)
            else:
                conti_para_std, all_L_std, Fe_flux_std, conti_para_std2, all_L_std2, Fe_flux_std2, Fe_EW_std2  = self._conti_mc(self.wave[tmp_all], self.flux[tmp_all],
                                                                                self.err[tmp_all], pp0, conti_fit.parinfo,
                                                                                self.n_trails, self.if_dump_MC_result)
            #DL = cosmo.luminosity_distance(self.z).value*10**6*3.086*10**18
            self.conti_result = np.array(
                [ra, dec, str(plateid), str(mjd), str(fiberid), '%04d'%plateid+'-%05d'%mjd+'-%04d'%fiberid, \
                 self.z, self.SN_ratio_conti, self.EBV, self.n_trails,
                 conti_fit.chi2_min, conti_fit.rchi2_min, conti_fit.dof, conti_fit.dof+conti_fit.nfree,
                 conti_fit.params[0], conti_para_std[0], conti_para_std2[0],
                 conti_fit.params[1], conti_para_std[1], conti_para_std2[1],
                 conti_fit.params[2], conti_para_std[2], conti_para_std2[2],
                 conti_fit.params[3], conti_para_std[3], conti_para_std2[3],
                 conti_fit.params[4], conti_para_std[4], conti_para_std2[4],
                 conti_fit.params[5], conti_para_std[5], conti_para_std2[5],
                 Fe_EW_val[0], Fe_EW_val[1], Fe_EW_std2[0], Fe_EW_std2[1],
                 conti_fit.params[6], conti_para_std[6], conti_para_std2[6],
                 conti_fit.params[7], conti_para_std[7], conti_para_std2[7],
                 conti_fit.params[8], conti_para_std[8], conti_para_std2[8],
                 conti_fit.params[9], conti_para_std[9], conti_para_std2[9],
                 conti_fit.params[10], conti_para_std[10], conti_para_std2[10],
                 conti_fit.params[11], conti_para_std[11], conti_para_std2[11],
                 conti_fit.params[12], conti_para_std[12], conti_para_std2[12],
                 conti_fit.params[13], conti_para_std[13], conti_para_std2[13],
                 L[0], all_L_std[0], all_L_std2[0],
                 L[1], all_L_std[1], all_L_std2[1],
                 L[2], all_L_std[2], all_L_std2[2],
                 L[3], all_L_std[3], all_L_std2[3]])
            self.conti_result_type = np.array(
                ['float', 'float', 'int', 'int', 'int', 'str', \
                 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', \
                 'float', 'float', 'float', # conti_para 0
                 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', # cconti_para 5
                 'float', 'float', 'float', 'float', # Fe EW
                 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', # conti_para 10
                 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', 'float', 'float', 'float', # LOGL
                 'float', 'float', 'float', 'float', 'float', 'float'])
            self.conti_result_name = np.array(
                ['RA', 'DEC', 'PLATE', 'MJD', 'FIBERID', 'ObjID', \
                 'Z_FIT', 'SNR_conti', 'EBV', 'MC_trails',
                 'conti_chi2', 'conti_rchi2', 'conti_dof', 'conti_npix',
                 'Fe_uv_norm', 'Fe_uv_norm_err', 'Fe_uv_norm_err2',
                 'Fe_uv_FWHM', 'Fe_uv_FWHM_err', 'Fe_uv_FWHM_err2',
                 'Fe_uv_shift', 'Fe_uv_shift_err', 'Fe_uv_shift_err2',
                 'Fe_op_norm', 'Fe_op_norm_err', 'Fe_op_norm_err2',
                 'Fe_op_FWHM', 'Fe_op_FWHM_err', 'Fe_op_FWHM_err2',
                 'Fe_op_shift', 'Fe_op_shift_err', 'Fe_op_shift_err2',
                 'Fe_uv_EW', 'Fe_op_EW', 'Fe_uv_EW_err', 'Fe_op_EW_err',
                 'PL_norm', 'PL_norm_err', 'PL_norm_err2',
                 'PL_slope', 'PL_slope_err', 'PL_slope_err2',
                 'Blamer_norm', 'Blamer_norm_err', 'Blamer_norm_err2',
                 'Balmer_Te', 'Balmer_Te_err', 'Balmer_Te_err2',
                 'Balmer_Tau', 'Balmer_Tau_err', 'Balmer_Tau_err2',
                 'POLY_a', 'POLY_a_err', 'POLY_a_err2',
                 'POLY_b', 'POLY_b_err', 'POLY_b_err2',
                 'POLY_c', 'POLY_c_err', 'POLY_c_err2', \
                 'L1350', 'L1350_err', 'L1350_err2', 'L1700', 'L1700_err', 'L1700_err2',
                 'L3000', 'L3000_err', 'L3000_err2', 'L5100', 'L5100_err', 'L5100_err2'])

            for iii in range(Fe_flux_result.shape[0]):
                self.conti_result = np.append(self.conti_result, [Fe_flux_result[iii], Fe_flux_std[iii], Fe_flux_std2[iii]])
                self.conti_result_type = np.append(self.conti_result_type, [Fe_flux_type[iii], Fe_flux_type[iii], Fe_flux_type[iii]])
                self.conti_result_name = np.append(self.conti_result_name,
                                                   [Fe_flux_name[iii], Fe_flux_name[iii]+'_err', Fe_flux_name[iii]+'_err2'])

            if self.if_dump_MC_result == True:
                self.conti_result_MC_name = np.array([['Fe_uv_norm_MC', 'Fe_uv_FWHM_MC', 'Fe_uv_shift_MC', \
                                                       'Fe_op_norm_MC', 'Fe_op_FWHM_MC', 'Fe_op_shift_MC', \
                                                        'PL_norm_MC', 'PL_slope_MC', 'Blamer_norm_MC', 'Balmer_Te_MC', 'Balmer_Tau_MC', \
                                                        'POLY_a_MC', 'POLY_b_MC', 'POLY_c_MC'], \
                                                       ['Fe_uv_EW_MC', 'Fe_op_EW_MC'], \
                                                       ['L1350_MC', 'L1700_MC', 'L3000_MC', 'L5100_MC'], \
                                                       [nn+'_MC' for nn in Fe_flux_name]])
                self.conti_result_MC = [conti_para_MC, Fe_EW_MC, all_L_MC, Fe_flux_MC]

        else:
            self.conti_result = np.array(
                [ra, dec, str(plateid), str(mjd), str(fiberid), '%04d'%plateid+'-%05d'%mjd+'-%04d'%fiberid, \
                 self.z, self.SN_ratio_conti, self.EBV,
                 conti_fit.chi2_min, conti_fit.rchi2_min, conti_fit.dof, conti_fit.dof+conti_fit.nfree,
                 conti_fit.params[0], conti_fit.params[1], conti_fit.params[2], conti_fit.params[3],
                 conti_fit.params[4], conti_fit.params[5], Fe_EW_val[0], Fe_EW_val[1],
                 conti_fit.params[6], conti_fit.params[7], conti_fit.params[8], conti_fit.params[9], conti_fit.params[10], \
                 conti_fit.params[11], conti_fit.params[12], conti_fit.params[13], \
                 L[0], L[1], L[2], L[3]])
            self.conti_result_type = np.array(
                ['float', 'float', 'int', 'int', 'int', 'str', \
                 'float', 'float', 'float',\
                 'float', 'float', 'float', 'float', # conti_fit_stat
                 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', # Fe property
                 'float', 'float', 'float', 'float', 'float', \
                 'float', 'float', 'float', \
                 'float', 'float', 'float', 'float'])
            self.conti_result_name = np.array(
                ['RA', 'DEC', 'PLATE', 'MJD', 'FIBERID', 'ObjID', 'Z_FIT', 'SNR_conti', 'EBV', \
                 'conti_chi2', 'conti_rchi2', 'conti_dof', 'conti_npix',
                 'Fe_uv_norm', 'Fe_uv_FWHM', 'Fe_uv_shift', 'Fe_op_norm', 'Fe_op_FWHM', 'Fe_op_shift', 'Fe_uv_EW', 'Fe_op_EW',\
                 'PL_norm', 'PL_slope', 'Blamer_norm', 'Balmer_Te', 'Balmer_Tau', \
                 'POLY_a', 'POLY_b', 'POLY_c', 'LOGL1350', 'LOGL1700', 'LOGL3000', 'LOGL5100'])
            self.conti_result = np.append(self.conti_result, Fe_flux_result)
            self.conti_result_type = np.append(self.conti_result_type, Fe_flux_type)
            self.conti_result_name = np.append(self.conti_result_name, Fe_flux_name)

        self.conti_fit = conti_fit
        self.tmp_all = tmp_all

        # save different models--------------------
        f_fe_mgii_model = self.Fe_flux_mgii(wave, conti_fit.params[0:3])
        f_fe_balmer_model = self.Fe_flux_balmer(wave, conti_fit.params[3:6])
        f_pl_model = conti_fit.params[6]*(wave/3000.0)**conti_fit.params[7]
        f_bc_model = self.Balmer_conti(wave, conti_fit.params[8:11])
        f_poly_model = self.F_poly_conti(wave, conti_fit.params[11:])
        f_conti_model = (f_pl_model+f_fe_mgii_model+f_fe_balmer_model+f_poly_model+f_bc_model)
        line_flux = flux-f_conti_model

        self.f_conti_model = f_conti_model
        self.f_bc_model = f_bc_model
        self.f_fe_uv_model = f_fe_mgii_model
        self.f_fe_op_model = f_fe_balmer_model
        self.f_pl_model = f_pl_model
        self.f_poly_model = f_poly_model
        self.line_flux = line_flux
        self.PL_poly_BC = f_pl_model+f_poly_model+f_bc_model

        return self.conti_result, self.conti_result_name

    def _L_conti(self, wave, pp):
        """Calculate continuum Luminoisity at 1350,1700,3000,5100A"""
        conti_flux = pp[6]*(wave/3000.0)**pp[7]+self.F_poly_conti(wave, pp[11:])
        # plt.plot(wave,conti_flux)
        L = np.array([])
        for LL in zip([1350., 1700., 3000., 5100.]):
            if wave.max() > LL[0] and wave.min() < LL[0]:
                L_tmp = np.asarray([np.log10(
                    LL[0]*self.Flux2L(conti_flux[np.where(abs(wave-LL[0]) < 5., True, False)].mean(), self.z))])
            else:
                L_tmp = np.array([0.])
            L = np.concatenate([L, L_tmp])  # save log10(L1350,L3000,L5100)
        return L

    def _f_conti_all(self, xval, pp):
        """
        Continuum components described by 14 parameters
         pp[0]: norm_factor for the MgII Fe_template
         pp[1]: FWHM for the MgII Fe_template
         pp[2]: small shift of wavelength for the MgII Fe template
         pp[3:5]: same as pp[0:2] but for the Hbeta/Halpha Fe template
         pp[6]: norm_factor for continuum f_lambda = (lambda/3000.0)^{-alpha}
         pp[7]: slope for the power-law continuum
         pp[8:10]: norm, Te and Tau_e for the Balmer continuum at <3646 A
         pp[11:13]: polynomial for the continuum
        """
        # iron flux for MgII line region
        f_Fe_MgII = self.Fe_flux_mgii(xval, pp[0:3])
        # iron flux for balmer line region
        f_Fe_Balmer = self.Fe_flux_balmer(xval, pp[3:6])
        # power-law continuum
        f_pl = pp[6]*(xval/3000.0)**pp[7]
        # Balmer continuum
        f_conti_BC = self.Balmer_conti(xval, pp[8:11])
        # polynormal conponent for reddened spectra
        f_poly = self.F_poly_conti(xval, pp[11:])

        if self.Fe_uv_op == True and self.poly == False and self.BC == False:
            yval = f_pl+f_Fe_MgII+f_Fe_Balmer
        elif self.Fe_uv_op == True and self.poly == True and self.BC == False:
            yval = f_pl+f_Fe_MgII+f_Fe_Balmer+f_poly
        elif self.Fe_uv_op == True and self.poly == False and self.BC == True:
            yval = f_pl+f_Fe_MgII+f_Fe_Balmer+f_conti_BC
        elif self.Fe_uv_op == False and self.poly == True and self.BC == False:
            yval = f_pl+f_poly
        elif self.Fe_uv_op == False and self.poly == False and self.BC == False:
            yval = f_pl
        elif self.Fe_uv_op == False and self.poly == False and self.BC == True:
            yval = f_pl+f_conti_BC
        elif self.Fe_uv_op == True and self.poly == True and self.BC == True:
            yval = f_pl+f_Fe_MgII+f_Fe_Balmer+f_poly+f_conti_BC
        elif self.Fe_uv_op == False and self.poly == True and self.BC == True:
            yval = f_pl+f_Fe_Balmer+f_poly+f_conti_BC
        else:
            raise RuntimeError('No this option for Fe_uv_op, poly and BC!')
        return yval

    def _residuals(self, pp, data):
        """Continual residual function used in kmpfit"""
        xval, yval, weight = data
        return (yval-self._f_conti_all(xval, pp))/weight

    # ---------MC error for continuum parameters-------------------
    def _conti_mc(self, x, y, err, pp0, pp_limits, n_trails, iif_dump_MC_result):
        """Calculate the continual parameters' Monte carlo errrors"""
        all_para = np.zeros((len(pp0), n_trails))
        all_Fe_EW = np.zeros((2, n_trails))
        all_L = np.zeros((4, n_trails))

        n_Fe_flux = np.array(self.Fe_flux_range).flatten().shape[0]//2
        all_Fe_flux = np.zeros((n_Fe_flux, n_trails))

        for tra in range(n_trails):
            flux = y+np.random.randn(len(y))*err
            conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(x, flux, err), maxiter=50)
            conti_fit.parinfo = pp_limits
            conti_fit.fit(params0=pp0)
            all_para[:, tra] = conti_fit.params
            all_Fe_EW[:,tra] = self.Get_Fe_EW(conti_fit.params[:6], np.append(conti_fit.params[6:8], conti_fit.params[11:]))
            all_L[:, tra] = np.asarray(self._L_conti(x, conti_fit.params))

            if self.Fe_flux_range is not None:
                Fe_flux_result, Fe_flux_type, Fe_flux_name = self.Get_Fe_flux(self.Fe_flux_range, conti_fit.params[:6])
                all_Fe_flux[:, tra] = Fe_flux_result

        all_para_std = all_para.std(axis=1)
        all_L_std = all_L.std(axis=1)
        all_Fe_flux_std = all_Fe_flux.std(axis=1)

        all_para_std2 = (np.percentile(all_para,84,axis=1)-np.percentile(all_para,16,axis=1))/2
        all_L_std2 = (np.percentile(all_L,84,axis=1)-np.percentile(all_L,16,axis=1))/2
        all_Fe_EW_std2 = (np.percentile(all_Fe_EW,84,axis=1)-np.percentile(all_Fe_EW,16,axis=1))/2
        all_Fe_flux_std2 = (np.percentile(all_Fe_flux,84,axis=1)-np.percentile(all_Fe_flux,16,axis=1))/2

        if iif_dump_MC_result == True:
            return all_para, all_L, all_Fe_flux, all_Fe_EW, \
                   all_para_std, all_L_std, all_Fe_flux_std, \
                   all_para_std2, all_L_std2, all_Fe_flux_std2, all_Fe_EW_std2
        else:
            return all_para_std, all_L_std, all_Fe_flux_std, \
                   all_para_std2, all_L_std2, all_Fe_flux_std2, all_Fe_EW_std2

    # line function-----------
    def _DoLineFit(self, wave, line_flux, err, f):
        """Fit the emission lines with Gaussian profile """

        # remove abosorbtion line in emission line region
        # remove the pixels below continuum

        # read line parameter
        linepara = fits.open(self.path+'qsopar.fits')
        linelist = linepara[1].data
        self.linelist = linelist

        ind_kind_line = np.where((linelist['lambda'] > wave.min()) & (linelist['lambda'] < wave.max()), True, False)
        if ind_kind_line.any() == True:
            # sort complex name with line wavelength
            uniq_linecomp, uniq_ind = np.unique(linelist['compname'][ind_kind_line], return_index=True)
            uniq_linecomp_sort = uniq_linecomp[linelist['lambda'][ind_kind_line][uniq_ind].argsort()]
            nncomp = len(uniq_linecomp_sort)
            ncomp = 0
            compname = linelist['compname']
            allcompcenter = np.sort(linelist['lambda'][ind_kind_line][uniq_ind])

            # loop over each complex and fit n lines simutaneously

            comp_result = np.array([])
            comp_result_type = np.array([])
            comp_result_name = np.array([])
            gauss_result = np.array([])
            gauss_result_type = np.array([])
            gauss_result_name = np.array([])
            all_comp_range = np.array([])
            if self.if_dump_MC_result == True:
                gauss_MCresult = []
                gauss_MCresult_name = []
            if self.if_read_line_prop == True:
                fur_result = np.array([])
                fur_result_type = np.array([])
                fur_result_name = np.array([])

            for ii in range(nncomp):
                compcenter = allcompcenter[ii]
                ind_line = np.where(linelist['compname'] == uniq_linecomp_sort[ii], True, False)  # get line index
                nline_fit = np.sum(ind_line)  # n line in one complex
                linelist_fit = linelist[ind_line]
                # n gauss in each line
                ngauss_fit = np.asarray(linelist_fit['ngauss'], dtype=int)

                # for iitmp in range(nline_fit):   # line fit together
                comp_range = [linelist_fit[0]['minwav'], linelist_fit[0]['maxwav']]  # read complex range from table

                # ----tie lines--------
                self._do_tie_line(linelist, ind_line)

                # cut the pixel within the complex range
                ind_n = np.where((wave > comp_range[0]) & (wave < comp_range[1]))[0]

                if (len(ind_n) > 10) and  (len(ind_n) > np.sum(ngauss_fit)*3):
                    all_comp_range = np.concatenate([all_comp_range, comp_range])
                    # call kmpfit for lines
                    line_fit = self._do_line_kmpfit(linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit)
                    ncomp += 1

                    # reject absorption in emission region with maximum interation
                    if self.rej_line_abs ==  True:
                        niter = 0
                        ind_n_tmp = np.copy(ind_n)
                        ind_rej = np.array([], dtype = 'int')
                        while (niter < self.rej_line_max_niter):
                            ind_line_abs = np.where((line_flux[ind_n_tmp]-self.Manygauss(np.log(wave[ind_n_tmp]), line_fit.params) < -3.*err[ind_n_tmp]) \
                                                     , False, True)
                            if len(ind_n_tmp[ind_line_abs])-10 < np.sum(ngauss_fit)*3:
                                break
                            else:
                                line_fit_rej = self._do_line_kmpfit(linelist, line_flux, ind_line, ind_n_tmp[ind_line_abs], nline_fit, ngauss_fit)
                                if line_fit_rej.rchi2_min >= line_fit.rchi2_min:
                                    break
                                else:
                                    niter += 1
                                    line_fit = line_fit_rej
                                    ind_rej = np.append(ind_rej, ind_n_tmp[~ind_line_abs])
                                    ind_n_tmp = ind_n_tmp[ind_line_abs]

                        ind_n = ind_n_tmp
                        self.rej_line_abs_wave = np.concatenate([self.rej_line_abs_wave, self.wave[ind_rej]])
                        self.rej_line_abs_flux = np.concatenate([self.rej_line_abs_flux, line_flux[ind_rej]])
                        self.rej_line_abs_err = np.concatenate([self.rej_line_abs_err, self.err[ind_rej]])

                    if self.if_save_spec:
                        self.linespec_wave = np.concatenate([self.linespec_wave, self.wave[ind_n]])
                        self.linespec_lineflux = np.concatenate([self.linespec_lineflux, line_flux[ind_n]])
                        self.linespec_err = np.concatenate([self.linespec_err, self.err[ind_n]])

                    # calculate MC err
                    if self.MC == True and self.n_trails > 0:
                        if self.if_dump_MC_result == True and self.if_read_line_prop == False:
                            all_para_val, all_para_std, all_para_std2  = self._line_mc( np.log(wave[ind_n]), line_flux[ind_n], err[ind_n], \
                                self.line_fit_ini, self.line_fit_par, self.n_trails, compcenter, self.if_read_line_prop, self.if_dump_MC_result)
                        elif self.if_dump_MC_result == True and self.if_read_line_prop == True:
                            all_para_val, all_para_std, fwhm_std, sigma_std, ew_std, peak_std, area_std, \
                            all_para_std2, fwhm_std2, sigma_std2, ew_std2, peak_std2, area_std2 = self._line_mc(
                                np.log(wave[ind_n]), line_flux[ind_n], err[ind_n], self.line_fit_ini, self.line_fit_par,
                                self.n_trails, compcenter, self.if_read_line_prop, self.if_dump_MC_result)
                        elif self.if_read_line_prop == True:
                                all_para_std, fwhm_std, sigma_std, ew_std, peak_std, area_std, \
                                all_para_std2, fwhm_std2, sigma_std2, ew_std2, peak_std2, area_std2 = self._line_mc(
                                    np.log(wave[ind_n]), line_flux[ind_n], err[ind_n], self.line_fit_ini, self.line_fit_par,
                                    self.n_trails, compcenter, self.if_read_line_prop, self.if_dump_MC_result)
                        else:
                            all_para_std, all_para_std2 = self._line_mc( np.log(wave[ind_n]), line_flux[ind_n], err[ind_n], \
                                self.line_fit_ini, self.line_fit_par, self.n_trails, compcenter, self.if_read_line_prop, self.if_dump_MC_result)


                    # ----------------------get line fitting results----------------------
                    # complex parameters

                    # tie lines would reduce the number of parameters increasing the dof
                    dof_fix = 0
                    if self.tie_lambda == True:
                        dof_fix += np.max((len(self.ind_tie_vindex1), 1))-1
                        dof_fix += np.max((len(self.ind_tie_vindex2), 1))-1
                        dof_fix += np.max((len(self.ind_tie_vindex3), 1))-1
                        dof_fix += np.max((len(self.ind_tie_vindex4), 1))-1
                    if self.tie_width == True:
                        dof_fix += np.max((len(self.ind_tie_windex1), 1))-1
                        dof_fix += np.max((len(self.ind_tie_windex2), 1))-1
                        dof_fix += np.max((len(self.ind_tie_windex3), 1))-1
                        dof_fix += np.max((len(self.ind_tie_windex4), 1))-1
                    if self.tie_flux_1 == True:
                        dof_fix += np.max((len(self.ind_tie_findex1), 1))-1
                        dof_fix += np.max((len(self.ind_tie_findex2), 1))-1
                        dof_fix += np.max((len(self.ind_tie_findex3), 1))-1
                        dof_fix += np.max((len(self.ind_tie_findex4), 1))-1

                    comp_result_tmp = np.array(
                        [[linelist['compname'][ind_line][0]], [line_fit.status], [line_fit.chi2_min],
                         [line_fit.chi2_min/(line_fit.dof+dof_fix)], [line_fit.niter], \
                         [line_fit.dof+line_fit.nfree], [line_fit.dof+dof_fix]]).flatten()
                    comp_result_type_tmp = np.array(['str', 'int', 'float', \
                                                    'float', 'float', 'float', 'float'])
                    comp_result_name_tmp = np.array(
                        [str(ncomp)+'_complex_name', str(ncomp)+'_line_status', str(ncomp)+'_line_min_chi2',
                         str(ncomp)+'_line_red_chi2', str(ncomp)+'_niter', \
                         str(ncomp)+'_npix', str(ncomp)+'_ndof'])
                    comp_result = np.concatenate([comp_result, comp_result_tmp])
                    comp_result_name = np.concatenate([comp_result_name, comp_result_name_tmp])
                    comp_result_type = np.concatenate([comp_result_type, comp_result_type_tmp])

                    # gauss result -------------

                    gauss_tmp = np.array([])
                    gauss_type_tmp = np.array([])
                    gauss_name_tmp = np.array([])

                    if self.if_dump_MC_result == True:
                        gauss_MCval_tmp = []
                        gauss_MCname_tmp = []

                    for gg in range(len(line_fit.params)):
                        gauss_tmp = np.concatenate([gauss_tmp, np.array([line_fit.params[gg]])])
                        if self.MC == True and self.n_trails > 0:
                            gauss_tmp = np.concatenate([gauss_tmp, np.array([all_para_std[gg]]), np.array([all_para_std2[gg]])])
                            if self.if_dump_MC_result == True:
                                gauss_MCval_tmp.append(all_para_val[gg,:])
                    gauss_result = np.concatenate([gauss_result, gauss_tmp])
                    if self.if_dump_MC_result == True:
                        gauss_MCresult.append([gauss_MCval_tmp])

                    # gauss result name -----------------
                    for n in range(nline_fit):
                        for nn in range(int(ngauss_fit[n])):
                            line_name = linelist['linename'][ind_line][n]+'_'+str(nn+1)
                            if self.MC == True and self.n_trails > 0:
                                gauss_type_tmp_tmp = ['float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
                                gauss_name_tmp_tmp = [line_name+'_scale', line_name+'_scale_err', line_name+'_scale_err2',
                                                      line_name+'_centerwave', line_name+'_centerwave_err', line_name+'_centerwave_err2',
                                                      line_name+'_sigma', line_name+'_sigma_err', line_name+'_sigma_err2']
                                if self.if_dump_MC_result == True:
                                    gauss_MCname_tmp_tmp = [line_name+'_scale_MC', line_name+'_centerwave_MC', line_name+'_sigma_MC']
                            else:
                                gauss_type_tmp_tmp = ['float', 'float', 'float']
                                gauss_name_tmp_tmp = [line_name+'_scale', line_name+'_centerwave', line_name+'_sigma']
                            gauss_name_tmp = np.concatenate([gauss_name_tmp, gauss_name_tmp_tmp])
                            gauss_type_tmp = np.concatenate([gauss_type_tmp, gauss_type_tmp_tmp])
                            if (self.if_dump_MC_result == True) and (self.MC == True and self.n_trails > 0):
                                gauss_MCname_tmp.append(gauss_MCname_tmp_tmp)

                    gauss_result_type = np.concatenate([gauss_result_type, gauss_type_tmp])
                    gauss_result_name = np.concatenate([gauss_result_name, gauss_name_tmp])
                    if  (self.if_dump_MC_result == True) and (self.MC == True and self.n_trails > 0):
                        gauss_MCresult_name.append(gauss_MCname_tmp)

                    # further line parameters ----------
                    if self.if_read_line_prop == True:
                        fur_result_tmp = np.array([])
                        fur_result_type_tmp = np.array([])
                        fur_result_name_tmp = np.array([])
                        fwhm, sigma, ew, peak, area = self.line_prop(compcenter, line_fit.params, 'broad')
                        br_name = uniq_linecomp_sort[ii]

                        if self.MC == True and self.n_trails > 0:
                            fur_result_tmp = np.array(
                                [fwhm, fwhm_std, fwhm_std2, sigma, sigma_std, sigma_std2, ew, ew_std, ew_std2,
                                peak, peak_std, peak_std2, area, area_std, area_std2])
                            fur_result_type_tmp = np.concatenate([fur_result_type_tmp,
                                                                  ['float', 'float', 'float', 'float', 'float', 'float',
                                                                   'float', 'float', 'float', 'float', 'float', 'float',
                                                                   'float', 'float', 'float']])
                            fur_result_name_tmp = np.array(
                                [br_name+'_whole_br_fwhm', br_name+'_whole_br_fwhm_err', br_name+'_whole_br_fwhm_err2',
                                 br_name+'_whole_br_sigma', br_name+'_whole_br_sigma_err', br_name+'_whole_br_sigma_err2',
                                 br_name+'_whole_br_ew', br_name+'_whole_br_ew_err', br_name+'_whole_br_ew_err2',
                                 br_name+'_whole_br_peak', br_name+'_whole_br_peak_err', br_name+'_whole_br_peak_err2',
                                 br_name+'_whole_br_area', br_name+'_whole_br_area_err', br_name+'_whole_br_area_err2'])
                        else:
                            fur_result_tmp = np.array([fwhm, sigma, ew, peak, area])
                            fur_result_type_tmp = np.concatenate(
                                [fur_result_type_tmp, ['float', 'float', 'float', 'float', 'float']])
                            fur_result_name_tmp = np.array(
                                [br_name+'_whole_br_fwhm', br_name+'_whole_br_sigma', br_name+'_whole_br_ew',
                                 br_name+'_whole_br_peak', br_name+'_whole_br_area'])
                        fur_result = np.concatenate([fur_result, fur_result_tmp])
                        fur_result_type = np.concatenate([fur_result_type, fur_result_type_tmp])
                        fur_result_name = np.concatenate([fur_result_name, fur_result_name_tmp])

                else:
                    print("less than 10 pixels in line fitting!")

            if self.if_read_line_prop == True:
                line_result = np.concatenate([comp_result, gauss_result, fur_result])
                line_result_type = np.concatenate([comp_result_type, gauss_result_type, fur_result_type])
                line_result_name = np.concatenate([comp_result_name, gauss_result_name, fur_result_name])
            else:
                line_result = np.concatenate([comp_result, gauss_result])
                line_result_type = np.concatenate([comp_result_type, gauss_result_type])
                line_result_name = np.concatenate([comp_result_name, gauss_result_name])

        else:
            line_result = np.array([])
            line_result_name = np.array([])
            comp_result = np.array([])
            gauss_result = np.array([])
            gauss_result_name = np.array([])
            line_result_type = np.array([])
            ncomp = 0
            all_comp_range = np.array([])
            uniq_linecomp_sort = np.array([])
            print("No line to fit! Pleasse set Line_fit to FALSE or enlarge wave_range!")

        self.comp_result = comp_result
        self.gauss_result = gauss_result
        self.gauss_result_name = gauss_result_name
        self.line_result = line_result
        self.line_result_type = line_result_type
        self.line_result_name = line_result_name
        self.ncomp = ncomp
        self.line_flux = line_flux
        self.all_comp_range = all_comp_range
        self.uniq_linecomp_sort = uniq_linecomp_sort
        if self.if_dump_MC_result == True:
            self.gauss_MCresult = gauss_MCresult
            self.gauss_MCresult_name = gauss_MCresult_name
        return self.line_result, self.line_result_name

    def _do_line_kmpfit(self, linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit):
        """The key function to do the line fit with kmpfit"""
        line_fit = kmpfit.Fitter(self._residuals_line, data=(
            np.log(self.wave[ind_n]), line_flux[ind_n], self.err[ind_n]))  # fitting wavelength in ln space
        line_fit_ini = np.array([])
        line_fit_par = np.array([])
        for n in range(nline_fit):
            for nn in range(ngauss_fit[n]):
                # set up initial parameter guess
                line_fit_ini0 = [0., np.log(linelist['lambda'][ind_line][n]), linelist['inisig'][ind_line][n]]
                line_fit_ini = np.concatenate([line_fit_ini, line_fit_ini0])
                # set up parameter limits
                lambda_low = np.log(linelist['lambda'][ind_line][n])-linelist['voff'][ind_line][n]
                lambda_up = np.log(linelist['lambda'][ind_line][n])+linelist['voff'][ind_line][n]
                sig_low = linelist['minsig'][ind_line][n]
                sig_up = linelist['maxsig'][ind_line][n]
                line_fit_par0 = [{'limits': (0., 10.**10)}, {'limits': (lambda_low, lambda_up)},
                                 {'limits': (sig_low, sig_up)}]
                line_fit_par = np.concatenate([line_fit_par, line_fit_par0])

        line_fit.parinfo = line_fit_par
        line_fit.fit(params0=line_fit_ini)
        line_fit.params = self.newpp
        self.line_fit = line_fit
        self.line_fit_ini = line_fit_ini
        self.line_fit_par = line_fit_par
        return line_fit

    def _do_tie_line(self, linelist, ind_line):
        """Tie line's central"""
        # --------------- tie parameter-----------
        # so far, only four groups of each properties are support for tying
        ind_tie_v1 = np.where(linelist['vindex'][ind_line] == 1., True, False)
        ind_tie_v2 = np.where(linelist['vindex'][ind_line] == 2., True, False)
        ind_tie_v3 = np.where(linelist['vindex'][ind_line] == 3., True, False)
        ind_tie_v4 = np.where(linelist['vindex'][ind_line] == 4., True, False)

        ind_tie_w1 = np.where(linelist['windex'][ind_line] == 1., True, False)
        ind_tie_w2 = np.where(linelist['windex'][ind_line] == 2., True, False)
        ind_tie_w3 = np.where(linelist['windex'][ind_line] == 3., True, False)
        ind_tie_w4 = np.where(linelist['windex'][ind_line] == 4., True, False)

        ind_tie_f1 = np.where(linelist['findex'][ind_line] == 1., True, False)
        ind_tie_f2 = np.where(linelist['findex'][ind_line] == 2., True, False)
        ind_tie_f3 = np.where(linelist['findex'][ind_line] == 3., True, False)
        ind_tie_f4 = np.where(linelist['findex'][ind_line] == 4., True, False)

        ind_tie_vindex1 = np.array([])
        ind_tie_vindex2 = np.array([])
        ind_tie_vindex3 = np.array([])
        ind_tie_vindex4 = np.array([])

        ind_tie_windex1 = np.array([])
        ind_tie_windex2 = np.array([])
        ind_tie_windex3 = np.array([])
        ind_tie_windex4 = np.array([])

        ind_tie_findex1 = np.array([])
        ind_tie_findex2 = np.array([])
        ind_tie_findex3 = np.array([])
        ind_tie_findex4 = np.array([])

        # get index of vindex windex in initial parameters
        #vindex1
        for iii in range(len(ind_tie_v1)):
            if ind_tie_v1[iii] == True:
                ind_tie_vindex1 = np.concatenate(
                    [ind_tie_vindex1, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        if np.any(ind_tie_v1):
            self.delta_lambda1 = (np.log(linelist['lambda'][ind_line][ind_tie_v1])-np.log(
                linelist['lambda'][ind_line][ind_tie_v1][0]))[1:]
        else:
            self.delta_lambda1 = np.array([])
        self.ind_tie_vindex1 = ind_tie_vindex1

        #vindex2
        for iii in range(len(ind_tie_v2)):
            if ind_tie_v2[iii] == True:
                ind_tie_vindex2 = np.concatenate(
                    [ind_tie_vindex2, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        if np.any(ind_tie_v2):
            self.delta_lambda2 = (np.log(linelist['lambda'][ind_line][ind_tie_v2])-np.log(
                linelist['lambda'][ind_line][ind_tie_v2][0]))[1:]
        else:
            self.delta_lambda2 = np.array([])
        self.ind_tie_vindex2 = ind_tie_vindex2

        #vindex3
        for iii in range(len(ind_tie_v3)):
            if ind_tie_v3[iii] == True:
                ind_tie_vindex3 = np.concatenate(
                    [ind_tie_vindex3, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        if np.any(ind_tie_v3):
            self.delta_lambda3 = (np.log(linelist['lambda'][ind_line][ind_tie_v3])-np.log(
                linelist['lambda'][ind_line][ind_tie_v3][0]))[1:]
        else:
            self.delta_lambda3 = np.array([])
        self.ind_tie_vindex3 = ind_tie_vindex3

        #vindex4
        for iii in range(len(ind_tie_v4)):
            if ind_tie_v4[iii] == True:
                ind_tie_vindex4 = np.concatenate(
                    [ind_tie_vindex4, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        if np.any(ind_tie_v4):
            self.delta_lambda4 = (np.log(linelist['lambda'][ind_line][ind_tie_v4])-np.log(
                linelist['lambda'][ind_line][ind_tie_v4][0]))[1:]
        else:
            self.delta_lambda4 = np.array([])
        self.ind_tie_vindex4 = ind_tie_vindex4

        #windex1
        for iii in range(len(ind_tie_w1)):
            if ind_tie_w1[iii] == True:
                ind_tie_windex1 = np.concatenate(
                    [ind_tie_windex1, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+2)])])
        self.ind_tie_windex1 = ind_tie_windex1

        #windex2
        for iii in range(len(ind_tie_w2)):
            if ind_tie_w2[iii] == True:
                ind_tie_windex2 = np.concatenate(
                    [ind_tie_windex2, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+2)])])
        self.ind_tie_windex2 = ind_tie_windex2

        #windex3
        for iii in range(len(ind_tie_w3)):
            if ind_tie_w3[iii] == True:
                ind_tie_windex3 = np.concatenate(
                    [ind_tie_windex3, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+2)])])
        self.ind_tie_windex3 = ind_tie_windex3

        #windex4
        for iii in range(len(ind_tie_w4)):
            if ind_tie_w4[iii] == True:
                ind_tie_windex4 = np.concatenate(
                    [ind_tie_windex4, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+2)])])
        self.ind_tie_windex4 = ind_tie_windex4

        # get index of findex for 1-4 case in initial parameters
        for iii_1 in range(len(ind_tie_f1)):
            if ind_tie_f1[iii_1] == True:
                ind_tie_findex1 = np.concatenate(
                    [ind_tie_findex1, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii_1])*3)])])

        for iii_2 in range(len(ind_tie_f2)):
            if ind_tie_f2[iii_2] == True:
                ind_tie_findex2 = np.concatenate(
                    [ind_tie_findex2, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii_2])*3)])])

        for iii_3 in range(len(ind_tie_f3)):
            if ind_tie_f3[iii_3] == True:
                ind_tie_findex3 = np.concatenate(
                    [ind_tie_findex3, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii_3])*3)])])

        for iii_4 in range(len(ind_tie_f4)):
            if ind_tie_f4[iii_4] == True:
                ind_tie_findex4 = np.concatenate(
                    [ind_tie_findex4, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii_4])*3)])])

        # get tied fvalue for case 1 and case 2
        if np.sum(ind_tie_f1) > 0:
            self.fvalue_factor_1 = linelist['fvalue'][ind_line][ind_tie_f1][1]/linelist['fvalue'][ind_line][ind_tie_f1][
                0]
        else:
            self.fvalue_factor_1 = np.array([])
        if np.sum(ind_tie_f2) > 0:
            self.fvalue_factor_2 = linelist['fvalue'][ind_line][ind_tie_f2][1]/linelist['fvalue'][ind_line][ind_tie_f2][
                0]
        else:
            self.fvalue_factor_2 = np.array([])
        if np.sum(ind_tie_f3) > 0:
            self.fvalue_factor_3 = linelist['fvalue'][ind_line][ind_tie_f3][1]/linelist['fvalue'][ind_line][ind_tie_f3][
                0]
        else:
            self.fvalue_factor_3 = np.array([])
        if np.sum(ind_tie_f4) > 0:
            self.fvalue_factor_4 = linelist['fvalue'][ind_line][ind_tie_f4][1]/linelist['fvalue'][ind_line][ind_tie_f4][
                0]
        else:
            self.fvalue_factor_4 = np.array([])

        self.ind_tie_findex1 = ind_tie_findex1
        self.ind_tie_findex2 = ind_tie_findex2
        self.ind_tie_findex3 = ind_tie_findex3
        self.ind_tie_findex4 = ind_tie_findex4

    # ---------MC error for emission line parameters-------------------
    def _line_mc(self, x, y, err, pp0, pp_limits, n_trails, compcenter, iif_read_line_prop, iif_dump_MC_result):
        """calculate the Monte Carlo errror of line parameters"""
        all_para_1comp = np.zeros((len(pp0), n_trails))
        all_para_std = np.zeros(len(pp0))
        all_para_std2 = np.zeros(len(pp0))
        if iif_read_line_prop == True:
            all_fwhm = np.zeros(n_trails)
            all_sigma = np.zeros(n_trails)
            all_ew = np.zeros(n_trails)
            all_peak = np.zeros(n_trails)
            all_area = np.zeros(n_trails)

        for tra in range(n_trails):
            flux = y+np.random.randn(len(y))*err
            line_fit = kmpfit.Fitter(residuals=self._residuals_line, data=(x, flux, err), maxiter=50)
            line_fit.parinfo = pp_limits
            line_fit.fit(params0=pp0)
            line_fit.params = self.newpp
            all_para_1comp[:, tra] = line_fit.params

            # further line properties
            if iif_read_line_prop == True:
                all_fwhm[tra], all_sigma[tra], all_ew[tra], all_peak[tra], all_area[tra] = self.line_prop(compcenter,
                                                                                                          line_fit.params,
                                                                                                          'broad')
                all_fwhm2 = (np.percentile(all_fwhm,84)-np.percentile(all_fwhm,16))/2
                all_peak2 = (np.percentile(all_peak,84)-np.percentile(all_peak,16))/2
                all_sigma2 = (np.percentile(all_sigma,84)-np.percentile(all_sigma,16))/2
                all_ew2 = (np.percentile(all_ew,84)-np.percentile(all_ew,16))/2
                all_area2 = (np.percentile(all_area,84)-np.percentile(all_area,16))/2

        for st in range(len(pp0)):
            all_para_std[st] = all_para_1comp[st, :].std()
            all_para_std2[st] = (np.percentile(all_para_1comp[st, :],84)-np.percentile(all_para_1comp[st, :],16))/2

        if iif_dump_MC_result == True:
            if iif_read_line_prop == True:
                return  all_para_1comp, all_para_std, all_fwhm.std(), all_sigma.std(), all_ew.std(), all_peak.std(), all_area.std(),\
                all_para_std2, all_fwhm2, all_sigma2, all_ew2, all_peak2, all_area2
            else:
                return all_para_1comp, all_para_std, all_para_std2
        elif iif_read_line_prop == True:
            return all_para_std, all_fwhm.std(), all_sigma.std(), all_ew.std(), all_peak.std(), all_area.std(),\
            all_para_std2, all_fwhm2, all_sigma2, all_ew2, all_peak2, all_area2
        else:
            return all_para_std, all_para_std2

    # -----line properties calculation function--------
    def line_prop(self, compcenter, pp, linetype):
        """
        Calculate the further results for the broad component in emission lines, e.g., FWHM, sigma, peak, line flux
        The compcenter is the theortical vacuum wavelength for the broad compoenet.
        """
        pp = pp.astype(float)
        if linetype == 'broad':
            ind_br = np.repeat(np.where(pp[2::3] > 0.0017, True, False), 3)

        elif linetype == 'narrow':
            ind_br = np.repeat(np.where(pp[2::3] <= 0.0017, True, False), 3)

        else:
            raise RuntimeError("line type should be 'broad' or 'narrow'!")

        ind_br[9:] = False  # to exclude the broad OIII and broad He II

        p = pp[ind_br]
        del pp
        pp = p

        c = 299792.458  # km/s
        n_gauss = int(len(pp)/3)
        if n_gauss == 0:
            fwhm, sigma, ew, peak, area = 0., 0., 0., 0., 0.
        else:
            cen = np.zeros(n_gauss)
            sig = np.zeros(n_gauss)

            for i in range(n_gauss):
                cen[i] = pp[3*i+1]
                sig[i] = pp[3*i+2]

            # print cen,sig,area
            left = min(cen-3*sig)
            right = max(cen+3*sig)
            disp = 1.e-4*np.log(10.)
            npix = int((right-left)/disp)

            xx = np.linspace(left, right, npix)
            yy = self.Manygauss(xx, pp)

            # here I directly use the continuum model to avoid the inf bug of EW when the spectrum range passed in is too short
            contiflux = self.conti_fit.params[6]*(np.exp(xx)/3000.0)**self.conti_fit.params[7]+self.F_poly_conti(
                np.exp(xx), self.conti_fit.params[11:])+self.Balmer_conti(np.exp(xx), self.conti_fit.params[8:11])

            # find the line peak location
            ypeak = yy.max()
            ypeak_ind = np.argmax(yy)
            peak = np.exp(xx[ypeak_ind])

            # find the FWHM in km/s
            # take the broad line we focus and ignore other broad components such as [OIII], HeII

            if n_gauss > 3:
                spline = interpolate.UnivariateSpline(xx,
                                                      self.Manygauss(xx, pp[0:9])-np.max(self.Manygauss(xx, pp[0:9]))/2,
                                                      s=0)
            else:
                spline = interpolate.UnivariateSpline(xx, yy-np.max(yy)/2, s=0)
            if len(spline.roots()) > 0:
                fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()
                fwhm = abs(np.exp(fwhm_left)-np.exp(fwhm_right))/compcenter*c

                # calculate the line sigma and EW in normal wavelength
                line_flux = self.Manygauss(xx, pp)
                line_wave = np.exp(xx)
                lambda0 = integrate.trapz(line_flux, line_wave)  # calculate the total broad line flux
                lambda1 = integrate.trapz(line_flux*line_wave, line_wave)
                lambda2 = integrate.trapz(line_flux*line_wave*line_wave, line_wave)
                ew = integrate.trapz(np.abs(line_flux/contiflux), line_wave)
                area = lambda0

                sigma = np.sqrt(lambda2/lambda0-(lambda1/lambda0)**2)/compcenter*c
            else:
                fwhm, sigma, ew, peak, area = 0., 0., 0., 0., 0.

        return fwhm, sigma, ew, peak, area

    def _DoLocalFit(self, wave, flux, err, if_plot_localfit, if_tie_localfit, save_localfit_plot_path):
        """Local fit narrow lines (PL+gauss): CaII, OII, NeV"""
        linepara_local = fits.open(self.path+'qsopar_local.fits')
        linelist_local = linepara_local[1].data
        self.linelist_local = linelist_local
        self.lcofit_stat = False

        ind_kind_line = np.where((linelist_local['lambda'] > wave.min()) & (linelist_local['lambda'] < wave.max()), True, False)
        if ind_kind_line.any() == True:
            # sort complex name with line wavelength
            uniq_linecomp, uniq_ind = np.unique(linelist_local['compname'][ind_kind_line], return_index=True)
            uniq_linecomp_sort = uniq_linecomp[linelist_local['lambda'][ind_kind_line][uniq_ind].argsort()]
            ncomp_local = len(uniq_linecomp_sort)
            compname = linelist_local['compname']
            allcompcenter = np.sort(linelist_local['lambda'][ind_kind_line][uniq_ind])

            local_result = np.array([])
            local_result_type = np.array([])
            local_result_name = np.array([])
            lco_para_result = np.array([])
            lco_para_result_type = np.array([])
            lco_para_result_name = np.array([])
            lco_para_MC_result = []
            lco_para_MC_result_name = np.array([])
            all_comp_range = np.array([])

            if if_plot_localfit == True and save_localfit_plot_path != None:
                plt.figure(figsize=(16,4))
                plot_flag = False

            for ii in range(ncomp_local):
                compcenter = allcompcenter[ii]
                ind_line = np.where(linelist_local['compname'] == uniq_linecomp_sort[ii], True, False)  # get line index
                nline_fit = np.sum(ind_line)  # n line in one complex
                linelist_fit = linelist_local[ind_line]
                # n gauss in each line
                ngauss_fit = np.asarray(linelist_fit['ngauss'], dtype=int)
                # for iitmp in range(nline_fit):   # line fit together
                comp_range = [linelist_fit[0]['minwav'], linelist_fit[0]['maxwav']]  # read complex range from table
                all_comp_range = np.concatenate([all_comp_range, comp_range])

                # ----tie lines--------
                self._do_tie_line_local(linelist_local, ind_line)

                ind_n = np.where((wave > comp_range[0]) & (wave < comp_range[1]))[0]

                if len(ind_n) > 10:
                    # call kmpfit for lines
                    if self.MC == True and self.n_trails > 0:
                        lco_fit, lco_mc_para = self._do_line_local_fit(linelist_local, self.wave[ind_n], self.flux[ind_n], self.err[ind_n], \
                                                        ind_line, nline_fit, ngauss_fit, self.MC, self.n_trails)
                        lco_para_MC_result.append(lco_mc_para)
                    else:
                        lco_fit = self._do_line_local_fit(linelist_local, self.wave[ind_n], self.flux[ind_n], self.err[ind_n], \
                                                        ind_line, nline_fit, ngauss_fit, self.MC, self.n_trails)
                    # tie lines would reduce the number of parameters increasing the dof

                    dof_fix = 0
                    if if_tie_localfit == True:
                        dof_fix += np.max((len(self.ind_tie_vindex1_lco), 1))-1
                        dof_fix += np.max((len(self.ind_tie_vindex2_lco), 1))-1

                    local_result_tmp = np.array(
                        [[linelist_local['compname'][ind_line][0]], [lco_fit.status], [lco_fit.chi2_min],
                         [lco_fit.chi2_min/(lco_fit.dof+dof_fix)], [lco_fit.niter],
                         [lco_fit.dof+dof_fix], [len(ind_n)]]).flatten()
                    local_result_type_tmp = np.array(['str', 'int', 'float', 'float', 'float', 'float', 'float'])
                    local_result_name_tmp = np.array(
                        [str(ii+1)+'_local_complex_name', str(ii+1)+'_local_line_status', str(ii+1)+'_local_line_min_chi2',
                         str(ii+1)+'_local_line_red_chi2', str(ii+1)+'_local_niter', str(ii+1)+'_local_ndof', str(ii+1)+'_local_npix'])
                    local_result = np.concatenate([local_result, local_result_tmp])
                    local_result_name = np.concatenate([local_result_name, local_result_name_tmp])
                    local_result_type = np.concatenate([local_result_type, local_result_type_tmp])

                    lco_para_tmp = np.array([])
                    lco_para_type_tmp = np.array(['float', 'float'])
                    lco_para_name_tmp = np.array([linelist_local['compname'][ind_line][0]+'_PL_norm', linelist_local['compname'][ind_line][0]+'_PL_slope'])

                    for gg in range(len(lco_fit.params)):
                        lco_para_tmp = np.concatenate([lco_para_tmp, np.array([lco_fit.params[gg]])])
                    lco_para_result = np.concatenate([lco_para_result, lco_para_tmp])

                    # lco_para result name -----------------
                    for n in range(nline_fit):
                        for nn in range(int(ngauss_fit[n])):
                            line_name = linelist_local['linename'][ind_line][n]+'_'+str(nn+1)
                            lco_para_type_tmp = np.append(lco_para_type_tmp, ['float', 'float', 'float'])
                            lco_para_name_tmp = np.append(lco_para_name_tmp, [line_name+'_scale', line_name+'_centerwave', line_name+'_sigma'])
                    lco_para_result_type = np.concatenate([lco_para_result_type, lco_para_type_tmp])
                    lco_para_result_name = np.concatenate([lco_para_result_name, lco_para_name_tmp])

                    # save localfit plot
                    if if_plot_localfit == True and save_localfit_plot_path != None:
                        #print('plot', [linelist_local['compname'][ind_line][0]])
                        ax = plt.subplot(1, ncomp_local, ii+1)
                        med_local_flux = medfilt(self.flux[ind_n], kernel_size=5)
                        ax.set_title(linelist_local['compname'][ind_line][0], fontsize=16)
                        ax.errorbar(self.wave[ind_n], self.flux[ind_n], yerr = self.err[ind_n], \
                                    c='k', ecolor='grey', zorder=1)
                        ax.plot(self.wave[ind_n], lco_fit.params[0]*(self.wave[ind_n]/3000)**lco_fit.params[1], \
                                c='orange', zorder=4)
                        for ng in range(len(lco_fit.params[2::3])):
                            ax.plot(self.wave[ind_n], lco_fit.params[0]*(self.wave[ind_n]/3000)**lco_fit.params[1] + self.Onegauss(np.log(self.wave[ind_n]), lco_fit.params[2+ng*3:5+ng*3]), \
                                    c='seagreen', zorder=2)
                        ax.plot(self.wave[ind_n], lco_fit.params[0]*(self.wave[ind_n]/3000)**lco_fit.params[1] + self.Manygauss(np.log(self.wave[ind_n]), lco_fit.params[2:]), \
                                c='r', zorder=3)
                        ax.set_xlim(comp_range[0], comp_range[1])
                        #ax.set_ylim(med_local_flux.min()-0.5, 1.25*med_local_flux.max())
                        ax.xaxis.set_ticks_position('both')
                        ax.yaxis.set_ticks_position('both')
                        ax.tick_params(axis='x',which='both',direction='in')
                        ax.tick_params(axis='y',which='both',direction='in')
                        ax.set_xlabel(r'wavelength ($\rm\AA$)')
                        ax.set_ylabel(r'Flux ($10^{-17}\rm erg\,s^{-1}\,cm^{-2}\,\AA^{-1}$)')
                        plot_flag = True

            if plot_flag:
                plt.savefig(save_localfit_plot_path+'localfit_'+self.sdss_name+'.pdf')
            plt.close()

            lco_line_result = np.concatenate([local_result, lco_para_result])
            lco_line_result_type = np.concatenate([local_result_type, lco_para_result_type])
            lco_line_result_name = np.concatenate([local_result_name, lco_para_result_name])

            self.lco_para_MC_result = lco_para_MC_result
            self.lco_line_result = lco_line_result
            self.lco_line_result_type = lco_line_result_type
            self.lco_line_result_name = lco_line_result_name
            self.lco_para_result_name = lco_para_result_name
            self.lcofit_stat = True

    def _do_tie_line_local(self, linelist, ind_line):
        """Tie line's central"""
        # --------------- tie parameter-----------
        # so far, only two groups of each properties are support for tying
        ind_tie_v1 = np.where(linelist['vindex'][ind_line] == 1., True, False)
        ind_tie_v2 = np.where(linelist['vindex'][ind_line] == 2., True, False)

        ind_tie_vindex1 = np.array([])
        ind_tie_vindex2 = np.array([])

        # get index of vindex windex in initial parameters
        for iii in range(len(ind_tie_v1)):
            if ind_tie_v1[iii] == True:
                ind_tie_vindex1 = np.concatenate(
                    [ind_tie_vindex1, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        if np.any(ind_tie_v1):
            self.delta_lambda1_lco = (np.log(linelist['lambda'][ind_line][ind_tie_v1])-np.log(
                linelist['lambda'][ind_line][ind_tie_v1][0]))[1:]
        else:
            self.delta_lambda1_lco = np.array([])
        self.ind_tie_vindex1_lco = ind_tie_vindex1

        for iii in range(len(ind_tie_v2)):
            if ind_tie_v2[iii] == True:
                ind_tie_vindex2 = np.concatenate(
                    [ind_tie_vindex2, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        if np.any(ind_tie_v2):
            self.delta_lambda2_lco = (np.log(linelist['lambda'][ind_line][ind_tie_v2])-np.log(
                linelist['lambda'][ind_line][ind_tie_v2][0]))[1:]
        else:
            self.delta_lambda2_lco = np.array([])
        self.ind_tie_vindex2_lco = ind_tie_vindex2

    def local_residuals(self, pp, data):
        xval, yval, weight = data

        if self.if_tie_localfit:
            if len(self.ind_tie_vindex1_lco) > 1:
                for xx in range(len(self.ind_tie_vindex1_lco)-1):
                    pp[int(self.ind_tie_vindex1_lco[xx+1])+2] = pp[int(self.ind_tie_vindex1_lco[0])+2]+self.delta_lambda1_lco[xx]

            if len(self.ind_tie_vindex2_lco) > 1:
                for xx in range(len(self.ind_tie_vindex2_lco)-1):
                    pp[int(self.ind_tie_vindex2_lco[xx+1])+2] = pp[int(self.ind_tie_vindex2_lco[0])+2]+self.delta_lambda2_lco[xx]

        self.newpp1 = pp.copy()
        return (yval - pp[0]*(np.exp(xval)/3000)**pp[1] - self.Manygauss(xval, pp[2:]))/weight

    def _do_line_local_fit(self, linelist_local, lam, flux, err, ind_line, nline_fit, ngauss_fit, if_MC, n_trails):
        """The key function to do the line fit with kmpfit"""
        lco_fit = kmpfit.Fitter(self.local_residuals, data=(np.log(lam), flux, err))  # fitting wavelength in ln space
        lco_fit_ini = np.array([1., -1.5])
        lco_fit_par = np.array([{'limits': (0., 10.**10)}, {'limits': (-5., 3.)}])
        for n in range(nline_fit):
            for nn in range(ngauss_fit[n]):
                # set up initial parameter guess
                lco_fit_ini = np.concatenate([lco_fit_ini, \
                                              [0., np.log(linelist_local['lambda'][ind_line][n]), linelist_local['inisig'][ind_line][n]]])
                # set up parameter limits
                lambda_low = np.log(linelist_local['lambda'][ind_line][n])-linelist_local['voff'][ind_line][n]
                lambda_up = np.log(linelist_local['lambda'][ind_line][n])+linelist_local['voff'][ind_line][n]
                sig_low = linelist_local['minsig'][ind_line][n]
                sig_up = linelist_local['maxsig'][ind_line][n]
                if linelist_local['compname'][ind_line][n] == 'CaII':
                    lco_fit_par0 = [{'limits': (-10.**10, 0.)}, {'limits': (lambda_low, lambda_up)},
                                     {'limits': (sig_low, sig_up)}]
                else:
                    lco_fit_par0 = [{'limits': (0., 10.**10)}, {'limits': (lambda_low, lambda_up)},
                                     {'limits': (sig_low, sig_up)}]
                lco_fit_par = np.concatenate([lco_fit_par, lco_fit_par0])

        lco_fit.parinfo = lco_fit_par
        lco_fit.fit(params0=lco_fit_ini)
        lco_fit.params = self.newpp1
        self.lco_fit = lco_fit
        self.lco_fit_ini = lco_fit_ini
        self.lco_fit_par = lco_fit_par

        if if_MC:
            lco_mc_para = np.zeros((n_trails, len(lco_fit_ini)))
            for nn in range(n_trails):
                lco_fit_mc = kmpfit.Fitter(self.local_residuals, data = (np.log(lam), flux+np.random.randn(len(flux))*err, err))
                lco_fit_mc.parinfo = lco_fit_par
                lco_fit_mc.fit(params0=lco_fit_ini)
                lco_fit_mc.params = self.newpp1
                lco_mc_para[nn, :] = lco_fit_mc.params
            wv_sigma = self.MC_1sigma_err(lco_mc_para[:,3])
            flag = 0
            if abs(np.median(lco_mc_para[:,3])-lco_fit.params[3]) > 3 * wv_sigma:
                lco_fit_new = kmpfit.Fitter(self.local_residuals, data=(np.log(lam), flux, err))  # fitting wavelength in ln space
                lco_fit_ini = np.array([1., -1.5])
                lco_fit_par = np.array([{'limits': (0., 10.**10)}, {'limits': (-5., 3.)}])
                for n in range(nline_fit):
                    for nn in range(ngauss_fit[n]):
                        # set up initial parameter guess
                        lco_fit_ini = np.concatenate([lco_fit_ini, [0., np.log(linelist_local['lambda'][ind_line][n]), linelist_local['inisig'][ind_line][n]]])
                        # set up parameter limits
                        lambda_low = np.log(linelist_local['lambda'][ind_line][n])-3*wv_sigma
                        lambda_up = np.log(linelist_local['lambda'][ind_line][n])+3*wv_sigma
                        if np.exp(lambda_up)-np.exp(lambda_low) < 1.:
                            flag = 1
                        sig_low = linelist_local['minsig'][ind_line][n]
                        sig_up = linelist_local['maxsig'][ind_line][n]
                        if linelist_local['compname'][ind_line][n] == 'CaII':
                            lco_fit_par0 = [{'limits': (-10.**10, 0.)}, {'limits': (lambda_low, lambda_up)},
                                             {'limits': (sig_low, sig_up)}]
                        else:
                            lco_fit_par0 = [{'limits': (0., 10.**10)}, {'limits': (lambda_low, lambda_up)},
                                             {'limits': (sig_low, sig_up)}]
                        lco_fit_par = np.concatenate([lco_fit_par, lco_fit_par0])

                if flag == 0:
                    lco_fit_new.parinfo = lco_fit_par
                    lco_fit_new.fit(params0=lco_fit_ini)
                    lco_fit_new.params = self.newpp1
                    self.lco_fit = lco_fit_new
                    self.lco_fit_ini = lco_fit_ini
                    self.lco_fit_par = lco_fit_par
                    lco_fit = lco_fit_new
            return lco_fit, lco_mc_para
        else:
            return lco_fit

    def MC_1sigma_err(self, ip_arr):
        return (np.percentile(ip_arr, 84)-np.percentile(ip_arr, 16))/2

    def _residuals_line(self, pp, data):
        "The line residual function used in kmpfit"
        xval, yval, weight = data

        # ------tie parameter------------
        if self.tie_lambda == True:
            if len(self.ind_tie_vindex1) > 1:
                for xx in range(len(self.ind_tie_vindex1)-1):
                    pp[int(self.ind_tie_vindex1[xx+1])] = pp[int(self.ind_tie_vindex1[0])]+self.delta_lambda1[xx]

            if len(self.ind_tie_vindex2) > 1:
                for xx in range(len(self.ind_tie_vindex2)-1):
                    pp[int(self.ind_tie_vindex2[xx+1])] = pp[int(self.ind_tie_vindex2[0])]+self.delta_lambda2[xx]

            if len(self.ind_tie_vindex3) > 1:
                for xx in range(len(self.ind_tie_vindex3)-1):
                    pp[int(self.ind_tie_vindex3[xx+1])] = pp[int(self.ind_tie_vindex3[0])]+self.delta_lambda3[xx]

            if len(self.ind_tie_vindex4) > 1:
                for xx in range(len(self.ind_tie_vindex4)-1):
                    pp[int(self.ind_tie_vindex4[xx+1])] = pp[int(self.ind_tie_vindex4[0])]+self.delta_lambda4[xx]

        if self.tie_width == True:
            if len(self.ind_tie_windex1) > 1:
                for xx in range(len(self.ind_tie_windex1)-1):
                    pp[int(self.ind_tie_windex1[xx+1])] = pp[int(self.ind_tie_windex1[0])]

            if len(self.ind_tie_windex2) > 1:
                for xx in range(len(self.ind_tie_windex2)-1):
                    pp[int(self.ind_tie_windex2[xx+1])] = pp[int(self.ind_tie_windex2[0])]

            if len(self.ind_tie_windex3) > 1:
                for xx in range(len(self.ind_tie_windex3)-1):
                    pp[int(self.ind_tie_windex3[xx+1])] = pp[int(self.ind_tie_windex3[0])]

            if len(self.ind_tie_windex4) > 1:
                for xx in range(len(self.ind_tie_windex4)-1):
                    pp[int(self.ind_tie_windex4[xx+1])] = pp[int(self.ind_tie_windex4[0])]

        if len(self.ind_tie_findex1) > 0 and self.tie_flux_1 == True:
            pp[int(self.ind_tie_findex1[1])] = pp[int(self.ind_tie_findex1[0])]*self.fvalue_factor_1
        if len(self.ind_tie_findex2) > 0 and self.tie_flux_2 == True:
            pp[int(self.ind_tie_findex2[1])] = pp[int(self.ind_tie_findex2[0])]*self.fvalue_factor_2
        if len(self.ind_tie_findex3) > 0 and self.tie_flux_3 == True:
            pp[int(self.ind_tie_findex3[1])] = pp[int(self.ind_tie_findex3[0])]*self.fvalue_factor_3
        if len(self.ind_tie_findex4) > 0 and self.tie_flux_4 == True:
            pp[int(self.ind_tie_findex4[1])] = pp[int(self.ind_tie_findex4[0])]*self.fvalue_factor_4
        # ---------------------------------

        # restore parameters
        self.newpp = pp.copy()
        return (yval-self.Manygauss(xval, pp))/weight

    def _SaveResult(self, conti_result, conti_result_type, conti_result_name, line_result, line_result_type, line_result_name, save_fits_path, save_fits_name, iif_dump_MC_result, iif_save_spec, iif_localfit):
        """Save all data to fits"""
        self.all_result = np.concatenate([conti_result, line_result])
        self.all_result_type = np.concatenate([conti_result_type, line_result_type])
        self.all_result_name = np.concatenate([conti_result_name, line_result_name])

        if iif_localfit == True and self.lcofit_stat == True:
            self.all_result = np.append(self.all_result, self.lco_line_result)
            self.all_result_type = np.append(self.all_result_type, self.lco_line_result_type)
            self.all_result_name = np.append(self.all_result_name, self.lco_line_result_name)

        if iif_dump_MC_result == True and self.MC == True:
            primary_hdu = fits.PrimaryHDU()
            basic_info_hdu = []
            for i in range(len(self.all_result)):
                if self.all_result_type[i] == 'float':
                    basic_info_hdu.append(fits.Column(name=str(self.all_result_name[i]), format='D', array=[self.all_result[i]]))
                elif self.all_result_type[i] == 'int':
                    basic_info_hdu.append(fits.Column(name=str(self.all_result_name[i]), format='K', array=[self.all_result[i]]))
                elif self.all_result_name[i] == 'ObjID':
                    basic_info_hdu.append(fits.Column(name=str(self.all_result_name[i]), format='18A', array=[self.all_result[i]]))
                else:
                    basic_info_hdu.append(fits.Column(name=str(self.all_result_name[i]), format='8A', array=[self.all_result[i]]))
            basic_info_hdu = fits.BinTableHDU.from_columns(basic_info_hdu)
            op = fits.HDUList([primary_hdu, basic_info_hdu])
            MC_result_hdu = []
            for i in range(len(self.conti_result_MC)):
                for j in range(len(self.conti_result_MC_name[i])):
                    MC_result_hdu.append(fits.Column(name=self.conti_result_MC_name[i][j], format='D', array=np.array(self.conti_result_MC[i][j])))
            if len(self.gauss_MCresult) > 0:
                for i in range(len(self.gauss_MCresult)):
                    for j in range(len(self.gauss_MCresult_name[i])):
                        for k in range(3):
                            MC_result_hdu.append(fits.Column(name=self.gauss_MCresult_name[i][j][k], format='D', array=self.gauss_MCresult[i][0][j*3+k]))
            if iif_localfit == True and self.lcofit_stat == True:
                ij = 0
                for i in range(len(self.lco_para_MC_result)):
                    for j in range(len(self.lco_para_MC_result[i].T)):
                        MC_result_hdu.append(fits.Column(name=self.lco_para_result_name[ij]+'_MC', format='D', array=self.lco_para_MC_result[i].T[j]))
                        ij += 1
            MC_result_hdu = fits.BinTableHDU.from_columns(MC_result_hdu)
            op.append(MC_result_hdu)
            if iif_save_spec:
                spec_hdu = []
                spec_hdu.append(fits.Column(name='wave_prereduced', format='D', array=self.wave_prereduced))
                spec_hdu.append(fits.Column(name='flux_prereduced', format='D', array=self.flux_prereduced))
                spec_hdu.append(fits.Column(name='err_prereduced', format='D', array=self.err_prereduced))
                spec_hdu.append(fits.Column(name='wave_conti', format='D', array=self.wave_conti))
                spec_hdu.append(fits.Column(name='flux_conti', format='D', array=self.flux_conti))
                spec_hdu.append(fits.Column(name='err_conti', format='D', array=self.err_conti))
                ind_sort = np.argsort(self.linespec_wave)
                spec_hdu.append(fits.Column(name='wave_line', format='D', array=self.linespec_wave[ind_sort]))
                spec_hdu.append(fits.Column(name='flux_line', format='D', array=self.linespec_lineflux[ind_sort]))
                spec_hdu.append(fits.Column(name='err_line', format='D', array=self.linespec_err[ind_sort]))
                if self.rej_line_abs:
                    spec_hdu.append(fits.Column(name='wave_line_abs', format='D', array=self.rej_line_abs_wave))
                    spec_hdu.append(fits.Column(name='flux_line_abs', format='D', array=self.rej_line_abs_flux))
                    spec_hdu.append(fits.Column(name='err_line_abs', format='D', array=self.rej_line_abs_err))
                #print('save spec')
                spec_hdu = fits.BinTableHDU.from_columns(spec_hdu)
                op.append(spec_hdu)
            op.writeto(save_fits_path+save_fits_name+'.fits', overwrite=True)

        else:
            t = Table(self.all_result, names=(self.all_result_name), dtype=self.all_result_type)
            t.write(save_fits_path+save_fits_name+'.fits', format='fits', overwrite=True)

    def continuum_PL_poly(self, wave_val, conti_ip_val):
        return conti_ip_val[6]*(wave_val/3000.0)**conti_ip_val[7]\
                + self.F_poly_conti(wave_val, conti_ip_val[11:])

    def _PlotFig(self, ra, dec, z, wave, flux, err, decomposition_host, linefit, tmp_all, gauss_result, f_conti_model, conti_fit, all_comp_range, uniq_linecomp_sort, line_flux, save_fig_path, target_info):
        """Plot the results"""

        conti_pl = self.continuum_PL_poly(wave, conti_fit.params)
        med_flux = medfilt(flux, kernel_size=5)

        if linefit == True:
            fig, axn = plt.subplots(nrows=2, ncols=np.max([self.ncomp, 1]), figsize=(15,8))
            ax1 = plt.subplot(2,1,1)

            if self.MC == True: mc_flag = 3
            else: mc_flag = 1

            temp_gauss_result = gauss_result
            br_emis_line = np.zeros(len(wave))
            na_emis_line = np.zeros(len(wave))

            for p in range(int(len(temp_gauss_result)/mc_flag/3)):
                # warn that the width used to separate narrow from broad is not exact 1200 km s-1 which would lead to wrong judgement
                line_single = self.Onegauss(np.log(wave), temp_gauss_result[p*3*mc_flag:(p+1)*3*mc_flag:mc_flag])
                if 2*np.sqrt(2*np.log(2))*(np.exp(temp_gauss_result[(2+p*3)*mc_flag]))*3.e5 < 1200.:
                    na_emis_line = na_emis_line + line_single
                    color='b'
                else:
                    br_emis_line = br_emis_line + line_single
                    color='seagreen'
                if self.ncomp > 1:
                    for c in range(self.ncomp):
                        axn[1][c].plot(wave, line_single, color=color, zorder=3)
                        axn[1][c].plot(self.rej_line_abs_wave, self.rej_line_abs_flux, ls='', \
                                        marker='x',mec = 'cornflowerblue', ms=4, zorder=8)
                else:
                    axn[1].plot(wave, line_single, color=color, zorder=3)
                    axn[1].plot(self.rej_line_abs_wave, self.rej_line_abs_flux, ls='', \
                                    marker='x',mec = 'cornflowerblue', ms=4, zorder=8)

            ax1.plot(wave, br_emis_line + f_conti_model, c='seagreen', zorder=2, label='br lines')
            ax1.plot(wave, na_emis_line + f_conti_model, c='b', zorder=2,label='na lines')
            ax1.plot(wave, na_emis_line + br_emis_line + f_conti_model, c='r', zorder=6, label='fit')

            if self.ncomp > 1:
                for l in range(self.ncomp):
                    ind = np.where((wave<all_comp_range[2*l+1])&(wave>all_comp_range[2*l]))
                    axn[1][l].errorbar(wave[ind], self.line_flux[ind], yerr=err[ind], \
                                        c='k', ecolor='silver', zorder=1)
                    axn[1][l].plot(wave[ind], na_emis_line[ind] + br_emis_line[ind], c='r', zorder=4)

                    axn[1][l].text(0.02, 0.9, uniq_linecomp_sort[l], fontsize=14, transform=axn[1][l].transAxes)
                    axn[1][l].text(0.02, 0.80, r'$\chi ^2_r=$'+str(np.round(float(self.comp_result[l*7+3]), 2)),
                                   fontsize=12, transform=axn[1][l].transAxes)
                    axn[1][l].set_xlim(all_comp_range[2*l], all_comp_range[2*l+1])
                    med_lineflux = medfilt(self.line_flux[ind], kernel_size=3)
                    axn[1][l].set_ylim(1.25*np.min(self.line_flux[ind]), 1.25*np.max(med_lineflux))
                    axn[1][l].xaxis.set_ticks_position('both')
                    axn[1][l].yaxis.set_ticks_position('both')
                    if all_comp_range[2*l+1]- all_comp_range[2*l]<201:
                        axn[1][l].xaxis.set_major_locator(MultipleLocator(50.))
                        axn[1][l].xaxis.set_minor_locator(MultipleLocator(10.))
                    else:
                        axn[1][l].xaxis.set_major_locator(MultipleLocator(100.))
                        axn[1][l].xaxis.set_minor_locator(MultipleLocator(20.))
                    axn[1][l].tick_params(axis='x',which='both',direction='in')
                    axn[1][l].tick_params(axis='y',which='both',direction='in')
            else:
                ind = np.where((wave<all_comp_range[1])&(wave>all_comp_range[0]))
                axn[1].errorbar(wave[ind], self.line_flux[ind], yerr=err[ind], \
                                    c='k', ecolor='silver', zorder=1)
                axn[1].plot(wave[ind], na_emis_line[ind] + br_emis_line[ind], c='r', zorder=4)

                axn[1].text(0.02, 0.9, uniq_linecomp_sort[0], fontsize=14, transform=axn[1].transAxes)
                axn[1].text(0.02, 0.80, r'$\chi ^2_r=$'+str(np.round(float(self.comp_result[3]), 2)),
                               fontsize=12, transform=axn[1].transAxes)
                axn[1].set_xlim(all_comp_range[0], all_comp_range[1])
                med_lineflux = medfilt(self.line_flux[ind], kernel_size=3)
                axn[1].set_ylim(1.25*np.min(self.line_flux[ind]), 1.25*np.max(med_lineflux))
                axn[1].xaxis.set_ticks_position('both')
                axn[1].yaxis.set_ticks_position('both')
                if all_comp_range[1]- all_comp_range[0]<201:
                    axn[1].xaxis.set_major_locator(MultipleLocator(50.))
                    axn[1].xaxis.set_minor_locator(MultipleLocator(10.))
                else:
                    axn[1].xaxis.set_major_locator(MultipleLocator(100.))
                    axn[1].xaxis.set_minor_locator(MultipleLocator(20.))
                axn[1].tick_params(axis='x',which='both',direction='in')
                axn[1].tick_params(axis='y',which='both',direction='in')

        else:
            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                   figsize=(12, 4))  # if no lines are fitted, there would be only one row


        ax1.errorbar(wave, flux, yerr=err, c='k', ecolor='silver', label='data', zorder=2)
        ax1.plot(wave, f_conti_model, c='orange', label='continuum', zorder=5)
        ax1.plot(wave, f_conti_model-conti_pl, c='skyblue', label='FeII', zorder=3)
        ax1.plot(wave, conti_pl, c='orange', ls='--', zorder=4)

        if decomposition_host == True and self.decomposed == True:
            ax1.plot(wave, self.qso+self.host, 'pink', label='host+qso temp', zorder=3)
            ax1.plot(wave, flux, 'grey', label='data-host', zorder=1)
            ax1.plot(wave, self.host, 'purple', label='host', zorder=4)
        else:
            host = self.flux_prereduced.min()

        ax1.scatter(wave[tmp_all], np.repeat(1.23*np.max(med_flux), len(wave[tmp_all])), color='grey',
                    marker='s', s=5, zorder=1)  # plot continuum region

        if self.BC == True:
            ax.plot(wave, self.f_pl_model+self.f_poly_model+self.f_bc_model, 'y', lw=2, label='BC', zorder=8)

        if self.decomposed == False:
            plot_bottom = flux.min()
        else:
            plot_bottom = min(self.host.min(), flux.min())

        ax1.axhline(0, c='grey', ls='--', zorder=1)
        ax1.set_xlim(wave.min(), wave.max())
        ax1.set_ylim(-1.5*abs(med_flux.min()), 1.25*med_flux.max())
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_major_locator(MultipleLocator(250.))
        ax1.xaxis.set_minor_locator(MultipleLocator(50.))
        #ax1.yaxis.set_minor_locator(MultipleLocator(5.))

        if self.plot_legend == True:
            ax1.legend(loc='lower left', frameon=False, ncol=2,  edgecolor='None', fontsize=10)

        # plot line name--------
        if self.plot_line_name == True:
            line_cen = np.array([6564.61,  6732.66, 4862.68, 5008.24, 4687.02, 4341.68, 3934.78, 3728.47, \
                                 3426.84, 2798.75, 1908.72, 1816.97, 1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1335.30, \
                                 1215.67])
            line_name = np.array(['Ha+[NII]','[SII]6718,6732', 'Hb', '[OIII]', 'HeII4687', 'Hr', 'CaII3934', '[OII]3728', \
                                  'NeV3426', 'MgII', 'CIII]', 'SiII1816', 'NIII]1750', 'NIV]1718', 'CIV', 'HeII1640', '', 'SiIV+OIV', \
                                  'CII1335', 'Lya'])
            for ll in range(len(line_cen)):
                if wave.min() < line_cen[ll] < wave.max():
                    ax1.axvline(line_cen[ll], c='grey', ls='--', zorder=1)
                    ax1.text(line_cen[ll]+7, med_flux.max()*1.2, line_name[ll], rotation=90, fontsize=10, va='top', zorder=5)

        if self.ra == -999. or self.dec == -999.:
            ax1.set_title(str(self.sdss_name)+'   z = '+str(np.round(z, 4)), fontsize=16)
        elif target_info is not None:
            ax1.set_title(str(self.sdss_name)+'   z = '+str(np.round(z, 4))+'   '+str(target_info), fontsize=16)
        else:
            ax1.set_title('ra,dec = ('+str(ra)+','+str(dec)+')   '+str(self.sdss_name)+'   z = '+str(np.round(z, 4)),
                         fontsize=16)
        ax1.tick_params(axis='x',which='both',direction='in')
        ax1.tick_params(axis='y',which='both',direction='in')
        ax1.set_xlabel(r'Wavelength ($\rm \AA$)', fontsize=16)

        if linefit == True:
            ax1.text(0.5, -1.4, r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=16, transform=ax1.transAxes,
                    ha='center')
            ax1.text(-0.1, -0.01, r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=16,
                    transform=ax1.transAxes, rotation=90, ha='center', rotation_mode='anchor')
        else:
            plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=16)
            plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=16)

        if self.save_fig == True:
            plt.savefig(save_fig_path+self.sdss_name+'.pdf', dpi=60)

        if linefit == True:
            fig.clf()
            plt.close(fig)
        else:
            plt.clf()
            plt.close()

    def CalFWHM(self, logsigma):
        """transfer the logFWHM to normal frame"""
        return 2*np.sqrt(2*np.log(2))*(np.exp(logsigma)-1)*300000.

    def Smooth(self, y, box_pts):
        "Smooth the flux with n pixels"
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def Fe_flux_mgii(self, xval, pp):
        "Fit the UV Fe compoent on the continuum from 1200 to 3500 A based on the Boroson & Green 1992."
        yval = np.zeros_like(xval)
        wave_Fe_mgii = 10**fe_uv[:, 0]
        flux_Fe_mgii = fe_uv[:, 1]*10**15
        Fe_FWHM = pp[1]
        xval_new = xval*(1.0+pp[2])

        ind = np.where((xval_new > 1200.) & (xval_new < 3500.), True, False)
        if np.sum(ind) > 100:
            if Fe_FWHM < 900.0:
                sig_conv = np.sqrt(910.0**2-900.0**2)/2./np.sqrt(2.*np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM**2-900.0**2)/2./np.sqrt(2.*np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv/106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
            khalfsz = np.round(4*sig_pix+1, 0)
            xx = np.arange(0, khalfsz*2, 1)-khalfsz
            kernel = np.exp(-xx**2/(2*sig_pix**2))
            kernel = kernel/np.sum(kernel)

            flux_Fe_conv = np.convolve(flux_Fe_mgii, kernel, 'same')
            tck = interpolate.splrep(wave_Fe_mgii, flux_Fe_conv)
            yval[ind] = pp[0]*interpolate.splev(xval_new[ind], tck)
        return yval

    def Fe_flux_balmer(self, xval, pp):
        "Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"
        yval = np.zeros_like(xval)

        wave_Fe_balmer = 10**fe_op[:, 0]
        flux_Fe_balmer = fe_op[:, 1]*10**15
        ind = np.where((wave_Fe_balmer > 3686.) & (wave_Fe_balmer < 7484.), True, False)
        wave_Fe_balmer = wave_Fe_balmer[ind]
        flux_Fe_balmer = flux_Fe_balmer[ind]
        Fe_FWHM = pp[1]
        xval_new = xval*(1.0+pp[2])
        ind = np.where((xval_new > 3686.) & (xval_new < 7484.), True, False)
        if np.sum(ind) > 100:
            if Fe_FWHM < 900.0:
                sig_conv = np.sqrt(910.0**2-900.0**2)/2./np.sqrt(2.*np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM**2-900.0**2)/2./np.sqrt(2.*np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv/106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
            khalfsz = np.round(4*sig_pix+1, 0)
            xx = np.arange(0, khalfsz*2, 1)-khalfsz
            kernel = np.exp(-xx**2/(2*sig_pix**2))
            kernel = kernel/np.sum(kernel)
            flux_Fe_conv = np.convolve(flux_Fe_balmer, kernel, 'same')
            tck = interpolate.splrep(wave_Fe_balmer, flux_Fe_conv)
            yval[ind] = pp[0]*interpolate.splev(xval_new[ind], tck)
        return yval

    def Balmer_conti(self, xval, pp):
        """Fit the Balmer continuum from the model of Dietrich+02"""
        # xval = input wavelength, in units of A
        # pp=[norm, Te, tau_BE] -- in units of [--, K, --]

        lambda_BE = 3646.  # A
        bbflux = blackbody_lambda(xval, pp[1]).value*3.14  # in units of ergs/cm2/s/A
        tau = pp[2]*(xval/lambda_BE)**3
        result = pp[0]*bbflux*(1.-np.exp(-tau))
        ind = np.where(xval > lambda_BE, True, False)
        if ind.any() == True:
            result[ind] = 0.
        return result

    def F_poly_conti(self, xval, pp):
        """Fit the continuum with a polynomial component account for the dust reddening with a*X+b*X^2+c*X^3"""
        xval2 = xval-3000.
        yval = 0.*xval2
        for i in range(len(pp)):
            yval = yval+pp[i]*xval2**(i+1)
        return yval

    def Flux2L(self, flux, z):
        """Transfer flux to luminoity assuming a flat Universe"""
        DL = cosmo.luminosity_distance(z).value*10**6*3.086*10**18  # unit cm
        L = flux*1.e-17*4.*np.pi*DL**2  # erg/s/A
        return L

    def Onegauss(self, xval, pp):
        """The single Gaussian model used to fit the emission lines
        Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave
        """

        term1 = np.exp(- (xval-pp[1])**2/(2.*pp[2]**2))
        yval = pp[0]*term1/(np.sqrt(2.*np.pi)*pp[2])
        return yval

    def Manygauss(self, xval, pp):
        """The multi-Gaussian model used to fit the emission lines, it will call the onegauss function"""
        ngauss = int(pp.shape[0]/3)
        if ngauss != 0:
            yval = 0.
            for i in range(ngauss):
                yval = yval+self.Onegauss(xval, pp[i*3:(i+1)*3])
            return yval
        else:
            return np.zeros_like(xval)

    def Get_Fe_flux(self, ranges, pp=None):
        """Calculate the flux of fitted FeII template within given wavelength ranges.
        ranges: 1-D array, 2-D array
            if 1-D array was given, it should contain two parameters contain a range of wavelength. FeII flux within this range would be calculate and documented in the result fits file.
            if 2-D array was given, it should contain a series of ranges. FeII flux within these ranges would be documented respectively.
        pp: 1-D array with 3 or 6 items.
            If 3 parameters were given, function will choose a proper template (MgII or balmer) according to the range.
            If the range give excess either template, an error would be arose.
            If 6 parameters were given (recommended), function would adopt the first three for the MgII template and the last three for the balmer."""
        if pp is None:
            pp = self.conti_fit.params[:6]

        Fe_flux_result = np.array([])
        Fe_flux_type = np.array([])
        Fe_flux_name = np.array([])
        if np.array(ranges).ndim == 1:
            Fe_flux_result = np.append(Fe_flux_result, self._calculate_Fe_flux(ranges, pp))
            Fe_flux_name = np.append(Fe_flux_name, 'Fe_flux_'+str(int(np.min(ranges)))+'_'+str(int(np.max(ranges))))
            Fe_flux_type = np.append(Fe_flux_type, 'float')

        elif np.array(ranges).ndim == 2:
            for iii in range(np.array(self.Fe_flux_range).shape[0]):
                Fe_flux_result = np.append(Fe_flux_result, self._calculate_Fe_flux(ranges[iii], pp))
                Fe_flux_name = np.append(Fe_flux_name,
                                         'Fe_flux_'+str(int(np.min(ranges[iii])))+'_'+str(int(np.max(ranges[iii]))))
                Fe_flux_type = np.append(Fe_flux_type, 'float')
        else:
            raise IndexError('The parameter ranges only adopts arrays with 1 or 2 dimensions.')

        return Fe_flux_result, Fe_flux_type, Fe_flux_name

    def _calculate_Fe_flux(self, range, pp):
        """Calculate the flux of fitted FeII template within one given wavelength range.
        The pp could be an array with a length of 3 or 6. If 3 parameters were give, function will choose a
        proper template (MgII or balmer) according to the range. If the range give excess both template, an
        error would be arose. If 6 parameters were give, function would adopt the first three for the MgII
        template and the last three for the balmer."""

        balmer_range = np.array([3686., 7484.])
        mgii_range = np.array([1200., 3500.])
        upper = np.min([np.max(range), np.max(self.wave)])
        lower = np.max([np.min(range), np.min(self.wave)])
        if upper < np.max(range) or lower > np.min(range):
            print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) exceeded '
                  'the boundary of spectrum wavelength range. The excess part would be set to zero!')
        disp = 1.e-4*np.log(10.)
        xval = np.exp(np.arange(np.log(lower), np.log(upper), disp))
        if len(xval) < 10:
            print('Warning: Available part in range '+str(range)+' is less than 10 pixel. Flux = -999 would be given!')
            return -999

        if len(pp) == 3:
            if upper <= mgii_range[1] and lower >= mgii_range[0]:
                yval = self.Fe_flux_mgii(xval, pp)
            elif upper <= balmer_range[1] and lower >= balmer_range[0]:
                yval = self.Fe_flux_balmer(xval, pp)
            else:
                raise OverflowError('Only 3 parameters are given in this function. \
                Make sure the range is within [1200., 3500.] or [3686., 7484.]!')

        elif len(pp) == 6:
            yval = self.Fe_flux_mgii(xval, pp[:3])+self.Fe_flux_balmer(xval, pp[3:])
            if upper > balmer_range[1] or lower < mgii_range[0]:
                print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) '
                      'exceeded the template range [1200., 7478.]. The excess part would be set to zero!')
            elif upper > mgii_range[1] and lower < balmer_range[0]:
                print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) '
                      'contained range [3500., 3686.] which is the gap between FeII templates and would be set to zero!')

        else:
            raise IndexError('The parameter pp only adopts a list of 3 or 6.')

        flux = integrate.trapz(yval[(xval >= lower) & (xval <= upper)], xval[(xval >= lower) & (xval <= upper)])
        return flux

    def Get_Fe_EW(self, fe_pp, conti_pp):
        """Calculate the Fe EW within [4434.,4684.] and [2250.,2651.]
            fe_pp = Fe_uv_norm, Fe_uv_FWHM, Fe_uv_shift, Fe_op_norm, Fe_op_FWHM, Fe_op_shift,
            conti_pp = PL_norm, PL_slope, poly_a, poly_b, poly_c """
        xval_uv = np.arange(2250.,2651.1)
        xval_op = np.arange(4434.,4684.1)
        fe_flux_uv = self.Fe_flux_mgii(xval_uv, fe_pp[:3])
        fe_flux_op = self.Fe_flux_balmer(xval_op, fe_pp[3:])
        conti_uv = conti_pp[0]*(xval_uv/3000.0)**conti_pp[1] + self.F_poly_conti(xval_uv, conti_pp[2:])
        conti_op = conti_pp[0]*(xval_op/3000.0)**conti_pp[1] + self.F_poly_conti(xval_op, conti_pp[2:])
        return np.array([np.sum(fe_flux_uv/conti_uv), np.sum(fe_flux_op/conti_op)])

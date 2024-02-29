# PyQSOFit: A code for quasar spectrum fitting
# Auther: Hengxiao Guo AT SHAO
# Email: hengxiaoguo AT gmail DOT com
# Co-Auther Yue Shen, Shu Wang, Wenke Ren, Colin J. Burke
# Email: rwk AT mail DOT ustc DOT edu DOT cn
#        colinjb2 AT illinois.edu
#
# -------------------------------------------------
# Version 1.2
# 10/26/2022 
# Key updates:
#     1) change the kmpfit to lmfit
#     2) no limit now for tie-line function
#     3) error estimation with MC or MCMC
#     4) coefficients in host decomposition are forced to be positive now
#     5) option for masking absorption pixels in emission line fitting.
#
# Version 1.2.1
# 07/12/2023
# Bug fix:
#     1) Add close fig function to avoid memory leak.
#        When savefig is set to True, the figure will be closed automatically. If one would like to show the figure
#        first, he/she can set savefig to False, call self.fig to exhibit the figure and save it manually.
#     2) Change the package sfdmap to sfdmap2.
#        Since the sfdmap is no longer maintained and has been conflicted with the latest numpy package, we change it
#        to a forked repository sfdmap2. ref: https://github.com/AmpelAstro/sfdmap2
#     3) Set the default jitter to 0.0 to avoid the unrepeatable fitting results. (temporary)
#        We do think the jitter will help to get unbiased fitting results especially when most of the initial parameters
#        are 0. However, since the jitter is not a physical parameter, it is not reasonable to set it as a default
#        global absolute value. Besides, variable results in repeated running are confusing to users. We will find out a
#        more feasible way to use the jitter in the future.
#
# Version 1.2.2
# 07/14/2023
# New features:
#     1) Add the second hdu table in qsopar.fits allow user to costume their results.
#        For now, there are only two params for user: the continuum luminosity wavelength and the Fe flux measurement
#        ranges.
#     2) Open the continuum luminosity measurement positions for user
#        Referencing the example of this code, one can add/delete/modify the wavelength positions for continuum
#        luminosity measurements in our final fits file by changing the params in qsopar.fits.
#     3) We adopt all local warning message in verbose switch.
#     4) Add a dict to better display the emission line names.
# Bug fix:
#     1) Improved the Luminosity measurement function
#     2) Fixed the error estimation function
#
# Version 1.2.3
# 07/19/2023
# Bug fix:
#     1) Add a few bug report in the to~do list
#     2) Automatically choose lstsq for PCA template
#     3) We uniformly set the default value of our code to -1.
#     4) Modify the function calculating the SN to avoid error arose by spectral discontinuous or low resolution.
#     5) Use the same initial logic for fur_result group and avoid the errors if the emission line is discard during
#     fitting procedure.
#     6）Amend the workflow of rejecting absorption line
#
# Version 1.2.4
# 07/24/2023
# Routine update:
#     1) Improve the host decomposition function using BC03 templates
#     2) Open the adjustment to the precision parameters for lmfit. By decrease the precision for continuum fitting,
#     the time consumption can be reduced and less likely tracked into local minimum.
# -------------------------------------------------
# Version 2.0
# 07/26/2023
# New Stable version for PyQSOFit supporting joint fitting of reverberation mapping spectra!
# New features:
#     1) Add new parameters to measure the host fraction at 5100A and 3000A.
#
# Version 2.1
# 07/28/2023
# New features:
#     1) Add a new method decompose the host component. In this method, we employ the prior of each PCQ template. By
#     applying a penalty, we restrict the factor of the last few component not to be too dominant. In this way, we
#     efficiently avoid the degeneration of the PCQ templates.
#     2) We rebuild the host decomposition module to make it more flexible. Now, the decomposition is performed through
#     HostDecomp.py module.
#     3) In HostDecomp.py module, we add the sigma measurements and Dn4000 estimation of the decomposed host component.
# -------------------------------------------------

import sys, os
import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

from sfdmap2 import sfdmap
from scipy import integrate, interpolate
from scipy.stats import median_abs_deviation
from lmfit import minimize, Parameters, report_fit

from PyAstronomy import pyasl
import scipy.optimize as opt
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM

from astropy.modeling.physical_models import BlackBody
from astropy.modeling.functional_models import Gaussian1D
from astropy.modeling import fitting

from astropy.table import Table

from .HostDecomp import Prior_decomp
from .HostDecomp import Linear_decomp
from .HostDecomp import ppxf_kinematics

import warnings

warnings.filterwarnings("ignore")


class QSOFit():

    def __init__(self, lam, flux, err, z, ra=-999, dec=-999, plateid=None, mjd=None, fiberid=None, path=None,
                 and_mask_in=None, or_mask_in=None, wdisp=None):
        """
        Get the input data prepared for the QSO spectral fitting
        
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
            
        path: str
            the path to the parameter file
            
        and_mask, or_mask: 1-D array with Npix, optional
            the bad pixels defined from SDSS data, which can be got from SDSS datacube.

        wdisp: float or 1-D array with Npix, optional
            The instrumental dispersion of the spectra in unit of pixel. If it is a float, we will assume the
            dispersion is uniform for full spectra; If a 1-D array is given, then we use the array to calculate the
            dispersion for each pixel separately. If that value is not given, we will deem it as 69 km/s, the average
            value for SDSS spectra. This value is only useful when calculating the kinematics features of the host.

        """

        self.lam_in = np.asarray(lam, dtype=np.float64)
        self.flux_in = np.asarray(flux, dtype=np.float64)
        self.err_in = np.asarray(err, dtype=np.float64)
        self.z = z
        self.and_mask_in = and_mask_in
        self.or_mask_in = or_mask_in
        self.wdisp = wdisp
        self.ra = ra
        self.dec = dec
        self.plateid = plateid
        self.mjd = mjd
        self.fiberid = fiberid
        self.path = path
        self.install_path = os.path.dirname(os.path.abspath(__file__))
        self.output_path = path

    def Fit(self, name=None, nsmooth=1, and_mask=False, or_mask=False, reject_badpix=True, deredden=True,
            wave_range=None,
            wave_mask=None, decompose_host=True, host_prior=False, host_prior_scale=0.2, host_line_mask=True,
            decomp_na_mask=False,
            qso_type='global', npca_qso=10, host_type='PCA', npca_gal=5, Fe_uv_op=True,
            poly=False, BC=False, rej_abs_conti=False, rej_abs_line=False, initial_guess=None,
            n_pix_min_conti=100, param_file_name='qsopar.fits', MC=False, MCMC=False, save_fits_name=None,
            nburn=20, nsamp=200, nthin=10, epsilon_jitter=0., linefit=True, save_result=True, plot_fig=True,
            save_fits_path='.',
            save_fig=True, plot_corner=True, verbose=False, kwargs_plot={}, kwargs_conti_emcee={},
            kwargs_line_emcee={}):

        """
        Fit the QSO spectrum and get different decomposed components and corresponding parameters
        
        Parameter:
        ----------
        name: str, optional
            source name, Default is None. If None, it will use plateid+mjd+fiberid as the name. If there are no
            such parameters, it will be empty.
            
        nsmooth: integer number, optional
            do n-pixel smoothing to the raw input flux and err spectra. The default is set to 1 (no smooth).
            It will return the same array size. We note that smooth the raw data is not suggested, this function is in case of some fail-fitted low S/N spectra.
              
        and_mask: bool, optional
            If True, and and_mask or or_mask is not None, it will delete the masked pixels, and only return the remained pixels. Default: False
            
        or_mask: bool, optional
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
            
        decompose_host: bool, optional    
            If True, the host galaxy-QSO decomposition will be applied. If no more than 100 pixels are negative, the result will be applied. The Decomposition is
            based on the PCA method of Yip et al. 2004 (AJ, 128, 585) & (128, 2603). Now the template is only available for redshift < 1.16 in specific absolute
            magnitude bins. For galaxy, the global model has 10 PCA components and first 5 will enough to reproduce 98.37% galaxy spectra. For QSO, the global model
            has 50, and the first 20 will reproduce 96.89% QSOs. If have i-band absolute magnitude, the Luminosity-redshift binned PCA components are available.
            Then the first 10 PCA in each bin is enough to reproduce most QSO spectrum. Default: False

        host_prior: bool, optional
            This parameter is only useful when the decompose_host is True and BC03 is False. If True, the code will
            adopt the prior parameters given in the pca file to perform host decomposition.

        host_prior_scale: float, optional
            If the prior decomposition is performed, the code will use this parameter to scale the prior penalty to the
            original chi2. Default: 0.2

        host_line_mask: bool, optional
            If True, the line region of galaxy will be masked when subtracted from original spectra. Default: True
            
        BC03: bool, optional
            if True, it will use Bruzual1 & Charlot 2003 host model to fit spectrum, high shift host will be low resolution R ~ 300, the rest is R ~ 2000. Default: False
        
        Mi: float, optional
            i-band absolute magnitude. It only works when decompose_host is True. If not None, the Luminosity redshift binned PCA will be used to decompose
            the spectrum. Default: None
            
        npca_gal: int, optional
            the number of galaxy PCA components applied. It only works when decompose_host is True. The default is 5,
            which is already account for 98.37% galaxies.
        
        npca_qso: int, optional
            the number of QSO PCA components applied. It only works when decompose_host is True. The default is 20,
            No matter the global or luminosity-redshift binned PCA is used, it can reproduce > 92% QSOs. The binned PCA
            is better if have Mi information.
         
        Fe_uv_op: bool, optional
            if True, fit continuum with UV and optical FeII template. Default: True

        poly: bool, optional
            if True, fit continuum with the polynomial component to account for the dust reddening. Default: False
        
        BC: bool, optional
            if True, fit continuum with Balmer continua from 1000 to 3646A. Default: False
            
        rej_abs_conti: bool, optional
            if True, it will iterate the continuum fitting once, rejecting 3 sigma outlier absorption pixels in the continuum
            (< 3500A), which might fall into the broad absorption lines. Default: False
            
        rej_abs_line: bool, optional
            if True, it will iterate the emission line fitting twice, rejecting 3 sigma outlier absorption pixels
            which might fall into the broad absorption lines. Default: False
        
        n_pix_min_conti: float, optional
            minimum number of negative pixels for host continuuum fit to be rejected. Default: 100
            
        param_file_name: str, optional
            name of the qso fitting parameter FITS file. Default: 'qsopar.fits'
        
        MC: bool, optional 
            if True, do Monte Carlo resampling of the spectrum based on the input error array to produce the MC error array.
            if False, the code will not save the MLE minimization error produced by lmfit since it is biased and can not be trusted.
            But it can be still output by using the lmfit attribute. Default: False
            
        MCMC: bool, optional 
            if True, do Markov Chain Monte Carlo sampling of the posterior probability densities after MLE fitting to produce the error array.
            Note: An error will be thrown if both MC and MCMC are True. Default: False
            
        nburn: int, optional
            the number of burn-in samples to run MCMC chain if MCMC=True. It only works when MCMC is True. Default: 20
        
        nsamp: int, optional
            the number of trials of the MC process to produce the error array (if MC=True) or number samples to run MCMC chain (if MCMC=True). Should be larger than 20. It only works when either MC or MCMC is True. Default: 200
            
        linefit: bool, optional
            if True, the emission line will be fitted. Default: True
           
        save_result: bool, optional
            if True, all the fitting results will be saved to a fits file, Default: True
            
        plot_fig: bool, optional
            if True, the fitting results will be plotted. Default: True
                    
        save_fig: bool, optional
            if True, the figure will be saved, and the path can be set by "save_fig_path". Default: True
            
        plot_corner: bool, optinoal
            whether or not to plot the corner plot results if MCMC=True. Default: True
        
        save_fig_path: str, optional
            the output path of the figure. If None, the default "save_fig_path" is set to "path"
        
        save_fits_path: str, optional
            the output path of the result fits. If None, the default "save_fits_path" is set to "path"
        
        save_fits_name: str, optional
            the output name of the result fits. Default: "result.fits"
            
        verbose: bool, optional
            turn on (True) or off (False) debugging output. Default: False
            
        kwargs_plot: dict, optional
            extra aguments for plot_fig for plotting results. See LINK TO PLOT_FIG_DOC. Default: {}
            
        kwargs_conti_emcee: dict, optional
            extra aguments for emcee Sampler for continuum fitting. Default: {}
            
        kwargs_line_emcee: dict, optional
            extra arguments for emcee Sampler for line fitting. Default: {}
            
        Return:
        -----------
        
        
        
        Properties:
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
            all the continuum fitting results, including best-fit parameters and Chisquare, etc. For details,
            see https://lmfit.github.io/lmfit-py/fitting.html
            
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
            Line fitting results for last complexes (From Lya to Ha) , including best-fit parameters, errors (lmfit derived) and Chisquare, etc. For details,
            see https://lmfit.github.io/lmfit-py/fitting.html
        
        .gauss_result: array
            3*n Gaussian parameters for all lines in the format of [scale, centerwave, sigma ], n is number of Gaussians for all complexes.
            ADD UNITS
            
        gauss_result_all: array
            [nsamp, 3*n] Gaussian parameters for all lines in the format of [scale, centerwave, sigma ], n is number of Gaussians for all complexes.
            ADD UNITS
            
        .conti_result: array
            continuum parameters, including widely used continuum parameters and monochromatic flux at 1350, 3000
            and 5100 Angstrom, etc. The corresponding names are listed in .conti_result_name. For all continuum fitting results,
            go to .conti_fit.params. 
            
        .conti_result_name: array
            the names for .conti_result.
            
        .fur_result: array
            emission line parameters, including FWHM, sigma, EW, measured from whole model of each main broad emission line covered.
            The corresponding names are listed in .line_result_name.
            
        .fur_result_name: array
            the names for .fur_result.
            
        .line_result: array
            emission line parameters, including FWHM, sigma, EW, measured from whole model of each main broad emission line covered,
            and fitting parameters of each Gaussian component. The corresponding names are listed in .line_result_name.
            
        .line_result_name: array
            the names for .line_result.
            
        .uniq_linecomp_sort: array
            the sorted complex names.
            
        .all_comp_range: array
            the start and end wavelength for each complex. e.g., Hb is [4640.  5100.] AA.
            
        .linelist: array
            the information listed in the param_file_name (qsopar.fits).
        """

        # Parameters that are set here should generally not be changed unless you know what you are doing

        self.name = name
        self.wave_range = wave_range
        self.wave_mask = wave_mask
        self.decompose_host = decompose_host
        self.linefit = linefit
        self.host_line_mask = host_line_mask
        self.host_type = host_type
        self.qso_type = qso_type
        self.npca_gal = npca_gal
        self.npca_qso = npca_qso
        self.maxOLs = 10
        self.alpha = 0.05
        self.initial_guess = initial_guess
        self.Fe_uv_op = Fe_uv_op
        self.poly = poly
        self.BC = BC
        self.rej_abs_conti = rej_abs_conti
        self.rej_abs_line = rej_abs_line
        self.rej_abs_line_max_niter = 2
        self.n_pix_min_conti = n_pix_min_conti  # pixels
        self.MC = MC
        self.MCMC = MCMC
        self.nburn = nburn
        self.nsamp = nsamp
        self.nthin = nthin
        self.epsilon_jitter = epsilon_jitter
        self.kwargs_conti_emcee = kwargs_conti_emcee
        self.kwargs_line_emcee = kwargs_line_emcee
        self.save_fig = save_fig
        self.plot_corner = plot_corner
        self.verbose = verbose
        self.param_file_name = param_file_name

        # Initial precision parameters for lmfit
        self.xtol_conti = 1e-8
        self.ftol_conti = 1e-10
        self.xtol_line = 1e-10
        self.ftol_line = 1e-10

        # Initial parameters for prior decomposition
        self.host_prior = host_prior
        self.host_prior_scale = host_prior_scale
        self.decomp_na_mask = decomp_na_mask

        self.read_out_params(os.path.join(self.path, self.param_file_name))

        # get the source name in plate-mjd-fiber, if no then None
        if name is None:
            if np.array([self.plateid, self.mjd, self.fiberid]).any() is not None:
                self.sdss_name = str(self.plateid).zfill(4) + '-' + str(self.mjd) + '-' + str(self.fiberid).zfill(4)
            else:
                self.sdss_name = ''
        else:
            self.sdss_name = name

        if self.plateid is None:
            self.plateid = 0
        if self.mjd is None:
            self.mjd = 0
        if self.fiberid is None:
            self.fiberid = 0

        # set default path for figure and fits
        if save_fits_name == None:
            if self.sdss_name == '':
                save_fits_name = 'result'
            else:
                save_fits_name = self.sdss_name
        else:
            save_fits_name = save_fits_name

        dustmap_path = os.path.join(self.install_path, 'sfddata')

        # Clean the data

        # Remove with error equal to 0 or inifity
        ind_gooderror = np.where(
            (self.err_in > 0) & np.isfinite(self.err_in) & (self.flux_in != 0) & np.isfinite(self.flux_in), True, False)
        self.err = self.err_in[ind_gooderror]
        self.flux = self.flux_in[ind_gooderror]
        self.lam = self.lam_in[ind_gooderror]

        # Renew And/or mask index
        if (self.and_mask_in is not None) and (self.or_mask_in is not None):
            self.and_mask_in = self.and_mask_in[ind_gooderror]
            self.or_mask_in = self.or_mask_in[ind_gooderror]
        else:
            self.and_mask_in = None
            self.or_mask_in = None

        # Clean and/or mask
        if (and_mask == True) and (self.and_mask_in is not None):
            self._MaskSdssAndOr(self.lam, self.flux, self.err, and_mask, or_mask)
        # Clean bad pixel
        if reject_badpix == True:
            self._RejectBadPix(self.lam, self.flux, self.err)
        # Smooth the data
        if nsmooth is not None:
            self.flux = self.Smooth(self.flux, nsmooth)
            self.err = self.Smooth(self.err, nsmooth)
        # Set fitting wavelength range
        if wave_range is not None:
            self._WaveTrim(self.lam, self.flux, self.err, self.z)
        # Set manual wavelength mask
        if wave_mask is not None:
            self._WaveMsk(self.lam, self.flux, self.err, self.z)
        # Deredden
        if deredden == True and self.ra != -999. and self.dec != -999.:
            self._DeRedden(self.lam, self.flux, self.err, self.ra, self.dec, dustmap_path)

        self._RestFrame(self.lam, self.flux, self.err, self.z)
        self._CalculateSN(self.wave, self.flux)
        self._OrignialSpec(self.wave, self.flux, self.err)

        """
        Do host decomposition
        """
        if decompose_host == True:
            self.decompose_host_qso(self.wave, self.flux, self.err, self.install_path)
        else:
            self.decomposed = False
            self.host_result = np.array([])
            self.host_result_type = np.array([])
            self.host_result_name = np.array([])

            self.frac_host_4200 = -1.
            self.frac_host_5100 = -1.

        """
        Fit the continuum
        """
        self.fit_continuum(self.wave, self.flux, self.err, self.ra, self.dec, self.plateid, self.mjd, self.fiberid)

        """
        Fit the emission lines
        """
        if linefit == True:
            self.fit_lines(self.wave, self.line_flux, self.err, self.conti_fit)
        else:
            self.ncomp = 0
            self.line_result = np.array([])
            self.line_result_type = np.array([])
            self.line_result_name = np.array([])
            self.gauss_result = np.array([])
            self.all_comp_range = np.array([])
            self.uniq_linecomp_sort = np.array([])

        """
        Save the results
        """
        if save_result == True:
            self.save_result(self.conti_result, self.conti_result_type, self.conti_result_name, self.line_result,
                             self.line_result_type, self.line_result_name, save_fits_path, save_fits_name)

        """
        Plot the results
        """
        if plot_fig == True:
            self.plot_fig(**kwargs_plot)

        return

    def _MaskSdssAndOr(self, lam, flux, err, and_mask, or_mask):
        """
        Remove SDSS and_mask and or_mask points are not zero
        Parameter:
        ----------
        lam: wavelength
        flux: flux
        err: 1 sigma error
        and_mask: SDSS flag "and_mask", mask out all non-zero pixels
        
        Retrun:
        ---------
        return the same size array of wavelength, flux, error
        """
        if (and_mask == True) and (or_mask == True):
            ind = np.where((self.and_mask_in == 0) & (self.and_mask_in == 0), True, False)
        if (and_mask == True) and (or_mask == False):
            ind = np.where(self.and_mask_in == 0, True, False)
        if (and_mask == False) and (or_mask == True):
            ind = np.where(self.or_mask == 0, True, False)

        self.lam, self.flux, self.err = lam[ind], flux[ind], err[ind]

    def _RejectBadPix(self, lam, flux, err, maxOLs=10, alpha=0.05):
        """
        Reject outliers in spectrum such as cosmic rays.
        See https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/outlier.html
        
        Parameter:
        ----------
        lam: array, required
            wavelength
            
        flux: array, required
            flux
            
        err: array, required
            1 sigma error
            
        maxOLS: int, optional
            Maximum number of outliers to reject. Default: 10
            
        alpha: float, optional
            Significance. Default: 0.05
            
        Return:
        ---------
        
        """
        # -----remove bad pixels, but not for high SN spectrum------------
        ind_bad = pyasl.pointDistGESD(flux, maxOLs, alpha)
        wv = np.asarray([i for j, i in enumerate(lam) if j not in ind_bad[1]], dtype=np.float64)
        fx = np.asarray([i for j, i in enumerate(flux) if j not in ind_bad[1]], dtype=np.float64)
        er = np.asarray([i for j, i in enumerate(err) if j not in ind_bad[1]], dtype=np.float64)
        # TODO: Below lines are confusing and generally bad practice
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = wv, fx, er
        return self.lam, self.flux, self.err

    def _WaveTrim(self, lam, flux, err, z):
        """
        Trim spectrum with a range in the rest frame. 
        """
        # trim spectrum e.g., local fit emiision lines
        ind_trim = np.where((lam / (1 + z) > self.wave_range[0]) & (lam / (1 + z) < self.wave_range[1]), True, False)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = lam[ind_trim], flux[ind_trim], err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError("No enough pixels in the input wave_range!")
        return self.lam, self.flux, self.err

    def _WaveMsk(self, lam, flux, err, z):
        """Block the bad pixels or absorption lines in spectrum."""

        for msk in range(len(self.wave_mask)):
            try:
                ind_not_mask = ~np.where(
                    (lam / (1 + z) > self.wave_mask[msk, 0]) & (lam / (1 + z) < self.wave_mask[msk, 1]),
                    True, False)
            except IndexError:
                raise RuntimeError("Wave_mask should be 2D array, e.g., np.array([[2000,3000],[3100,4000]]).")
            # TODO: test if these arrays are needed to be deleted
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
        err_unred = err * flux_unred / flux
        flux_unred[zero_flux] = 0
        del self.flux, self.err
        self.flux = flux_unred
        self.err = err_unred
        return self.flux

    def _RestFrame(self, lam, flux, err, z):
        """Move wavelenth and flux to rest frame"""
        self.wave = lam / (1 + z)
        self.flux = flux * (1 + z)
        self.err = err * (1 + z)
        return self.wave, self.flux, self.err

    def _OrignialSpec(self, wave, flux, err):
        """Save the orignial spectrum before host galaxy decompsition"""
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _CalculateSN(self, wave, flux, alter=True):
        """
        Calculate the spectral SN ratio for 1350, 3000, 5100A, return the mean value of Three spots
        This function will automatically check if the 50A vicinity of at the default three wavelength contain more than
        10 pixels. If so, this function will calculate the continuum SN ratio from available regions. If not, it may
        imply that the give spectrum are very low resolution or have frequent gaps in their wavelength coverage. We
        provide another algorithm to calculate the SNR regardless of the continuum.
        :param wave:
        :param flux:
        :return:
        """
        ind5100 = np.where((wave > 5080) & (wave < 5130), True, False)
        ind3000 = np.where((wave > 3000) & (wave < 3050), True, False)
        ind1350 = np.where((wave > 1325) & (wave < 1375), True, False)

        if np.all(np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) < 10):

            if alter is False:
                self.SN_ratio_conti = -1.
                return self.SN_ratio_conti

            # referencing: www.stecf.org/software/ASTROsoft/DER_SNR/
            input_data = np.array(flux)
            # Values that are exactly zero (padded) are skipped
            input_data = np.array(input_data[np.where(input_data != 0.0)])
            n = len(input_data)
            # For spectra shorter than this, no value can be returned
            if (n > 4):
                signal = np.median(input_data)
                noise = 0.6052697 * np.median(np.abs(2.0 * input_data[2:n - 2] - input_data[0:n - 4] - input_data[4:n]))
                self.SN_ratio_conti = float(signal / noise)
            else:
                self.SN_ratio_conti = -1.

        else:
            tmp_SN = np.array([flux[ind5100].mean() / flux[ind5100].std(), flux[ind3000].mean() / flux[ind3000].std(),
                               flux[ind1350].mean() / flux[ind1350].std()])
            if not np.all(np.isnan(tmp_SN)):
                self.SN_ratio_conti = np.nanmean(tmp_SN)
            else:
                self.SN_ratio_conti = -1.

        return self.SN_ratio_conti

    def decompose_host_qso(self, wave, flux, err, path):
        """Decompose the host galaxy from QSO"""
        if self.host_prior is True:
            prior_fitter = Prior_decomp(self.wave, self.flux, self.err, self.npca_gal, self.npca_qso,
                                        path, host_type=self.host_type, qso_type=self.qso_type,
                                        na_mask=self.decomp_na_mask)
            datacube, frac_host_4200, frac_host_5100, qso_par, gal_par = prior_fitter.auto_decomp(self.host_prior_scale)
        else:
            linear_fitter = Linear_decomp(self.wave, self.flux, self.err, self.npca_gal, self.npca_qso, path,
                                          host_type=self.host_type, qso_type=self.qso_type,
                                          na_mask=self.decomp_na_mask)
            datacube, frac_host_4200, frac_host_5100, qso_par, gal_par = linear_fitter.auto_decomp()

        # for some negative host template, we do not do the decomposition # not apply anymore
        # For a few cases, the host template is too weak that the host spectra (data - qso) would be mostly negative
        # through the data itself wouldn't
        flux_level = np.median(np.abs(datacube[1, :]))
        host_spec = datacube[1, :] - datacube[4, :]
        if np.sum(np.where(datacube[3, :] < 0, True, False) | np.where(datacube[4, :] < 0, True, False)) > 0.1 * \
                datacube.shape[1] or np.median(datacube[3, :]) < 0.01 * flux_level or np.median(host_spec) < 0:
            self.host = np.zeros(len(wave))
            self.decomposed = False
            self.host_result = np.array([])
            self.host_result_type = np.array([])
            self.host_result_name = np.array([])
            if self.verbose:
                print('Got negative host galaxy / QSO flux over 10% of coverage, decomposition is not applied!')
        else:
            self.decomposed = True
            del self.wave, self.flux, self.err
            self.wave = datacube[0, :]

            rchi2_decomp = np.sum((datacube[1, :] - datacube[4, :] - datacube[3, :]) ** 2 / datacube[2, :] ** 2) / (
                    len(datacube[1, :]) - self.npca_qso - self.npca_gal)

            # Block OIII, Ha, NII, SII, OII, Ha, Hb, Hr, Hdelta
            if self.host_line_mask == True:
                line_mask = np.where(
                    (self.wave < 4970) & (self.wave > 4950) | (self.wave < 5020) & (self.wave > 5000) |
                    (self.wave < 6590) & (self.wave > 6540) | (self.wave < 6740) & (self.wave > 6710) |
                    (self.wave < 3737) & (self.wave > 3717) | (self.wave < 4872) & (self.wave > 4852) |
                    (self.wave < 4350) & (self.wave > 4330) | (self.wave < 4111) & (self.wave > 4091),
                    True, False)
            else:
                line_mask = np.full(len(self.wave), False)

            f = interpolate.interp1d(self.wave[~line_mask], datacube[3, :][~line_mask], bounds_error=False,
                                     fill_value=0)
            masked_host = f(self.wave)
            self.spec = datacube[1, :]  # Original spectra flux
            self.flux = datacube[1, :] - masked_host  # QSO flux without host
            self.err = datacube[2, :]
            self.host = datacube[3, :]
            self.qso = datacube[4, :]
            self.host_data = datacube[1, :] - self.qso

            if self.MC is True:
                n_iter = self.nsamp
            else:
                n_iter = 0
            fit_range = (4000, 5350)
            try:
                fit_pp, ppxf_mask, ppxf_model = ppxf_kinematics(self.wave, self.host_data, self.err, path, fit_range,
                                                                MC_iter=n_iter)
                sigma, sigma_err, v_off, v_off_err, rchi2_ppxf = fit_pp
            except:
                ppxf_model = np.zeros(len(self.wave))
                ppxf_mask = np.ones(len(self.wave), dtype=bool)
                sigma, sigma_err, v_off, v_off_err, rchi2_ppxf = -999, -999, -999, -999, -999
            self.ppxf_mask = ppxf_mask
            self.ppxf_model = ppxf_model

            # TODO: A very messy way to get the SN ratio, will integrated to the _Calculate_SN function in the future.
            input_data = np.array(self.host_data)
            input_data = np.array(input_data[np.where(input_data != 0.0)])
            n = len(input_data)
            # For spectra shorter than this, no value can be returned
            if (n > 4):
                signal = np.median(input_data)
                noise = 0.6052697 * np.median(np.abs(2.0 * input_data[2:n - 2] - input_data[0:n - 4] - input_data[4:n]))
                SN_host = float(signal / noise)
            else:
                SN_host = -1.

            # measure the Dn4000 of the host galaxy
            # TODO: will be separated as a independent function in the future
            lower_idx = np.where((self.wave > 3850) & (self.wave < 3950), True, False)
            upper_idx = np.where((self.wave > 4000) & (self.wave < 4100), True, False)
            if np.sum(lower_idx) > 10 and np.sum(upper_idx) > 10:
                Dn4000 = np.mean(self.host[upper_idx]) / np.mean(self.host[lower_idx])
            else:
                Dn4000 = -1.

            self.host_result = np.array(
                [SN_host, rchi2_decomp, frac_host_4200, frac_host_5100, Dn4000, sigma, sigma_err, v_off, v_off_err,
                 rchi2_ppxf])
            self.host_result_type = np.full(len(self.host_result), 'float')
            self.host_result_name = np.array(
                ['SN_host', 'rchi2_decomp', 'frac_host_4200', 'frac_host_5100', 'Dn4000', 'sigma', 'sigma_err', 'v_off',
                 'v_off_err', 'rchi2_ppxf'])

            self.host_result = np.concatenate([self.host_result, gal_par, qso_par])
            self.host_result_type = np.concatenate(
                [self.host_result_type, np.full(len(gal_par), 'float'), np.full(len(qso_par), 'float')])
            self.host_result_name = np.concatenate([self.host_result_name,
                                                    np.array(['gal_par_' + str(i) for i in range(len(gal_par))]),
                                                    np.array(['qso_par_' + str(i) for i in range(len(qso_par))])])

        return self.wave, self.flux, self.err

    def fit_continuum(self, wave, flux, err, ra, dec, plateid, mjd, fiberid):
        """Fit the continuum with PL, Polynomial, UV/optical FeII, Balmer continuum
        
        Parameter:
        ----------
        lam: array, required
            wavelength
            
        flux: array, required
            flux
            
        err: array, required
            1 sigma error
            
        conti_windows: 2d array, optional
            Continuum fitting windows. If None, use default windows. For special situations, you
            may need to change these to improve the fitting (e.g., Ly-alpha absorption troughs, etc.) Default: None
            
        Return:
        ---------
        
        
        """
        self.fe_uv = np.genfromtxt(os.path.join(self.install_path, 'fe_uv.txt'))
        self.fe_op = np.genfromtxt(os.path.join(self.install_path, 'fe_optical.txt'))

        # Read line parameter file
        contilist, window_all = read_conti_params(os.path.join(self.path, self.param_file_name))

        # Convert the windows to a mask
        tmp_all = np.array([np.repeat(False, len(wave))]).flatten()
        for jj in range(len(window_all)):
            tmp = np.where((wave > window_all[jj, 0]) & (wave < window_all[jj, 1]), True, False)
            tmp_all = np.any([tmp_all, tmp], axis=0)

        if wave[tmp_all].shape[0] < 10:
            if self.verbose:
                print('Less than 10 total pixels in the continuum fitting.')

        """
        Setup parameters for fitting
        """

        # Set initial parameters for continuum
        pp0 = np.array([c['initial'] for c in contilist])

        # It's usually a good idea to jitter the parameters a bit
        pp0 += np.abs(np.random.normal(0, self.epsilon_jitter, len(pp0)))

        fit_params = Parameters()
        # norm_factor, FWHM, and small shift of wavelength for the MgII Fe_template
        fit_params.add('Fe_uv_norm', value=pp0[0], min=contilist[0]['min'], max=contilist[0]['max'],
                       vary=bool(contilist[0]['vary']))
        fit_params.add('Fe_uv_FWHM', value=pp0[1], min=contilist[1]['min'], max=contilist[1]['max'],
                       vary=bool(contilist[1]['vary']))
        fit_params.add('Fe_uv_shift', value=pp0[2], min=contilist[2]['min'], max=contilist[2]['max'],
                       vary=bool(contilist[2]['vary']))
        # same as above but for the Hbeta/Halpha Fe template
        fit_params.add('Fe_op_norm', value=pp0[3], min=contilist[3]['min'], max=contilist[3]['max'],
                       vary=bool(contilist[3]['vary']))
        fit_params.add('Fe_op_FWHM', value=pp0[4], min=contilist[4]['min'], max=contilist[4]['max'],
                       vary=bool(contilist[4]['vary']))
        fit_params.add('Fe_op_shift', value=pp0[5], min=contilist[5]['min'], max=contilist[5]['max'],
                       vary=bool(contilist[5]['vary']))
        # norm_factor for continuum f_lambda = (lambda/3000.0)^{-alpha}
        fit_params.add('PL_norm', value=pp0[6], min=contilist[6]['min'], max=contilist[6]['max'],
                       vary=bool(contilist[6]['vary']))
        # slope for the power-law continuum
        fit_params.add('PL_slope', value=pp0[7], min=contilist[7]['min'], max=contilist[7]['max'],
                       vary=bool(contilist[7]['vary']))
        # norm, Te and Tau_e for the Balmer continuum at <3646 A
        fit_params.add('Blamer_norm', value=pp0[8], min=contilist[8]['min'], max=contilist[8]['max'],
                       vary=bool(contilist[8]['vary']))
        fit_params.add('Balmer_Te', value=pp0[9], min=contilist[9]['min'], max=contilist[9]['max'],
                       vary=bool(contilist[9]['vary']))
        fit_params.add('Balmer_Tau', value=pp0[10], min=contilist[10]['min'], max=contilist[10]['max'],
                       vary=bool(contilist[10]['vary']))
        # polynomial for the continuum f = a_0 x^0 + a_1 x^1 + a_2 x^2 + ... 
        # XXX Bounds have to be None so lmfit will select them automatically to avoid numerical problems
        fit_params.add('conti_a_0', value=pp0[11], min=None, max=None, vary=bool(contilist[11]['vary']))
        fit_params.add('conti_a_1', value=pp0[12], min=None, max=None, vary=bool(contilist[12]['vary']))
        fit_params.add('conti_a_2', value=pp0[13], min=None, max=None, vary=bool(contilist[13]['vary']))

        # Check if we will attempt to fit the UV FeII continuum region
        ind_uv = np.where((wave[tmp_all] > 1200) & (wave[tmp_all] < 3500), True, False)
        if (self.Fe_uv_op == False) or (np.sum(ind_uv) <= self.n_pix_min_conti):
            fit_params['Fe_uv_norm'].value = 0
            fit_params['Fe_uv_norm'].vary = False
            fit_params['Fe_uv_FWHM'].vary = False
            fit_params['Fe_uv_shift'].vary = False

        # Check if we will attempt to fit the optical FeII continuum region
        ind_opt = np.where((wave[tmp_all] > 3686.) & (wave[tmp_all] < 7484.), True, False)
        if (self.Fe_uv_op == False and self.BC == False) or (np.sum(ind_opt) <= self.n_pix_min_conti):
            fit_params['Fe_op_norm'].value = 0
            fit_params['Fe_op_norm'].vary = False
            fit_params['Fe_op_FWHM'].vary = False
            fit_params['Fe_op_shift'].vary = False

        # Check if we will attempt to fit the Balmer continuum region
        ind_BC = np.where(wave[tmp_all] < 3646, True, False)
        if (self.BC == False) or (np.sum(ind_BC) <= 100):
            fit_params['Blamer_norm'].value = 0
            fit_params['Blamer_norm'].vary = False
            fit_params['Balmer_Te'].vary = False
            fit_params['Balmer_Tau'].vary = False

        # Check if we will fit the polynomial component
        if self.poly == False:
            fit_params['conti_a_0'].value = 0
            fit_params['conti_a_1'].value = 0
            fit_params['conti_a_2'].value = 0
            fit_params['conti_a_0'].vary = False
            fit_params['conti_a_1'].vary = False
            fit_params['conti_a_2'].vary = False

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

        # Get continuum model ahead of time and pass it to the residuals function
        if self.Fe_uv_op == True and self.poly == False and self.BC == False:
            _conti_model = lambda xval, pp: self.PL(xval, pp) + self.Fe_flux_mgii(xval, pp[0:3]) + self.Fe_flux_balmer(
                xval, pp[3:6])
        elif self.Fe_uv_op == True and self.poly == True and self.BC == False:
            _conti_model = lambda xval, pp: self.PL(xval, pp) + self.Fe_flux_mgii(xval, pp[0:3]) + self.Fe_flux_balmer(
                xval, pp[3:6]) + self.F_poly_conti(xval, pp[11:])
        elif self.Fe_uv_op == True and self.poly == False and self.BC == True:
            _conti_model = lambda xval, pp: self.PL(xval, pp) + self.Fe_flux_mgii(xval, pp[0:3]) + self.Fe_flux_balmer(
                xval, pp[3:6]) + self.Balmer_conti(xval, pp[8:11])
        elif self.Fe_uv_op == False and self.poly == True and self.BC == False:
            _conti_model = lambda xval, pp: self.PL(xval, pp) + self.F_poly_conti(xval, pp[11:])
        elif self.Fe_uv_op == False and self.poly == False and self.BC == False:
            _conti_model = lambda xval, pp: self.PL(xval, pp)
        elif self.Fe_uv_op == False and self.poly == False and self.BC == True:
            _conti_model = lambda xval, pp: self.PL(xval, pp) + self.Balmer_conti(xval, pp[8:11])
        elif self.Fe_uv_op == True and self.poly == True and self.BC == True:
            _conti_model = lambda xval, pp: self.PL(xval, pp) + self.Fe_flux_mgii(xval, pp[0:3]) + self.Fe_flux_balmer(
                xval, pp[3:6]) + self.F_poly_conti(xval, pp[11:]) + self.Balmer_conti(xval, pp[8:11])
        elif self.Fe_uv_op == False and self.poly == True and self.BC == True:
            _conti_model = lambda xval, pp: self.PL(xval, pp) + self.Fe_flux_balmer(xval, pp[3:6]) + self.F_poly_conti(
                xval, pp[11:]) + self.Balmer_conti(xval, pp[8:11])
        else:
            raise RuntimeError('Invalid options for continuum model!')

        """
        Perform the fitting
        
        Perform two iterations to remove 3-sigma pixels below the first continuum fit
        to avoid the continuum windows that fall within a BAL trough
        """

        # Print parameters to be fit
        if self.verbose:
            fit_params.pretty_print()
            print('Fitting continuum')

        # Initial fit of the continuum
        conti_fit = minimize(self._residuals, fit_params,
                             args=(wave[tmp_all], flux[tmp_all], err[tmp_all], _conti_model),
                             calc_covar=False, xtol=self.xtol_conti, ftol=self.ftol_conti)
        params_dict = conti_fit.params.valuesdict()
        par_names = list(params_dict.keys())
        params = list(params_dict.values())

        # Calculate the continuum fit and mask the absorption lines before re-fitting
        if self.rej_abs_conti == True:
            if self.poly == True:
                tmp_conti = self.PL(wave[tmp_all], params) + self.F_poly_conti(wave[tmp_all], params[11:])
            else:
                tmp_conti = self.PL(wave[tmp_all], params)

            ind_noBAL = ~np.where((flux[tmp_all] < tmp_conti - 3 * err[tmp_all]) & (wave[tmp_all] < 3500), True, False)

            # Second fit of the continuum
            conti_fit = minimize(self._residuals, fit_params,
                                 args=(wave[tmp_all][ind_noBAL],
                                       self.Smooth(flux[tmp_all][ind_noBAL], 10),  # XXX Why smooth here?
                                       err[tmp_all][ind_noBAL], _conti_model),
                                 calc_covar=False, xtol=self.xtol_conti, ftol=self.ftol_conti)
            params_dict = conti_fit.params.valuesdict()
            params = list(params_dict.values())

        else:
            ind_noBAL = np.full(len(wave[tmp_all]), True)

        # Print fit report
        if self.verbose:
            print('Fit report')
            report_fit(conti_fit.params)

        """
        Uncertainty estimation
        """
        if (self.MCMC == True or self.MC == True) and self.nsamp > 0:

            if (self.MCMC == True) and (self.MC == False):
                """
                MCMC sampling
                """
                # Sample with MCMC, using the initial minima
                conti_samples = minimize(self._residuals, params=conti_fit.params,
                                         args=(wave[tmp_all][ind_noBAL],
                                               self.Smooth(flux[tmp_all][ind_noBAL], 10),  # XXX Why smooth here?
                                               err[tmp_all][ind_noBAL], _conti_model),
                                         method='emcee', nan_policy='omit',
                                         burn=self.nburn, steps=self.nsamp, thin=self.nthin,
                                         **self.kwargs_conti_emcee, is_weighted=True)
                samples_dict = conti_samples.params.valuesdict()
                df_samples = conti_samples.flatchain

                # Print fit report
                if self.verbose:
                    print(
                        f'acceptance fraction = {np.mean(conti_samples.acceptance_fraction)} +/- {np.std(conti_samples.acceptance_fraction)}')
                    # As a rule of thumb the value should be between 0.2 and 0.5
                    print('median of posterior probability distribution')
                    print('--------------------------------------------')
                    report_fit(conti_samples.params)

                if self.plot_corner:
                    import corner
                    truths = [params_dict[k] for k in df_samples.columns.values.tolist()]
                    emcee_plot = corner.corner(conti_samples.flatchain, labels=conti_samples.var_names,
                                               quantiles=[0.16, 0.5, 0.84], truths=truths)

                # After doing MCMC, the samples array will not include fixed parameters
                # We need to add their fixed values back in so the order is preserved

                # Loop through each parameter
                for k, name in enumerate(par_names):
                    # Add a column with all zeros if the parameter is fixed
                    if name not in df_samples.columns.values.tolist():
                        df_samples[name] = params_dict[name]

                # Sort the samples dataframe back to its original order
                df_samples = df_samples[par_names]
                samples = df_samples.to_numpy()

            elif (self.MCMC == False) and (self.MC == True):
                """
                MC resampling
                """
                # Resample the spectrum using the measurement error

                samples = np.zeros((self.nsamp, len(pp0)))

                for k in range(self.nsamp):
                    flux_resampled = flux + np.random.randn(len(flux)) * err

                    conti_fit = minimize(self._residuals, conti_fit.params,
                                         args=(wave[tmp_all][ind_noBAL],
                                               self.Smooth(flux_resampled[tmp_all][ind_noBAL], 10),
                                               # XXX Why smooth here?
                                               err[tmp_all][ind_noBAL], _conti_model),
                                         calc_covar=False)
                    params_dict = conti_fit.params.valuesdict()
                    params_mc = list(params_dict.values())
                    samples[k] = params_mc

            else:
                RuntimeError('MCMC and MC modes are both True')

            # Parameter error estimates
            params_err = get_err(samples, axis=0)
            """
            Calculate physical properties
            """

            # Calculate continuum luminosity errors
            Ls = np.empty((np.shape(samples)[0], len(self.L_conti_wave)))
            # Samples loop
            for k, s in enumerate(samples):
                Ls[k] = self._L_conti(wave, s, self.L_conti_wave)

            L_std = get_err(Ls)

            # Calculate FeII flux errors
            Fe_flux_results = np.empty((len(samples), np.shape(np.ravel(self.Fe_flux_range))[0] // 2))

            if self.Fe_flux_range is not None:
                # Samples loop
                for k, s in enumerate(samples):
                    Fe_flux_results[k], Fe_flux_type, Fe_flux_name = self.Get_Fe_flux(self.Fe_flux_range, s[:6])

                Fe_flux_std = get_err(Fe_flux_results)
            else:
                Fe_flux_std = None

            # Point estimates
            # Calculate continuum luminosities
            L = self._L_conti(wave, params, self.L_conti_wave)

            # Calculate FeII flux
            Fe_flux_result, Fe_flux_type, Fe_flux_name = self.Get_Fe_flux(self.Fe_flux_range, params[:6])

            """
            Save the results
            """
            # For standard parameters
            par_names = list(params_dict.keys())
            par_err_names = [n + '_err' for n in par_names]

            self.conti_result = np.concatenate(([ra, dec, str(plateid), str(mjd), str(fiberid), self.z,
                                                 self.SN_ratio_conti],
                                                list(chain.from_iterable(zip(params, params_err)))))
            self.conti_result_type = np.full(len(self.conti_result), 'float')
            self.conti_result_type[2:5] = 'int'
            self.conti_result_name = np.concatenate((['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift',
                                                      'SN_ratio_conti'],
                                                     list(chain.from_iterable(zip(par_names, par_err_names)))))

            # For customized parameters
            self.conti_result = np.append(self.conti_result, list(chain.from_iterable(zip(L, L_std))))
            self.conti_result_type = np.append(self.conti_result_type, ['float'] * (len(L) * 2))
            L_names = [f'L{int(lam):d}' for lam in self.L_conti_wave]
            L_err_names = [f'L{int(lam):d}_err' for lam in self.L_conti_wave]
            self.conti_result_name = np.append(self.conti_result_name,
                                               list(chain.from_iterable(zip(L_names, L_err_names))))

            Fe_flux_err_name = [n + '_err' for n in Fe_flux_name]
            self.conti_result = np.append(self.conti_result,
                                          list(chain.from_iterable(zip(Fe_flux_result, Fe_flux_std))))
            self.conti_result_type = np.append(self.conti_result_type,
                                               list(chain.from_iterable(zip(Fe_flux_type, Fe_flux_type))))
            self.conti_result_name = np.append(self.conti_result_name,
                                               list(chain.from_iterable(zip(Fe_flux_name, Fe_flux_err_name))))
        else:

            """
            Calculate physical properties
            """

            # Point estimates
            # Calculate continuum luminosities
            L = self._L_conti(wave, params, self.L_conti_wave)

            # Calculate FeII flux
            Fe_flux_result, Fe_flux_type, Fe_flux_name = self.Get_Fe_flux(self.Fe_flux_range, params[:6])

            """
            Save the results
            """
            # For standard parameters
            self.conti_result = np.concatenate(([ra, dec, str(plateid), str(mjd), str(fiberid), self.z,
                                                 self.SN_ratio_conti],
                                                params))
            self.conti_result_type = np.full(len(self.conti_result), 'float')
            self.conti_result_type[2:5] = 'int'
            self.conti_result_name = np.concatenate((['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift',
                                                      'SN_ratio_conti'], par_names))

            # For customized parameters
            self.conti_result = np.append(self.conti_result, L)
            self.conti_result_type = np.append(self.conti_result_type, ['float'] * len(L))
            self.conti_result_name = np.append(self.conti_result_name, [f'L{int(lam):d}' for lam in self.L_conti_wave])

            self.conti_result = np.append(self.conti_result, Fe_flux_result)
            self.conti_result_type = np.append(self.conti_result_type, Fe_flux_type)
            self.conti_result_name = np.append(self.conti_result_name, Fe_flux_name)

        self.conti_result = np.append(self.conti_result, self.host_result)
        self.conti_result_type = np.append(self.conti_result_type, self.host_result_type)
        self.conti_result_name = np.append(self.conti_result_name, self.host_result_name)

        self.conti_fit = conti_fit
        self.conti_params = params
        self.tmp_all = tmp_all

        # Save individual models
        self.f_fe_mgii_model = self.Fe_flux_mgii(wave, params[0:3])
        self.f_fe_balmer_model = self.Fe_flux_balmer(wave, params[3:6])
        self.f_pl_model = self.PL(wave, params)
        self.f_bc_model = self.Balmer_conti(wave, params[8:11])
        self.f_poly_model = self.F_poly_conti(wave, params[11:])
        self.f_conti_model = self.f_pl_model + self.f_fe_mgii_model + self.f_fe_balmer_model + self.f_poly_model + self.f_bc_model
        self.line_flux = flux - self.f_conti_model
        self.PL_poly_BC = self.f_pl_model + self.f_poly_model + self.f_bc_model

        return self.conti_result, self.conti_result_name

    def _L_conti(self, wave, pp, waves=np.array([1350, 3000, 5100])):
        """
        Calculate continuum Luminoisity at given waves
        """
        waves = np.array(waves)
        L = np.full(len(waves), -1.0)  # to save the luminosity results
        valid_idx = np.where((waves < np.max(wave)) & (waves > np.min(wave)), True, False)
        conti_flux = self.PL(waves[valid_idx], pp) + self.F_poly_conti(waves[valid_idx], pp[11:])
        Llam = waves[valid_idx] * self.flux2L(conti_flux, self.z)
        Llam[Llam <= 0] = 1e-1  # to make the log of these invalid values to be -1.
        L[valid_idx] = np.log10(Llam)

        return L

    def _residuals(self, p, xval, yval, weight, _conti_model):
        """Continual residual function used in lmfit"""
        pp = list(p.valuesdict().values())
        return (yval - _conti_model(xval, pp)) / weight

    def fit_lines(self, wave, line_flux, err, f):
        """Fit the emission lines with Gaussian profiles """

        # Remove abosorbtion lines in emission line region, pixels below continuum
        ind_neg_line = ~np.where(((((wave > 2700.) & (wave < 2900.)) | ((wave > 1700.) & (wave < 1970.)) |
                                   ((wave > 1500.) & (wave < 1700.)) | ((wave > 1290.) & (wave < 1450.)) |
                                   ((wave > 1150.) & (wave < 1290.))) & (line_flux < -err)), True, False)

        # Read line parameter file
        linelist = read_line_params(os.path.join(self.path, self.param_file_name))
        self.linelist = linelist

        # Ensure the spectrum covers the rest-frame wavelengths of the line complexes
        ind_kind_line = np.where((linelist['lambda'] > wave.min()) & (linelist['lambda'] < wave.max()), True, False)

        # Initialize some lists
        line_result = []
        line_result_name = []
        line_result_type = []
        comp_result = []
        comp_result_type = []
        comp_result_name = []
        gauss_result = []
        gauss_result_all = []
        gauss_result_type = []
        gauss_result_name = []
        fur_result = []
        fur_result_type = []
        fur_result_name = []
        all_comp_range = []
        self.f_line_model = np.zeros_like(wave)

        if ind_kind_line.any() == True:

            # Sort complex name with line wavelength
            uniq_linecomp, uniq_ind = np.unique(linelist['compname'][ind_kind_line], return_index=True)
            uniq_linecomp_sort = uniq_linecomp[linelist['lambda'][ind_kind_line][uniq_ind].argsort()]
            ncomp = len(uniq_linecomp_sort)
            compname = linelist['compname']
            allcompcenter = np.sort(linelist['lambda'][ind_kind_line][uniq_ind])

            """
            Setup parameters for fitting
            
            Loop through each line complex and set the parameter limits and initial conditions
            """

            # Number of emission lines complexes loop
            for ii in range(ncomp):

                # Get the number of emission lines in the complex
                compcenter = allcompcenter[ii]
                ind_line = np.where(linelist['compname'] == uniq_linecomp_sort[ii], True, False)  # get line index
                nline_fit = np.sum(ind_line)  # n line in one complex
                linelist_fit = linelist[ind_line]

                # Get the number Gaussian components that make up each line
                ngauss_fit = np.asarray(linelist_fit['ngauss'], dtype=int)

                # Restrict fitting to wavelength range covered by spectrum
                comp_range = [linelist_fit[0]['minwav'], linelist_fit[0]['maxwav']]  # read complex range from table
                all_comp_range = np.concatenate([all_comp_range, comp_range])

                # Get the pixel indicies within the line complex region and remove absorption lines in line region
                ind_n = np.where((wave > comp_range[0]) & (wave < comp_range[1]) & (ind_neg_line == True), True, False)

                # Ensure there are at least 10 pixels in the data
                # TODO: Better compare the number of pixels and the number of free parameters.
                if np.sum(ind_n) > 3 * np.sum(ngauss_fit):

                    fit_params = Parameters()
                    ln_lambda_0s = []

                    # Number of emission lines within line complex loop
                    for n in range(nline_fit):

                        if ngauss_fit[n] > 0:

                            line_name = linelist['linename'][ind_line][n]  # Must be unique

                            # Get parameter limits
                            ln_lambda_0 = np.log(linelist['lambda'][ind_line][n])  # ln line center
                            voff = linelist['voff'][ind_line][n]

                            # It's usually a good idea to jitter the parameters a bit
                            # sigma
                            sig_0 = linelist['inisig'][ind_line][n] + np.abs(np.random.normal(0, self.epsilon_jitter))
                            sig_low = linelist['minsig'][ind_line][n]
                            sig_up = linelist['maxsig'][ind_line][n]

                            # scale
                            scale_0 = linelist['inisca'][ind_line][n] + np.abs(np.random.normal(0, self.epsilon_jitter))
                            scale_low = linelist['minsca'][ind_line][n]
                            scale_up = linelist['maxsca'][ind_line][n]

                            vary = bool(linelist['vary'][ind_line][n])

                            # change in wav relative to complex center
                            dwave_0 = np.random.normal(0, self.epsilon_jitter)

                            # Number of Gaussians loop
                            for nn in range(ngauss_fit[n]):
                                fit_params.add(f'{line_name}_{nn + 1}_scale', value=scale_0, min=scale_low,
                                               max=scale_up, vary=vary)
                                fit_params.add(f'{line_name}_{nn + 1}_dwave', value=dwave_0, min=-voff, max=voff,
                                               vary=vary)
                                ln_lambda_0s.append(ln_lambda_0)
                                fit_params.add(f'{line_name}_{nn + 1}_sigma', value=sig_0, min=sig_low, max=sig_up,
                                               vary=vary)

                    """
                    Tie lines

                    Tie line properties together during the fitting using the "expr" keyword in lmfit
                    
                    Right now only lines in the same line complex can be tied together
                    
                    We will always tie to the 1st parameter with the same index
                    and skip tiing the first parameter itself, which is redundant and gives a recusion error
                    """

                    # Number of emission lines within line complex loop
                    for n in range(nline_fit):

                        line_name = linelist['linename'][ind_line][n]  # Must be unique

                        # Tie velocity
                        vindex = linelist['vindex'][ind_line][n]

                        if vindex > 0:
                            # Find all the line_names within the complex that have the same tie index
                            mask_index = linelist['vindex'][ind_line] == vindex
                            line_name_mask = linelist['linename'][ind_line][mask_index][0]
                            # Generate constraint expression
                            expr = f'{line_name_mask}_1_dwave'

                            # Number of Gaussians loop
                            for nn in range(ngauss_fit[n]):

                                # Don't assign expr constraint to the first line, which has already been constrained
                                if f'{line_name}_{nn + 1}_dwave' != expr:
                                    fit_params[f'{line_name}_{nn + 1}_dwave'].expr = expr

                        # Tie width
                        windex = linelist['windex'][ind_line][n]

                        if windex > 0:
                            # Find all the line_names within the complex that have the same tie index
                            mask_index = linelist['windex'][ind_line] == windex
                            line_name_mask = linelist['linename'][ind_line][mask_index][0]
                            # Generate constraint expression
                            expr = f'{line_name_mask}_1_sigma'

                            # Number of Gaussians loop
                            for nn in range(ngauss_fit[n]):

                                # Don't assign expr constraint to the first line, which has already been constrained
                                if f'{line_name}_{nn + 1}_sigma' != expr:
                                    fit_params[f'{line_name}_{nn + 1}_sigma'].expr = expr

                        # Tie flux ratios
                        findex = linelist['findex'][ind_line][n]
                        fvalue = linelist['fvalue'][ind_line][n]

                        if findex > 0:
                            # Find all the line_names within the complex that have the same tie index
                            mask_index = linelist['findex'][ind_line] == findex
                            # Tie to first line
                            line_name_mask = linelist['linename'][ind_line][mask_index][0]

                            # Generate constraint expression
                            expr_base = f'{line_name_mask}_1_scale'

                            fvalue_tie = linelist['fvalue'][ind_line][mask_index][0]
                            fratio = fvalue / fvalue_tie  # All masked fvalues should be the same
                            expr = f'{fratio} * {expr_base}'

                            # Number of Gaussians loop
                            for nn in range(ngauss_fit[n]):

                                # Don't assign expr constraint to the first line, which has already been constrained
                                if (f'{line_name}_{nn + 1}_scale' != expr_base):
                                    fit_params[f'{line_name}_{nn + 1}_scale'].expr = expr

                    """
                    Perform the MLE fitting while optionally iteratively masking absorption pixels
                    """

                    # Print parameters to be fit
                    if self.verbose:
                        fit_params.pretty_print()
                        print(fr'Fitting complex {linelist["compname"][ind_line][0]}')

                    # Check max absorption iterations

                    # Firstly, we run the fit once to give the initial fitting results

                    ind_line_abs = np.full(len(self.wave), True)
                    args = (np.log(self.wave[ind_n & ind_line_abs]),
                            line_flux[ind_n & ind_line_abs],
                            self.err[ind_n & ind_line_abs],
                            ln_lambda_0s)
                    line_fit = minimize(self._residual_line, fit_params, args=args,
                                        calc_covar=False, xtol=self.xtol_line, ftol=self.ftol_line)

                    # Only if when the self.rej_abs_line is True, we let the code go into the iteration
                    if self.rej_abs_line == True:
                        redchi = line_fit.redchi
                        for n_iter in range(self.rej_abs_line_max_niter):
                            resid_full = np.zeros_like(self.wave)
                            resid_full[ind_n & ind_line_abs] = line_fit.residual
                            ind_line_abs_tmp = ind_line_abs & np.where(resid_full < -3, False, True) & np.where(
                                resid_full > 3, False, True)

                            # Check if number of valid pixels minus 10 is not larger than the number of fitted gaussian parameters
                            if len(self.wave[ind_n & ind_line_abs_tmp]) - 10 < len(fit_params):
                                break

                            # Fit wavelength in ln space
                            args = (np.log(self.wave[ind_n & ind_line_abs_tmp]),
                                    line_flux[ind_n & ind_line_abs_tmp],
                                    self.err[ind_n & ind_line_abs_tmp],
                                    ln_lambda_0s)
                            line_fit_tmp = minimize(self._residual_line, fit_params, args=args,
                                                    calc_covar=False, xtol=self.xtol_line, ftol=self.ftol_line)

                            # Check if the reduced chi squared has not improved
                            if line_fit_tmp.redchi >= redchi:
                                break
                            else:
                                # Accept the fit
                                redchi = line_fit_tmp.redchi
                                ind_line_abs = ind_line_abs_tmp
                                line_fit = line_fit_tmp

                    params_dict = line_fit.params.valuesdict()
                    par_names = list(params_dict.keys())
                    params = list(params_dict.values())
                    chisqr = line_fit.chisqr
                    bic = line_fit.bic
                    redchi = line_fit.redchi

                    # Print fit report
                    if self.verbose:
                        print('Fit report')
                        report_fit(line_fit.params)

                    """
                    Uncertainty estimation
                    """
                    if (self.MCMC == True or self.MC == True) and self.nsamp > 0:

                        """
                        MCMC sampling
                        """
                        if (self.MCMC == True) and (self.MC == False):
                            # Sample with MCMC, using the initial minima
                            args = (np.log(self.wave[ind_n & ind_line_abs]),
                                    line_flux[ind_n & ind_line_abs],
                                    self.err[ind_n & ind_line_abs],
                                    ln_lambda_0s)
                            line_samples = minimize(self._residual_line, params=line_fit.params, args=args,
                                                    method='emcee', nan_policy='omit', burn=self.nburn,
                                                    steps=self.nsamp, thin=self.nthin,
                                                    is_weighted=True, **self.kwargs_line_emcee)
                            p = line_samples.params.valuesdict()
                            df_samples = line_samples.flatchain
                            samples = df_samples.to_numpy()

                            # Print fit report
                            if self.verbose:
                                print(
                                    f'acceptance fraction = {np.mean(line_samples.acceptance_fraction)} +/- {np.std(line_samples.acceptance_fraction)}')
                                # As a rule of thumb the value should be between 0.2 and 0.5
                                print('median of posterior probability distribution')
                                print('--------------------------------------------')
                                report_fit(line_samples.params)

                            if self.plot_corner:
                                import corner
                                truths = [params_dict[k] for k in df_samples.columns.values.tolist()]
                                emcee_plot = corner.corner(df_samples, labels=line_samples.var_names,
                                                           quantiles=[0.16, 0.5, 0.84], truths=truths)

                            # Loop through each parameter
                            for k, name in enumerate(par_names):
                                # Add a column with initial value if the parameter is fixed
                                if name not in df_samples.columns.values.tolist():
                                    df_samples[name] = params_dict[name]

                            # Sort the samples dataframe back to its original order
                            df_samples = df_samples[par_names]
                            samples = df_samples.to_numpy()

                        """
                        MC resampling
                        """
                        if (self.MCMC == False) and (self.MC == True):
                            # Resample the spectrum using the measurement error using the best-fit parameters as initial conditions
                            samples = np.zeros((self.nsamp, len(params)))

                            for k in range(self.nsamp):
                                # np.random.seed(k)

                                line_flux_resampled = line_flux + np.random.randn(len(line_flux)) * self.err

                                # Use fit_params or line_fit.params?
                                args = (np.log(self.wave[ind_n & ind_line_abs]),
                                        line_flux_resampled[ind_n & ind_line_abs],
                                        self.err[ind_n & ind_line_abs],
                                        ln_lambda_0s)
                                line_samples = minimize(self._residual_line, fit_params, args=args, calc_covar=False,
                                                        xtol=self.xtol_line, ftol=self.ftol_line)
                                params_dict = line_samples.params.valuesdict()
                                params_mc = list(params_dict.values())
                                samples[k] = params_mc

                        if (self.MCMC == True) and (self.MC == True):
                            RuntimeError('MCMC and MC modes cannot both be True')

                        # Error estimates
                        params_err = get_err(samples)

                    """
                    Save the results
                    """

                    # Reshape parameters array for vectorization
                    ngauss = len(params) // 3
                    params_shaped = np.reshape(params, (ngauss, 3))
                    params_shaped[:, 1] += ln_lambda_0s  # Transform ln lambda ~ d ln lambda + ln lambda0
                    params = params_shaped.reshape(-1)

                    # Get line fitting results
                    comp_name = linelist['compname'][ind_line][0]
                    line_status = int(line_fit.success)
                    comp_result.append([comp_name, line_status, chisqr, bic, redchi, line_fit.nfev, line_fit.nfree])
                    comp_result_type.append(['str', 'int', 'float', 'float', 'float', 'int', 'int'])
                    comp_result_name.append(
                        [str(ii + 1) + '_complex_name', str(ii + 1) + '_line_status', str(ii + 1) + '_line_min_chi2',
                         str(ii + 1) + '_line_bic', str(ii + 1) + '_line_red_chi2', str(ii + 1) + '_niter',
                         str(ii + 1) + '_ndof'])

                    # Line properties
                    br_name = uniq_linecomp_sort[ii]

                    # XXX temporary solution for complex line measurement
                    # br_line_name = linelist['linename'][ind_line][0]

                    # Gauss result
                    if (self.MCMC == True or self.MC == True) and self.nsamp > 0:
                        # Reshape samples array for vectorization
                        samples_shaped = np.reshape(samples, (np.shape(samples)[0], ngauss, 3))
                        samples_shaped[:, :, 1] += ln_lambda_0s  # Transform ln lambda ~ d ln lambda + ln lambda0
                        samples = samples_shaped.reshape(np.shape(samples))

                        # Parameter uncertainties
                        params_err = get_err(samples)
                        gauss_result.append(list(chain.from_iterable(zip(params, params_err))))
                        gauss_result_all.append(samples)

                        # Physical results
                        fur_result_temp = np.zeros([6, np.shape(samples)[0]])
                        for k, s in enumerate(samples):
                            # TODO: Exclude other lines, like OIII, HeII
                            fur_result_temp[:, k] = self.line_prop(compcenter, s, 'broad')
                            # XXX temporary solution
                            # fur_result_temp[:, k] = self.line_prop_from_name(br_line_name, 'broad')

                        fur_result_std = get_err(fur_result_temp, axis=1)

                        for n in range(nline_fit):
                            for nn in range(int(ngauss_fit[n])):
                                line_name = linelist['linename'][ind_line][n] + '_' + str(nn + 1)
                                gauss_result_type.append(['float'] * 6)
                                gauss_result_name.append(
                                    [line_name + '_scale', line_name + '_scale_err', line_name + '_centerwave',
                                     line_name + '_centerwave_err', line_name + '_sigma', line_name + '_sigma_err'])

                        # Line properties
                        fur_result.append(list(chain.from_iterable(
                            zip(self.line_prop(compcenter, params, 'broad'), fur_result_std))))
                        # XXX temporary solution
                        # fur_result.append(list(chain.from_iterable(
                        #     zip(self.line_prop_from_name(compcenter, 'broad'), fur_result_std))))
                        fur_result_type.append(['float'] * 12)
                        fur_result_name.append([br_name + '_whole_br_fwhm', br_name + '_whole_br_fwhm_err',
                                                br_name + '_whole_br_sigma',
                                                br_name + '_whole_br_sigma_err', br_name + '_whole_br_ew',
                                                br_name + '_whole_br_ew_err',
                                                br_name + '_whole_br_peak', br_name + '_whole_br_peak_err',
                                                br_name + '_whole_br_area',
                                                br_name + '_whole_br_area_err', br_name + '_whole_br_snr',
                                                br_name + '_whole_br_snr_err'])
                    else:
                        # Gauss results
                        gauss_result.append(params)

                        for n in range(nline_fit):
                            for nn in range(int(ngauss_fit[n])):
                                line_name = linelist['linename'][ind_line][n] + '_' + str(nn + 1)
                                gauss_result_type.append(['float'] * 3)
                                gauss_result_name.append(
                                    [line_name + '_scale', line_name + '_centerwave', line_name + '_sigma'])

                        # Line properties
                        fur_result.append(self.line_prop(compcenter, params, 'broad'))
                        # XXX temporary solution
                        # fur_result.append(self.line_prop_from_name(compcenter, 'broad'))
                        fur_result_type.append(['float'] * 6)
                        fur_result_name.append([br_name + '_whole_br_fwhm', br_name + '_whole_br_sigma',
                                                br_name + '_whole_br_ew',
                                                br_name + '_whole_br_peak', br_name + '_whole_br_area',
                                                br_name + '_whole_br_snr'])
                else:
                    if self.verbose:
                        print("Less than 10 pixels in line fitting!")

            # Flatten arrays
            if len(comp_result) > 0:
                comp_result = np.concatenate(comp_result)
                comp_result_type = np.concatenate(comp_result_type)
                comp_result_name = np.concatenate(comp_result_name)

                gauss_result = np.concatenate(gauss_result)
                if (self.MCMC == True or self.MC == True) and self.nsamp > 0:
                    gauss_result_all = np.concatenate(gauss_result_all, axis=1)
                gauss_result_type = np.concatenate(gauss_result_type)
                gauss_result_name = np.concatenate(gauss_result_name)

                fur_result = np.concatenate(fur_result)
                fur_result_type = np.concatenate(fur_result_type)
                fur_result_name = np.concatenate(fur_result_name)

            # Add results to line_result
            line_result = np.concatenate([comp_result, gauss_result, fur_result])
            line_result_type = np.concatenate([comp_result_type, gauss_result_type, fur_result_type])
            line_result_name = np.concatenate([comp_result_name, gauss_result_name, fur_result_name])

            # Save the line model flux 
            if (self.MCMC == True or self.MC == True) and self.nsamp > 0:
                # For each Gaussian line component
                for p in range(len(gauss_result) // (2 * 3)):
                    # Evaluate the line component
                    gauss_result_p = gauss_result[p * 3 * 2:(p + 1) * 3 * 2:2]
                    self.f_line_model += self.Onegauss(np.log(wave), gauss_result_p)
            else:
                # For each Gaussian line component
                for p in range(len(gauss_result) // 3):
                    # Evaluate the line component
                    gauss_result_p = gauss_result[p * 3:(p + 1) * 3:1]
                    self.f_line_model += self.Onegauss(np.log(wave), gauss_result_p)

        else:
            ncomp = 0
            uniq_linecomp_sort = np.array([])
            if self.verbose:
                print("No line to fit! Please set line_fit to FALSE or enlarge wave_range!")

        # Save properties
        self.comp_result = np.array(comp_result)

        self.gauss_result = np.array(gauss_result)
        self.gauss_result_all = np.array(gauss_result_all)
        self.gauss_result_name = np.array(gauss_result_name)

        self.fur_result = np.array(fur_result)
        self.fur_result_type = np.array(fur_result_type)
        self.fur_result_name = np.array(fur_result_name)

        self.line_result = np.array(line_result)
        self.line_result_type = np.array(line_result_type)
        self.line_result_name = np.array(line_result_name)

        self.ncomp = ncomp
        self.line_flux = line_flux
        self.all_comp_range = np.array(all_comp_range)
        self.uniq_linecomp_sort = uniq_linecomp_sort

        return self.line_result, self.line_result_name

    def line_prop_from_name(self, line_name, line_type='broad', ln_sigma_br=0.0017):
        """
        line_name: line name e.g., 'Ha_br'
        """

        # Get the complex center wavelength of the line_name component
        mask_name = self.linelist['linename'] == line_name

        # Check if no line exists
        if np.count_nonzero(mask_name) == 0:
            return 0, 0, 0, 0, 0, 0

        # Get each Gaussian component
        compcenter = self.linelist[mask_name]['lambda'][0]
        ngauss = int(self.linelist[mask_name]['ngauss'][0])
        pp_shaped = np.zeros((ngauss, 3))

        # Check if no component is fit
        mask_result_name = self.line_result_name == f'{line_name}_{1}_scale'
        if np.count_nonzero(mask_result_name) == 0:
            return 0, 0, 0, 0, 0, 0

        # Number of Gaussian components loop
        for n in range(ngauss):
            # Get the Gaussian properties
            pp_shaped[n, 0] = float(self.line_result[self.line_result_name == f'{line_name}_{n + 1}_scale'][0])
            pp_shaped[n, 1] = float(self.line_result[self.line_result_name == f'{line_name}_{n + 1}_centerwave'][0])
            pp_shaped[n, 2] = float(self.line_result[self.line_result_name == f'{line_name}_{n + 1}_sigma'][0])

        # Flatten
        pp = pp_shaped.reshape(-1)

        return self.line_prop(compcenter, pp, line_type, ln_sigma_br)

    def line_prop(self, compcenter, pp, linetype='broad', ln_sigma_br=0.0017):
        """
        Calculate the further results for the broad component in emission lines, e.g., FWHM, sigma, peak, line flux
        The compcenter is the theortical vacuum wavelength for the broad compoenet.
        compcenter:
        pp:
        linetype: 'broad' or 'narrow'
        ln_sigma_br: line sigma separating broad and narrow lines (AA??)
        ln_sigma_max: Max sigma to consider in the calculation (used to exclude ultra-broad wings, etc.)
        """
        pp = np.array(pp).astype(float)

        # # XXX Very temporary change to measure only the component I need!!! Would be completely wrong in other projects!
        # if len(pp) > 9:
        #     pp = pp[:9]
        # ind_br = np.ones_like(pp, dtype='bool')

        if linetype.lower() == 'broad':
            mask_br = (pp[2::3] > ln_sigma_br) & (pp[2::3] > 0)
            ind_br = np.repeat(np.where(mask_br, True, False), 3)

        elif linetype.lower() == 'narrow':
            mask_br = (pp[2::3] <= ln_sigma_br) & (pp[2::3] > 0)
            ind_br = np.repeat(np.where(mask_br, True, False), 3)

        else:
            raise RuntimeError("line type should be 'broad' or 'narrow'!")

        # TODO: Exclude broad lines that are not used for BH masses, etc.

        # If you want to exclude certain lines like OIII or HeII, you should use line_prop_from_name
        # and take out those line names. We should assume nothing about the order of the lines
        pp_br = pp[ind_br]

        c = const.c.to(u.km / u.s).value  # km/s
        ngauss = len(pp_br) // 3

        pp_shaped = pp.reshape([len(pp) // 3, 3])
        pp_br_shaped = pp_br.reshape([ngauss, 3])

        if ngauss == 0:
            fwhm, sigma, ew, peak, area, snr = 0, 0, 0, 0, 0, 0
        else:
            cen = pp_br_shaped[:, 1]
            sig = pp_br_shaped[:, 2]

            # print cen,sig,area
            left = np.min(cen - 3 * sig)
            right = np.max(cen + 3 * sig)
            disp = 1e-4 * np.log(10)
            npix = int((right - left) / disp)

            xx = np.linspace(left, right, npix)
            yy_br = self._Manygauss(xx, pp_br_shaped)

            ## plt.plot(np.exp(xx), yy_br) #####

            # Use the continuum model to avoid the inf bug of EW when the spectrum range passed in is too short
            contiflux = self.PL(np.exp(xx), self.conti_params) + self.F_poly_conti(
                np.exp(xx), self.conti_params[11:]) + self.Balmer_conti(np.exp(xx), self.conti_params[8:11])

            # Find the line peak location
            ypeak = yy_br.max()
            ypeak_ind = np.argmax(yy_br)
            peak = np.exp(xx[ypeak_ind])

            # Compute the S/N = peak / sigma_MAD(residual)
            mask_complex = (self.wave > compcenter - 400) & (self.wave < compcenter + 400)
            residual = self.line_flux - self._Manygauss(np.log(self.wave), pp_shaped)
            noise = median_abs_deviation(residual[mask_complex], scale='normal')  # MAD of noise
            snr = ypeak / noise

            # Find the FWHM in km/s
            spline = interpolate.UnivariateSpline(xx, yy_br - np.max(yy_br) / 2, s=0)

            if len(spline.roots()) > 0:
                fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()
                fwhm = abs(np.exp(fwhm_left) - np.exp(fwhm_right)) / compcenter * c

                # Calculate the line sigma and EW in normal wavelength
                line_flux = self._Manygauss(xx, pp_br_shaped)
                line_wave = np.exp(xx)
                lambda0 = integrate.trapz(line_flux, line_wave)  # calculate the total broad line flux
                lambda1 = integrate.trapz(line_flux * line_wave, line_wave)
                lambda2 = integrate.trapz(line_flux * line_wave * line_wave, line_wave)
                ew = integrate.trapz(np.abs(line_flux / contiflux), line_wave)
                area = lambda0

                sigma = np.sqrt(lambda2 / lambda0 - (lambda1 / lambda0) ** 2) / compcenter * c
            else:
                fwhm, sigma, ew, peak, area, snr = 0, 0, 0, 0, 0, 0

        return fwhm, sigma, ew, peak, area, snr

    def _residual_line(self, params, xval, yval, weight, ln_lambda_0s):
        """
        Calculate total residual for fitting of line complexes
        """

        pp = list(params.valuesdict().values())

        # Reshape parameters array for vectorization
        ngauss = len(pp) // 3
        pp_shaped = np.reshape(pp, (ngauss, 3))
        pp_shaped[:, 1] += ln_lambda_0s  # Transform ln lambda ~ d ln lambda + ln lambda0

        resid = (yval - self._Manygauss(xval, pp_shaped)) / weight

        return resid

    def save_result(self, conti_result, conti_result_type, conti_result_name, line_result, line_result_type,
                    line_result_name, save_fits_path, save_fits_name):
        """Save all data to fits"""
        self.all_result = np.concatenate([conti_result, line_result])
        self.all_result_type = np.concatenate([conti_result_type, line_result_type])
        self.all_result_name = np.concatenate([conti_result_name, line_result_name])

        t = Table(self.all_result, names=(self.all_result_name), dtype=self.all_result_type)
        t.write(os.path.join(save_fits_path, save_fits_name + '.fits'), format='fits', overwrite=True)
        return

    def set_mpl_style(fsize=15, tsize=18, tdir='in', major=5.0, minor=3.0, lwidth=1.8, lhandle=2.0):

        """Function to set MPL style"""

        plt.style.use('default')
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = fsize
        plt.rcParams['legend.fontsize'] = tsize
        plt.rcParams['xtick.direction'] = tdir
        plt.rcParams['ytick.direction'] = tdir
        plt.rcParams['xtick.major.size'] = major
        plt.rcParams['xtick.minor.size'] = minor
        plt.rcParams['ytick.major.size'] = 5.0
        plt.rcParams['ytick.minor.size'] = 3.0
        plt.rcParams['axes.linewidth'] = lwidth
        plt.rcParams['legend.handlelength'] = lhandle

        return

    def plot_fig(self, save_fig_path='.', broad_fwhm=1200, plot_line_name=True, plot_legend=True, ylims=None,
                 plot_residual=True, show_title=True, plot_br_prop=False):
        """Plot the results
        
        broad_fwhm: float, optional
            Definition for width of the broad lines. Default: 1200 km/s (careful, is not the exact separation used in line_prop)
        
        plot_line_name: bool, optional
            if True, serval main emission lines will be plotted in the first panel of the output figure. Default: False
            
            
        TODO: Consider splitting up into plot_conti and plot_complex functions
        Wenke: I totally agree with that!!!
        to encourage flexibility/reuse
        
        """

        def _pretty_name(plain_name):
            special_list = {'Ha' : r'H\alpha',
                            'Hb' : r'H\beta',
                            'Hg' : r'H\gamma',
                            'Hd' : r'H\delta',
                            'Hep': r'H\epsilon',
                            'Lya': r'Ly\alpha'}
            if plain_name in special_list.keys():
                format_name = special_list[plain_name]
            elif 'I' in plain_name:
                insert_idx = plain_name.find('I')
                format_name = plain_name[:insert_idx] + '\,' + plain_name[insert_idx:]
            else:
                format_name = plain_name
            return plain_name #rf'$\mathrm{{{format_name}}}$'

        pp = list(self.conti_fit.params.valuesdict().values())

        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)

        wave_eval = np.linspace(np.min(self.wave) - 200, np.max(self.wave) + 200, 5000)
        f_conti_model_eval = self.PL(wave_eval, pp) + self.Fe_flux_mgii(wave_eval, pp[0:3]) + self.Fe_flux_balmer(
            wave_eval, pp[3:6]) + self.F_poly_conti(wave_eval, pp[11:]) + self.Balmer_conti(wave_eval, pp[8:11])

        # Plot lines
        if (self.linefit == True) & (len(self.line_result) > 0):
            # If errors are in the results
            if (self.MCMC == True or self.MC == True) and self.nsamp > 0:
                mc_flag = 2
            else:
                mc_flag = 1

            # Number of line complexes actually fitted
            ncomp_fit = len(self.fur_result) // (
                        mc_flag * 6)  # TODO: Not 5 here. But better not use absolute value to fully fix this bug
            
            # Prepare for the emission line subplots in the second row
            fig, axn = plt.subplots(nrows=2, ncols=np.max([ncomp_fit, 1]), figsize=(15, 8), squeeze=False)
            gs = axn[0, 0].get_gridspec()
            for axi in axn[0, :]:
                axi.remove()
            ax = fig.add_subplot(gs[0, :])
                
            self.f_line_narrow_model = np.zeros_like(self.wave)
            self.f_line_br_model = np.zeros_like(self.wave)
            lines_total = np.zeros_like(wave_eval)
            line_order = {'r': 3, 'g': 7}  # Ensure narrow lines plot above the broad lines

            # For each Gaussian line component
            for p in range(len(self.gauss_result) // (mc_flag * 3)):
                gauss_result_p = self.gauss_result[p * 3 * mc_flag:(p + 1) * 3 * mc_flag:mc_flag]

                # Broad or narrow line check
                if self.CalFWHM(self.gauss_result[(2 + p * 3) * mc_flag]) < broad_fwhm:
                    # Narrow
                    color = 'g'
                    self.f_line_narrow_model += self.Onegauss(np.log(self.wave), gauss_result_p)
                else:
                    # Broad
                    color = 'r'
                    self.f_line_br_model += self.Onegauss(np.log(self.wave), gauss_result_p)

                # Evaluate the line component
                line_single = self.Onegauss(np.log(wave_eval), gauss_result_p)
                # self.f_line_model += self.Onegauss(np.log(wave), gauss_result_p)

                # Plot the line component
                ax.plot(wave_eval, line_single + f_conti_model_eval, color=color, zorder=5)
                for c in range(ncomp_fit):
                    axn[1][c].plot(wave_eval, line_single, color=color, zorder=line_order[color])

                lines_total += line_single

            # Supplement the emission lines in the first subplot
            ax.plot(wave_eval, lines_total + f_conti_model_eval, 'b', label='line', zorder=6)

            # Line complex subplots
            for c in range(ncomp_fit):
                axn[1][c].plot(wave_eval, lines_total, color='b', zorder=10)

                # Set axis limits
                axn[1][c].set_xlim(self.all_comp_range[2 * c:2 * c + 2])

                mask_complex = np.where(
                    (self.wave > self.all_comp_range[2 * c]) & (self.wave < self.all_comp_range[2 * c + 1]), True,
                    False)

                # Mask outliers
                r = self.line_flux - self.f_line_model
                mad = median_abs_deviation(r[mask_complex], scale='normal')  # MAD of noise
                mask_outliers = np.where(r < 3 * mad, True, False)
                f_max = np.max(self.line_flux[mask_complex & mask_outliers])
                # f_min = np.min([-1, np.min(self.line_flux[mask_complex & mask_outliers])])
                f_min = np.min([-1, -3 * mad])

                if ylims is None:
                    axn[1][c].set_ylim(f_min * 0.9, f_max * 1.1)
                else:
                    axn[1][c].set_ylim(ylims[0], ylims[1])

                axn[1][c].set_xticks([self.all_comp_range[2 * c],
                                      np.round((self.all_comp_range[2 * c] + self.all_comp_range[2 * c + 1]) / 2, -1),
                                      self.all_comp_range[2 * c + 1]])

                axn[1][c].text(0.02, 0.9, _pretty_name(self.uniq_linecomp_sort[c]), fontsize=20,
                               transform=axn[1][c].transAxes)
                axn[1][c].text(0.02, 0.825, r'$\chi ^2_\nu=$' + str(np.round(float(self.comp_result[c * 7 + 4]), 2)),
                               fontsize=12, transform=axn[1][c].transAxes)

                # Broad line properties
                if plot_br_prop == True:
                    # FWHM and luminosity
                    fwhm = self.fur_result[self.fur_result_name == f'{self.uniq_linecomp_sort[c]}_whole_br_fwhm'][0]
                    area = self.fur_result[self.fur_result_name == f'{self.uniq_linecomp_sort[c]}_whole_br_area'][0]
                    snr = self.fur_result[self.fur_result_name == f'{self.uniq_linecomp_sort[c]}_whole_br_snr'][0]

                    if mc_flag == 2:
                        fwhm_err = \
                        self.fur_result[self.fur_result_name == f'{self.uniq_linecomp_sort[c]}_whole_br_fwhm_err'][0]
                        area_err = \
                        self.fur_result[self.fur_result_name == f'{self.uniq_linecomp_sort[c]}_whole_br_area_err'][0]

                        axn[1][c].text(0.02, 0.75,
                                       fr'$L_{{\rm{{br}}}}=10^{{{{{np.round(np.log10(self.flux2L(area)), 2)}}}\pm{{{np.round(0.434 * self.flux2L(area_err) / self.flux2L(area), 2)}}}}}$' + r'$\ \rm{erg}\ \rm{s}^{-1}$',
                                       fontsize=12, transform=axn[1][c].transAxes)
                        axn[1][c].text(0.02, 0.675,
                                       fr'${{\rm{{FWHM}}}}_{{\rm{{br}}}}={{{int(np.round(fwhm, 0))}}}\pm{{{int(np.round(fwhm_err, 0))}}}$' + r'$\ \rm{km}\ \rm{s}^{-1}$',
                                       fontsize=12, transform=axn[1][c].transAxes)

                    else:
                        axn[1][c].text(0.02, 0.75,
                                       fr'$L_{{\rm{{br}}}}=10^{{{np.round(np.log10(self.flux2L(area)), 1)}}}$' + r'$\ \rm{erg}\ \rm{s}^{-1}$',
                                       fontsize=12, transform=axn[1][c].transAxes)
                        axn[1][c].text(0.02, 0.675,
                                       fr'${{\rm{{FWHM}}}}_{{\rm{{br}}}}={{{int(np.round(fwhm, 0))}}}$' + r'$\ \rm{km}\ \rm{s}^{-1}$',
                                       fontsize=12, transform=axn[1][c].transAxes)

                    axn[1][c].text(0.02, 0.6, fr'$S/N_{{\rm{{br}}}}={np.round(snr, 1)}$',
                                   fontsize=12, transform=axn[1][c].transAxes)

                # Wave mask
                if self.wave_mask is not None:

                    for j, w in enumerate(self.wave_mask):

                        axn[1][c].axvspan(w[0], w[1], color='k', alpha=0.25)
                        # Plot avoiding drawing lines between masked values
                        label_data = None
                        label_resid = None

                        if j == 0:
                            mask = self.wave < w[0]
                            label_data = 'data'
                            label_resid = 'resid'
                            axn[1][c].plot(self.wave[mask], self.line_flux[mask], 'k', label=label_data, lw=1, zorder=2)
                        if j == len(self.wave_mask) - 1:
                            mask = self.wave_prereduced > w[1]
                            axn[1][c].plot(self.wave[mask], self.line_flux[mask], 'k', label=label_data, lw=1, zorder=2)
                        else:
                            mask = (self.wave > w[1]) & (self.wave < self.wave_mask[j + 1, 0])
                            axn[1][c].plot(self.wave[mask], self.line_flux[mask], 'k', label=label_data, lw=1, zorder=2)

                        # Residual
                        if plot_residual:
                            axn[1][c].axhline(-5, color='k', zorder=0, lw=0.5)
                            axn[1][c].plot(self.wave[mask], self.line_flux[mask] - self.f_line_model[mask] - 5, 'gray',
                                           label=label_resid, linestyle='dotted', lw=1, zorder=3)
                else:

                    axn[1][c].plot(self.wave, self.line_flux, 'k', label='data', lw=1, zorder=2)

                    # Residual
                    if plot_residual:
                        axn[1][c].axhline(-5, color='k', zorder=0, lw=0.5)
                        axn[1][c].plot(self.wave, self.line_flux - self.f_line_model - 5, 'gray',
                                       label='resid', linestyle='dotted', lw=1, zorder=3)
        else:
            # If no lines are fitted, there would be only one row
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

        # End line complex subplots

        # Main figure
        if self.wave_mask is not None:

            for j, w in enumerate(self.wave_mask):

                ax.axvspan(w[0], w[1], color='k', alpha=0.25)

                # Plot avoiding drawing lines between masked values
                if j == 0:
                    mask = self.wave_prereduced < w[0]
                    ax.plot(self.wave_prereduced[mask], self.flux_prereduced[mask], 'k', label='data', lw=1, zorder=2)
                    # ax.plot(self.wave_prereduced[mask], self.err_prereduced[mask], 'gray', label='error', lw=1, zorder=1)
                if j == len(self.wave_mask) - 1:
                    mask = self.wave_prereduced > w[1]
                    ax.plot(self.wave_prereduced[mask], self.flux_prereduced[mask], 'k', lw=1, zorder=2)
                    # ax.plot(self.wave_prereduced[mask], self.err_prereduced[mask], 'gray', lw=1, zorder=1)
                else:
                    mask = (self.wave_prereduced > w[1]) & (self.wave_prereduced < self.wave_mask[j + 1, 0])
                    ax.plot(self.wave_prereduced[mask], self.flux_prereduced[mask], 'k', lw=1, zorder=2)
                    # ax.plot(self.wave_prereduced[mask], self.err_prereduced[mask], 'gray', lw=1, zorder=1)

        else:

            ax.plot(self.wave_prereduced, self.flux_prereduced, 'k', label='data', lw=1, zorder=2)

            # Residual
            if plot_residual == True:
                if self.linefit == True:
                    ax.plot(self.wave, self.line_flux - self.f_line_model, 'gray',
                            label='resid', linestyle='dotted', lw=1, zorder=3)
                else:
                    ax.plot(self.wave, self.flux - self.f_conti_model, 'gray',
                            label='resid', linestyle='dotted', lw=1, zorder=3)

        # Title
        if show_title == True:

            if self.ra == -999 or self.dec == -999:
                ax.set_title(f'{self.sdss_name}   z = {np.round(float(self.z), 4)}', fontsize=20)
            else:
                ax.set_title(
                    f'ra,dec = ({np.round(self.ra, 4)},{np.round(self.dec, 4)})   {self.sdss_name}   z = {np.round(float(self.z), 4)}',
                    fontsize=20)

        # Host decomposition
        if self.decompose_host == True and self.decomposed == True:
            ax.plot(self.wave, self.qso + self.host, 'pink', label='host+qso temp', zorder=3)
            ax.plot(self.wave, self.flux, 'grey', label='data-host', zorder=1)
            ax.plot(self.wave, self.host, 'purple', label='host', zorder=4)
        else:
            host = self.flux_prereduced.min()

        # Line legend hack
        ax.plot([0, 0], [0, 0], 'r', label='line br', zorder=5)
        ax.plot([0, 0], [0, 0], 'g', label='line na', zorder=5)

        # Continuum results
        ax.plot(wave_eval, f_conti_model_eval, 'c', lw=2, label='FeII', zorder=7)

        if self.BC == True:
            ax.plot(wave_eval,
                    self.PL(wave_eval, pp) + self.F_poly_conti(wave_eval, pp[11:]) + self.Balmer_conti(wave_eval,
                                                                                                       pp[8:11]), 'y',
                    lw=2, label='BC', zorder=8)

        ax.plot(wave_eval, self.PL(wave_eval, pp) + self.F_poly_conti(wave_eval, pp[11:]), color='orange', lw=2,
                label='conti', zorder=9)

        # Axis limits
        if self.linefit == True:
            r = self.line_flux - self.f_line_model
            mad = median_abs_deviation(r, scale='normal')  # MAD of noise
            mask_outliers = np.where(r < 3 * mad, True, False)
            plot_top = np.max(self.flux[mask_outliers])
            plot_bottom = np.min([-1, -3 * mad])  # TODO: stupid absolute lower limit here
        else:
            if self.decomposed == False:
                plot_top = self.flux.max()
                plot_bottom = self.flux.min()
            else:
                plot_top = max(self.host.max(), self.flux.max())
                plot_bottom = min(self.host.min(), self.flux.min())

        if ylims is None:
            ylims = [plot_bottom * 0.9, plot_top * 1.1]

        ax.set_ylim(ylims[0], ylims[1])
        # Plot continuum regions
        ax.scatter(self.wave[self.tmp_all], np.repeat(plot_top * 1.05, len(self.wave[self.tmp_all])), color='grey',
                   marker='o')

        if plot_legend == True:
            ax.legend(loc='best', frameon=False, ncol=2, fontsize=10)

        # Plot line names
        if plot_line_name == True:
            line_cen = np.array(
                [6564.60, 6549.85, 6585.27, 6718.29, 6732.66, 4862.68, 5008.24, 4687.02, 4341.68, 3934.78, 3728.47,
                 3426.84, 2798.75, 1908.72, 1816.97, 1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1335.30, \
                 1215.67])

            line_name = np.array(
                ['', '', r'H$\alpha$+[NII]', '', '[SII]6718,6732', r'H$\beta$', '[OIII]', 'HeII4687', r'H$\gamma$',
                 'CaII3934', '[OII]3728',
                 'NeV3426', 'MgII', 'CIII]', 'SiII1816', 'NIII]1750', 'NIV]1718', 'CIV', 'HeII1640', '', 'SiIV+OIV',
                 'CII1335', r'Ly$\alpha$'])

            # Line position
            axis_to_data = ax.transAxes + ax.transData.inverted()
            points_data = axis_to_data.transform((0, 0.92))

            for ll in range(len(line_cen)):
                if self.wave.min() < line_cen[ll] < self.wave.max():
                    ax.axvline(line_cen[ll], color='k', linestyle=':')
                    # ax.plot([line_cen[ll], line_cen[ll]], ylims, 'k:')
                    ax.text(line_cen[ll] + 7, points_data[1], line_name[ll], rotation=90, fontsize=10, va='top')

        ax.set_xlim(self.wave.min(), self.wave.max())

        # Label axes
        if self.linefit == True:
            fig.supxlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
            fig.supylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)
        else:
            fig.supxlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
            fig.supylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)
        # Save figure
        if self.save_fig == True:
            if self.verbose:
                print('Saving figure as', os.path.join(save_fig_path, self.sdss_name + '.pdf'))
            fig.savefig(os.path.join(save_fig_path, self.sdss_name + '.pdf'))
        
        plt.show()
        plt.close(fig)  # Close figure to save memory

        self.fig = fig
        return

    def CalFWHM(self, logsigma):
        """transfer the logFWHM to normal frame"""
        return 2 * np.sqrt(2 * np.log(2)) * (np.exp(logsigma) - 1) * 300000.

    def Smooth(self, y, box_pts):
        "Smooth the flux with n pixels"
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def Fe_flux_mgii(self, xval, pp):
        "Fit the UV FeII component on the continuum from 1200 to 3500 A based on Boroson & Green 1992."
        yval = np.zeros_like(xval)
        wave_Fe_mgii = 10 ** self.fe_uv[:, 0]
        flux_Fe_mgii = self.fe_uv[:, 1] * 1e15
        Fe_FWHM = pp[1]
        xval_new = xval * (1.0 + pp[2])

        ind = np.where((xval_new > 1200.) & (xval_new < 3500.), True, False)
        if np.sum(ind) > self.n_pix_min_conti:
            if Fe_FWHM < 900.0:
                sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv / 106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
            khalfsz = np.round(4 * sig_pix + 1, 0)
            xx = np.arange(0, khalfsz * 2, 1) - khalfsz
            kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
            kernel = kernel / np.sum(kernel)

            flux_Fe_conv = np.convolve(flux_Fe_mgii, kernel, 'same')
            tck = interpolate.splrep(wave_Fe_mgii, flux_Fe_conv)
            yval[ind] = pp[0] * interpolate.splev(xval_new[ind], tck)
        return yval

    def Fe_flux_balmer(self, xval, pp):
        "Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"
        yval = np.zeros_like(xval)

        wave_Fe_balmer = 10 ** self.fe_op[:, 0]
        flux_Fe_balmer = self.fe_op[:, 1] * 1e15
        ind = np.where((wave_Fe_balmer > 3686.) & (wave_Fe_balmer < 7484.), True, False)
        wave_Fe_balmer = wave_Fe_balmer[ind]
        flux_Fe_balmer = flux_Fe_balmer[ind]
        Fe_FWHM = pp[1]
        xval_new = xval * (1.0 + pp[2])
        ind = np.where((xval_new > 3686.) & (xval_new < 7484.), True, False)
        if np.sum(ind) > self.n_pix_min_conti:
            if Fe_FWHM < 900.0:
                sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv / 106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
            khalfsz = np.round(4 * sig_pix + 1, 0)
            xx = np.arange(0, khalfsz * 2, 1) - khalfsz
            kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
            kernel = kernel / np.sum(kernel)
            flux_Fe_conv = np.convolve(flux_Fe_balmer, kernel, 'same')
            tck = interpolate.splrep(wave_Fe_balmer, flux_Fe_conv)
            yval[ind] = pp[0] * interpolate.splev(xval_new[ind], tck)
        return yval

    def PL(self, xval, pp, x0=3000):
        return pp[6] * (xval / x0) ** pp[7]

    def Balmer_conti(self, xval, pp):
        """Fit the Balmer continuum from the model of Dietrich+02"""
        # xval = input wavelength, in units of A
        # pp=[norm, Te, tau_BE] -- in units of [--, K, --]
        xval = xval * u.AA
        lambda_BE = 3646.  # A
        bb_lam = BlackBody(pp[1] * u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
        bbflux = bb_lam(xval).value * 3.14  # in units of ergs/cm2/s/A
        tau = pp[2] * (xval.value / lambda_BE) ** 3
        result = pp[0] * bbflux * (1 - np.exp(-tau))
        ind = np.where(xval.value > lambda_BE, True, False)
        if ind.any() == True:
            result[ind] = 0
        return result

    def F_poly_conti(self, xval, pp, x0=3000):
        """Fit the continuum with a polynomial component account for the dust reddening with a*X+b*X^2+c*X^3
        
        TODO: See if LMFIT's built-in modeles improved performance and numerical stability
        https://lmfit.github.io/lmfit-py/builtin_models.html
        
        """
        xval2 = xval - x0
        # rescale pp for numerical precision
        yvals = [(pp[i] / 1e6) * xval2 ** (i + 1) for i in range(len(pp))]
        return np.sum(yvals, axis=0)

    def flux2L(self, flux, z=None):
        """Transfer flux to luminoity assuming a flat Universe"""
        if z is None:
            z = self.z
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        d_L = cosmo.luminosity_distance(z).to(u.cm).value  # unit cm
        L = flux * 1e-17 * 4 * np.pi * d_L ** 2  # erg/s/A
        return L

    def Onegauss(self, xval, pp):
        """The single Gaussian model used to fit the emission lines 
        Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave
        
        This is evaluated many times within the scipy optimize, so we want to keep the code as fast as possible
        Hence, we avoid calling any external libraries like astropy's Gaussian here
        
        It is slightly faster to fit the (un-normalized) amplitude directly to avoid blow-up at small sigma
        
        xval: wavelength array in AA

        TODO: See if LMFIT's built-in modeles improved performance
        https://lmfit.github.io/lmfit-py/builtin_models.html
        
        pp: Paramaters [3]
            scale: line amplitude
            wave: central ln wavelength in AA
            sigma: width in km/s
        """

        return pp[0] * np.exp(-(xval - pp[1]) ** 2 / (2 * pp[2] ** 2))

    def _Manygauss(self, xval, pp):
        """
        Fast multi-Gaussian model used to fit the emission lines
        
        This is evaluated many times within the scipy optimize, so we want to keep the code as fast as possible
        Hence, we avoid calling any external libraries like astropy's Gaussian here
        It is vectorized so pp must have shape [ngauss, 3]
        
        It is slightly faster to fit the (un-normalized) amplitude directly to avoid blow-up at small sigma
        
        xval: wavelength array in AA
        
        pp: Paramaters array [ngauss, 3]
            scale: line amplitude
            wave: central ln wavelength in AA
            sigma: width in km/s
        """

        return np.sum(pp[:, 0] * np.exp(-(xval[:, np.newaxis] - pp[:, 1]) ** 2 / (2 * pp[:, 2] ** 2)), axis=1)

    def Manygauss(self, xval, pp):
        """
        Robust function for multi-Gaussian model used to fit the emission lines
        
        This is evaluated many times within the scipy optimize, so we want to keep the code as fast as possible
        Hence, it is vectorized so pp must have shape [ngauss, 3]
        
        xval: wavelength array in AA
        
        pp: Paramaters [ngauss*3]
            scale: line amplitude
            wave: central ln wavelength in AA
            sigma: width in km/s
        """

        # Reshape parameters array for vectorization
        ngauss = len(pp) // 3
        if ngauss > 0:
            pp_shaped = np.reshape(pp, (ngauss, 3))
            return self._Manygauss(xval, pp_shaped)
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
            pp = self.conti_params[:6]

        Fe_flux_result = np.array([])
        Fe_flux_type = np.array([])
        Fe_flux_name = np.array([])

        if ranges is not None:
            if np.array(ranges).ndim == 1:
                Fe_flux_result = np.append(Fe_flux_result, self._calculate_Fe_flux(ranges, pp))
                Fe_flux_name = np.append(Fe_flux_name,
                                         'Fe_flux_' + str(int(np.min(ranges))) + '_' + str(int(np.max(ranges))))
                Fe_flux_type = np.append(Fe_flux_type, 'float')

            elif np.array(ranges).ndim == 2:
                for iii in range(np.array(ranges).shape[0]):
                    Fe_flux_result = np.append(Fe_flux_result, self._calculate_Fe_flux(ranges[iii], pp))
                    Fe_flux_name = np.append(Fe_flux_name,
                                             'Fe_flux_' + str(int(np.min(ranges[iii]))) + '_' + str(
                                                 int(np.max(ranges[iii]))))
                    Fe_flux_type = np.append(Fe_flux_type, 'float')
            else:
                raise IndexError('The parameter ranges only adopts arrays with 1 or 2 dimensions.')

        return Fe_flux_result, Fe_flux_type, Fe_flux_name

    def _calculate_Fe_flux(self, measure_range, pp):
        """Calculate the flux of fitted FeII template within one given wavelength range.
        The pp could be an array with a length of 3 or 6. If 3 parameters were give, function will choose a
        proper template (MgII or balmer) according to the range. If the range give excess both template, an
        error would be arose. If 6 parameters were give, function would adopt the first three for the MgII
        template and the last three for the balmer."""

        balmer_range = np.array([3686., 7484.])
        mgii_range = np.array([1200., 3500.])
        upper = np.min([np.max(measure_range), np.max(self.wave)])
        lower = np.max([np.min(measure_range), np.min(self.wave)])
        if upper < np.max(measure_range) or lower > np.min(measure_range):
            if self.verbose:
                print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) exceeded '
                      'the boundary of spectrum wavelength range. The excess part would be set to zero!')
        disp = 1e-4 * np.log(10)
        xval = np.exp(np.arange(np.log(lower), np.log(upper), disp))
        if len(xval) < self.n_pix_min_conti:
            if self.verbose:
                print(f'Warning: Available part in range {measure_range} is less than {self.n_pix_min_conti} pixel. '
                      f'Flux = -1 would be given!')
            return -1

        if len(pp) == 3:
            if upper <= mgii_range[1] and lower >= mgii_range[0]:
                yval = self.Fe_flux_mgii(xval, pp)
            elif upper <= balmer_range[1] and lower >= balmer_range[0]:
                yval = self.Fe_flux_balmer(xval, pp)
            else:
                raise OverflowError('Only 3 parameters are given in this function. \
                Make sure the range is within [1200., 3500.] or [3686., 7484.]!')

        elif len(pp) == 6:
            yval = self.Fe_flux_mgii(xval, pp[:3]) + self.Fe_flux_balmer(xval, pp[3:])
            if upper > balmer_range[1] or lower < mgii_range[0]:
                if self.verbose:
                    print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) '
                          'exceeded the template range [1200., 7478.]. The excess part would be set to zero!')
            elif upper > mgii_range[1] and lower < balmer_range[0]:
                if self.verbose:
                    print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) '
                          'contained range [3500., 3686.] which is the gap between FeII templates and would be set to zero!')

        else:
            raise IndexError('The parameter pp only adopts a list of 3 or 6.')

        flux = integrate.trapz(yval[(xval >= lower) & (xval <= upper)], xval[(xval >= lower) & (xval <= upper)])
        return flux

    def read_out_params(self, param_file_path='qsopar.fits'):
        # read result customized parameters
        hdul = fits.open(param_file_path)

        data = hdul[4].data
        self.Fe_flux_range = np.array(data['Fe_flux_range'][0])
        self.L_conti_wave = np.array(data['cont_loc'][0])

        return data


def get_err(s, margin=0.16, axis=0, default_value=-1.):
    """
    Get 100*margin percent distribution of a given data.
    :param s: 1-D array or 2-D array. If a 1-D array is given, the data will deem the array as the data sample and the
    axis parameter will be ignored. If a 2-D array is given, how the function deel with this data will depend on the
    axis. If axis==0, the function will calculate the distribution of each column of the array. If axis==1, the
    function will calculate the distribution of each row of the array.
    :param margin: The margin of the distribution. The default value is 16%, which means the function will calculate
    about 1 sigma error for each sample
    :param axis: How the function deel with the data, see above.
    :return: float or 1-D array, depends on the input data.
    """
    s = np.array(s)
    s[s == default_value] = np.nan
    margin_per = int(margin * 100)
    if s.ndim == 1:
        N_samp = len(s)
        if np.sum(np.isnan(s)) / N_samp > 0.5:
            return default_value
        else:
            # if self.verbose:
            #     print('Warning: The input data contains more than 50% nan values. The error would be set to -1.')
            return np.diff(np.nanpercentile(s, (margin_per, 100 - margin_per)))[0] / 2
    elif s.ndim == 2:
        if axis == 1:
            s = s.T
        if not axis in [0, 1]:
            raise IndexError('The axis parameter only adopts 0 or 1.')
        N_samp = s.shape[0]
        Na_idx = np.where(np.sum(np.isnan(s), axis=0) > N_samp / 2, True, False)
        data_err = np.diff(np.nanpercentile(s, (margin_per, 100 - margin_per), axis=0), axis=0)[0] / 2
        data_err[Na_idx] = default_value
        return data_err
    else:
        raise IndexError('The input data only adopts 1-D or 2-D array.')


def read_conti_params(param_file_path='qsopar.fits'):
    # read line parameter
    hdul = fits.open(param_file_path)

    conti_windows = np.vstack([np.array(t) for t in hdul[2].data])
    data = hdul[3].data

    return data, conti_windows


def read_line_params(param_file_path='qsopar.fits'):
    # read line parameter
    hdul = fits.open(param_file_path)
    data = hdul[1].data

    # print('Reading parameter file:', param_file_path)

    return data

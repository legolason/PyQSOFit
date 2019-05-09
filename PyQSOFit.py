# A code for quasar spectrum fitting
#Last modified on 2/27/2019
#Auther: Hengxiao Guo AT UIUC
#Email: hengxiaoguo AT gmail DOT com
#Co-Auther Shu Wang, Yue Shen
#version 1.0
#-------------------------------------------------

import glob, os,sys,timeit
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sfdmap
from scipy import interpolate
from kapteyn import kmpfit
from PyAstronomy import pyasl
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.blackbody import blackbody_lambda
from astropy.table import Table
import warnings
warnings.filterwarnings("ignore")


def smooth(y, box_pts):
    "Smooth the flux with n pixels"
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode = 'same')
    return y_smooth


def Fe_flux_mgii(xval,pp):
    "Fit the UV Fe compoent on the continuum from 1200 to 3500 A based on the Boroson & Green 1992."
    yval = np.zeros_like(xval)
    wave_Fe_mgii = 10**fe_uv[:,0]
    flux_Fe_mgii = fe_uv[:,1]*10**15
    Fe_FWHM = pp[1]
    xval_new = xval*(1.0 + pp[2])      
    
    ind = np.where((xval_new > 1200.) & (xval_new < 3500.), True,False)
    if np.sum(ind) > 100:
        if Fe_FWHM < 900.0:
            sig_conv = np.sqrt(910.0**2 - 900.0**2)/2./np.sqrt(2.*np.log(2.)) 
        else:
            sig_conv = np.sqrt(Fe_FWHM**2 - 900.0**2)/2./np.sqrt(2.*np.log(2.))   #in km/s
        #Get sigma in pixel space
        sig_pix = sig_conv/106.3     # 106.3 km/s is the dispersion for the BG92 FeII template
        khalfsz = np.round(4*sig_pix+1,0)
        xx= np.arange(0,khalfsz*2,1) - khalfsz
        kernel = np.exp(-xx**2/(2*sig_pix**2))
        kernel = kernel/np.sum(kernel)
        
        flux_Fe_conv = np.convolve(flux_Fe_mgii, kernel,'same')
        tck = interpolate.splrep(wave_Fe_mgii, flux_Fe_conv) 
        yval[ind] = pp[0]*interpolate.splev(xval_new[ind], tck)
    return yval

def Fe_flux_balmer(xval,pp):
    "Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"
    yval = np.zeros_like(xval)
   
    wave_Fe_balmer = 10**fe_op[:,0]
    flux_Fe_balmer = fe_op[:,1]*10**15
    ind = np.where((wave_Fe_balmer > 3686.) & (wave_Fe_balmer < 7484.),True,False)
    wave_Fe_balmer = wave_Fe_balmer[ind]
    flux_Fe_balmer = flux_Fe_balmer[ind]
    Fe_FWHM = pp[1]
    xval_new = xval*(1.0 + pp[2])
    ind = np.where((xval_new > 3686.) & (xval_new < 7484.),True,False)
    if np.sum(ind) > 100:
        if Fe_FWHM < 900.0:
            sig_conv = np.sqrt(910.0**2 - 900.0**2)/2./np.sqrt(2.*np.log(2.)) 
        else:
            sig_conv = np.sqrt(Fe_FWHM**2 - 900.0**2)/2./np.sqrt(2.*np.log(2.))  #in km/s
        #Get sigma in pixel space
        sig_pix = sig_conv/106.3    # 106.3 km/s is the dispersion for the BG92 FeII template
        khalfsz = np.round(4*sig_pix+1,0)
        xx= np.arange(0,khalfsz*2,1) - khalfsz
        kernel = np.exp(-xx**2/(2*sig_pix**2))
        kernel = kernel/np.sum(kernel)
        flux_Fe_conv = np.convolve(flux_Fe_balmer, kernel,'same')
        tck = interpolate.splrep(wave_Fe_balmer, flux_Fe_conv) 
        yval[ind] = pp[0]*interpolate.splev(xval_new[ind], tck)
    return yval

def balmer_conti(xval, pp):
    """Fit the Balmer continuum from the model of Dietrich+02"""
    #xval = input wavelength, in units of A
    #pp=[norm, Te, tau_BE] -- in units of [--, K, --]

    lambda_BE = 3646.  # A
    bbflux = blackbody_lambda(xval,pp[1]).value*3.14 # in units of ergs/cm2/s/A
    tau = pp[2]*(xval/lambda_BE)**3
    result = pp[0]*bbflux*(1. - np.exp(-tau))
    ind = np.where(xval > lambda_BE,True,False)
    if ind.any() == True:
        result[ind] = 0.
    return result


def f_poly_conti(xval, pp):
    """Fit the continuum with a polynomial component account for the dust reddening with a*X+b*X^2+c*X^3"""
    xval2 = xval - 3000.
    yval = 0.*xval2
    for i in range(len(pp)):
        yval = yval + pp[i]*xval2**(i+1)
    return yval

def flux2L(flux,z):
    """Transfer flux to luminoity assuming a flat Universe"""
    cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
    DL = cosmo.luminosity_distance(z).value*10**6*3.08*10**18 # unit cm
    L = flux*1.e-17*4.*np.pi*DL**2   #erg/s/A
    return L


#----line function------
def onegauss(xval, pp):
    """The single Gaussian model used to fit the emission lines 
    Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave
    """
    
    term1 = np.exp( - (xval - pp[1])**2 / (2. * pp[2]**2) )
    yval = pp[0] * term1 / (np.sqrt(2.*np.pi) * pp[2])
    return yval

def manygauss(xval, pp):
    """The multi-Gaussian model used to fit the emission lines, it will call the onegauss function"""
    ngauss = int(pp.shape[0]/3)
    if ngauss != 0:
        yval = 0.
        for i in range(ngauss):
            yval = yval + onegauss(xval, pp[i*3:(i+1)*3])
        return yval




class QSOFit():
    
    def __init__(self,lam,flux,err,z,ra =- 999.,dec = -999.,plateid = None,mjd = None,fiberid = None,path = None,and_mask = None, or_mask = None):
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
            
        path: str
            the path of the input data
            
        and_mask, or_mask: 1-D array with Npix, optional
            the bad pixels defined from SDSS data, which can be got from SDSS datacube. 

        """
        
        self.lam = np.asarray(lam, dtype = np.float64)
        self.flux = np.asarray(flux, dtype = np.float64)
        self.err = np.asarray(err, dtype = np.float64)
        self.z = z
        self.and_mask = and_mask
        self.or_mask = or_mask
        self.ra = ra
        self.dec = dec
        self.plateid = plateid
        self.mjd = mjd
        self.fiberid = fiberid
        self.path = path
        
        
        
        
        


    def Fit(self, name = None, nsmooth = 1,and_or_mask = True,reject_badpix = True, deredden = True,wave_range = None,\
            wave_mask = None,decomposition_host = True, BC03 = False, Mi = None,npca_gal = 5, npca_qso = 20, \
            Fe_uv_op = True, poly = False, BC = False, rej_abs = False,initial_guess = None, MC = True, \
            n_trails = 1,linefit = True, tie_lambda = True, tie_width = True, tie_flux_1 = True,tie_flux_2 = True,\
            save_result = True, plot_fig = True,save_fig = True,plot_line_name = True, plot_legend = True, dustmap_path = None,\
            save_fig_path = None,save_fits_path = None,save_fits_name = None):
        
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
        
        poly: bool, optional
            if True, fit continuum with the polynomial component to account for the dust reddening. Default: False
        
        BC: bool, optional
            if True, fit continuum with Balmer continua from 1000 to 3646A. Default: False
            
        rej_abs: bool, optional
            if True, it will iterate the continuum fitting for deleting some 3 sigmas out continuum window points
            (< 3500A), which might fall into the broad absorption lines. Default: False 
            
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
        self.poly = poly
        self.BC = BC
        self.rej_abs = rej_abs
        self.MC = MC
        self.n_trails = n_trails
        self.tie_lambda = tie_lambda
        self.tie_width = tie_width
        self.tie_flux_1 = tie_flux_1
        self.tie_flux_2 = tie_flux_2
        self.plot_line_name = plot_line_name
        self.plot_legend = plot_legend
        self.save_fig = save_fig
        
        
        
        #get the source name in plate-mjd-fiber, if no then None
        if name is None:
            if np.array([self.plateid,self.mjd,self.fiberid]).any() is not None :
                self.sdss_name = str(self.plateid).zfill(4)+'-'+str(self.mjd)+'-'+str(self.fiberid).zfill(4)
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
            
        
        #set default path for figure and fits
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
        ind_gooderror = np.where( (self.err != 0) & ~np.isinf(self.err), True, False)
        err_good = self.err[ind_gooderror]
        flux_good = self.flux[ind_gooderror]
        lam_good = self.lam[ind_gooderror]
        
        if (self.and_mask is not None) & (self.or_mask is not None):
            and_mask_good = self.and_mask[ind_gooderror]
            or_mask_good = self.or_mask[ind_gooderror]
            del self.and_mask,self.or_mask
            self.and_mask = and_mask_good
            self.or_mask = or_mask_good
        del self.err, self.flux, self.lam
        self.err = err_good
        self.flux = flux_good
        self.lam = lam_good


        if nsmooth is not None:
            self.flux = smooth(self.flux,nsmooth)
            self.err = smooth(self.err,nsmooth)
        if (and_or_mask == True) and (self.and_mask is not None or self.or_mask is not None) :
            self._MaskSdssAndOr(self.lam,self.flux,self.err,self.and_mask,self.or_mask)
        if reject_badpix == True:
            self._RejectBadPix(self.lam,self.flux,self.err)
        if wave_range is not None:
            self._WaveTrim(self.lam,self.flux,self.err,self.z)
        if wave_mask is not None:
            self._WaveMsk(self.lam,self.flux,self.err,self.z)    
        if deredden == True and self.ra != -999. and self.dec != -999.:
            self._DeRedden(self.lam,self.flux,self.ra,self.dec,dustmap_path)



        self._RestFrame(self.lam,self.flux,self.z)
        self._CalculateSN(self.lam,self.flux)
        self._OrignialSpec(self.wave,self.flux,self.err)

        # do host decomposition --------------
        if self.z < 1.16 and decomposition_host == True:
            self._DoDecomposition(self.wave,self.flux,self.err,self.path)
        else:
            self.decomposed = False
            if self.z > 1.16 and decomposition_host == True:
                print('redshift larger than 1.16 is not allowed for host decomposion!')

        #fit continuum --------------------
        self._DoContiFit(self.wave,self.flux,self.err,self.ra,self.dec,self.plateid,self.mjd,self.fiberid)
        #fit line
        if linefit == True:

            self._DoLineFit(self.wave,self.line_flux,self.err,self.conti_fit)

        #save data -------
        if save_result == True:
            if linefit == False:
                self.line_result = np.array([])
                self.line_result_name = np.array([])
            self._SaveResult(self.conti_result,self.conti_result_name,self.line_result,self.line_result_name,save_fits_path,save_fits_name)


        #plot fig and save ------
        if plot_fig == True:
            if linefit == False:
                self.gauss_result = np.array([])
                self.all_comp_range =np.array([])
                self.uniq_linecomp_sort = np.array([])
            self._PlotFig(self.ra,self.dec,self.z,self.wave,self.flux,self.err,decomposition_host,linefit,\
                         self.tmp_all,self.gauss_result,self.f_conti_model,self.conti_fit,self.all_comp_range,\
                         self.uniq_linecomp_sort,self.line_flux,save_fig_path)


    def _MaskSdssAndOr(self,lam,flux,err,and_mask,or_mask):
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
        ind_and_or=np.where( (and_mask == 0) & (or_mask == 0), True,False)
        del self.lam,self.flux,self.err
        self.lam,self.flux,self.err = lam[ind_and_or],flux[ind_and_or],err[ind_and_or]
        
    def _RejectBadPix(self,lam,flux,err):
        """
        Reject 10 most possiable outliers, input wavelength, flux and error. Return a different size wavelength,
        flux, and error.
        """
        #-----remove bad pixels, but not for high SN spectrum------------
        ind_bad = pyasl.pointDistGESD(flux, 10)
        wv = np.asarray([i for j, i in enumerate(lam) if j not in ind_bad[1]], dtype = np.float64)
        fx = np.asarray([i for j, i in enumerate(flux) if j not in ind_bad[1]], dtype = np.float64)
        er = np.asarray([i for j, i in enumerate(err) if j not in ind_bad[1]], dtype = np.float64)
        del self.lam,self.flux,self.err
        self.lam,self.flux,self.err = wv,fx,er
        return self.lam,self.flux,self.err

    def _WaveTrim(self,lam,flux,err,z):
        """
        Trim spectrum with a range in the rest frame. 
        """
        # trim spectrum e.g., local fit emiision lines
        ind_trim = np.where( (lam/(1.+z) > self.wave_range[0]) & (lam/(1.+z) < self.wave_range[1]),True,False)
        del self.lam,self.flux,self.err
        self.lam,self.flux,self.err = lam[ind_trim],flux[ind_trim],err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError("No enough pixels in the input wave_range!")
        return self.lam,self.flux,self.err

    def _WaveMsk(self,lam,flux,err,z):
        """Block the bad pixels or absorption lines in spectrum."""
             
        for msk in range(len(self.wave_mask)):
            try:
                ind_not_mask = ~np.where( (lam/(1.+z) > self.wave_mask[msk,0]) & (lam/(1.+z) < self.wave_mask[msk,1]), True, False)
            except IndexError:
                raise RuntimeError("Wave_mask should be 2D array, e.g., np.array([[2000,3000],[3100,4000]]).")
   
            del self.lam,self.flux,self.err
            self.lam,self.flux,self.err = lam[ind_not_mask],flux[ind_not_mask],err[ind_not_mask]
            lam, flux, err = self.lam, self.flux, self.err
        return self.lam,self.flux,self.err
            
    def _DeRedden(self,lam,flux,ra,dec,dustmap_path):
        """Correct the Galatical extinction"""          
        m = sfdmap.SFDMap(dustmap_path) 
        flux_unred = pyasl.unred(lam,flux,m.ebv(ra,dec))
        del self.flux
        self.flux = flux_unred
        return self.flux
        
    def _RestFrame(self,lam,flux,z):
        """Move wavelenth and flux to rest frame"""
        self.wave = lam/(1.+z)
        self.flux = flux*(1.+z)
        return self.wave,self.flux
    
    def _OrignialSpec(self,wave,flux,err):
        """save the orignial spectrum before host galaxy decompsition"""
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _CalculateSN(self,wave,flux):
        """calculate the spectral SN ratio for 1350, 3000, 5100A, return the mean value of Three spots"""
        if (wave.min() < 1350. and wave.max() >1350.) or (wave.min() < 3000. and wave.max() >3000.) or \
        (wave.min() < 5100. and wave.max() >5100.):
            ind5100 = np.where( (wave >5080.) & (wave<5130.),  True,False)
            ind3000 = np.where( (wave >3000.) & (wave<3050.),  True,False)
            ind1350 = np.where( (wave >1325.) & (wave<1375.),  True,False)
            
            tmp_SN = np.array([flux[ind5100].mean()/flux[ind5100].std(),flux[ind3000].mean()/flux[ind3000].std(),\
                          flux[ind1350].mean()/flux[ind1350].std()])
            tmp_SN = tmp_SN[~np.isnan(tmp_SN)]
            self.SN_ratio_conti = tmp_SN.mean()
        else:
            self.SN_ratio_conti = -1. 

        return self.SN_ratio_conti
    
    def _DoDecomposition(self,wave,flux,err,path):
        """Decompose the host galaxy from QSO"""
        datacube = self._HostDecompose(self.wave,self.flux,self.err,self.z,self.Mi,self.npca_gal,self.npca_qso,path)

        #for some negtive host templete, we do not do the decomposition 
        if np.sum(np.where( datacube[3,:] < 0., True,False)) > 100:
            self.host = np.zeros(len(wave))
            self.decomposed = False
            print( 'Get negtive host galaxy flux larger than 100 pixels, decomposition is not applied!')
        else:
            self.decomposed = True
            del self.wave,self.flux,self.err
            self.wave = datacube[0,:]
            #block OIII, ha,NII,SII,OII,Ha,Hb,Hr,hdelta
        
            line_mask = np.where( (self.wave < 4970.) & (self.wave > 4950.) |
                                  (self.wave < 5020.) & (self.wave > 5000.) | 
                                  (self.wave < 6590.) & (self.wave > 6540.) |
				  (self.wave < 6740.) & (self.wave > 6710.) |
                                  (self.wave < 3737.) & (self.wave > 3717.) |
                                  (self.wave < 4872.) & (self.wave > 4852.) |
                                  (self.wave < 4350.) & (self.wave > 4330.) |
                                  (self.wave < 4111.) & (self.wave > 4091.), True,False)
            
            f = interpolate.interp1d(self.wave[~line_mask],datacube[3,:][~line_mask],bounds_error = False, fill_value = 0)
            masked_host = f(self.wave)
            self.flux = datacube[1,:]-masked_host # QSO flux without host
            self.err = datacube[2,:]
            self.host = datacube[3,:]
            self.qso = datacube[4,:]
            self.host_data = datacube[1,:]-self.qso
        return self.wave,self.flux,self.err 
    
    
    def _HostDecompose(self,wave,flux,err,z,Mi,npca_gal,npca_qso,path):
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
            flux_gal = gal['pca'].reshape(gal['pca'].shape[1],gal['pca'].shape[2])
        if self.BC03 == True:
            cc=0
            flux03 = np.array([])
            for i in glob.glob(path+'/bc03/*.gz'):
                cc=cc+1
                gal_temp=np.genfromtxt(i)
                wave_gal = gal_temp[:,0]
                flux03 = np.concatenate((flux03,gal_temp[:,1]))
            flux_gal = np.array(flux03).reshape(cc,-1)

        if Mi is None:
            quasar = fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_global.fits')
        else:    
            if -24 < Mi <= -22 and 0.08 <= z < 0.53:
                quasar=fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_CZBIN1.fits')
            elif -26 < Mi <= -24 and 0.08 <= z < 0.53:
                quasar=fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_DZBIN1.fits')
            elif -24 < Mi <= -22 and 0.53 <= z < 1.16:
                quasar=fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_BZBIN2.fits')
            elif -26 < Mi <= -24 and 0.53 <= z < 1.16:
                quasar=fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_CZBIN2.fits')
            elif -28 < Mi <= -26 and 0.53 <= z < 1.16:
                quasar=fits.open(path+'pca/Yip_pca_templates/qso_eigenspec_Yip2004_DZBIN2.fits')
            else:
                raise RuntimeError('Host galaxy template is not available for this redshift and Magnitude!')
            

        qso = quasar[1].data
        wave_qso = qso['wave'].flatten()
        flux_qso = qso['pca'].reshape(qso['pca'].shape[1],qso['pca'].shape[2])

        # get the shortest wavelength range
        wave_min = max(wave.min(),wave_gal.min(),wave_qso.min())
        wave_max = min(wave.max(),wave_gal.max(),wave_qso.max())


        ind_data = np.where( (wave > wave_min) & (wave < wave_max),True,False)
        ind_gal = np.where( (wave_gal > wave_min-1.) & (wave_gal < wave_max+1.), True,False)
        ind_qso = np.where( (wave_qso > wave_min-1.) & (wave_qso < wave_max+1.), True,False)



        flux_gal_new = np.zeros(flux_gal.shape[0]*flux[ind_data].shape[0]).reshape(flux_gal.shape[0],flux[ind_data].shape[0])
        flux_qso_new = np.zeros(flux_qso.shape[0]*flux[ind_data].shape[0]).reshape(flux_qso.shape[0],flux[ind_data].shape[0])
        for i in range(flux_gal.shape[0]):
            fgal = interpolate.interp1d(wave_gal[ind_gal], flux_gal[i,ind_gal], bounds_error = False, fill_value = 0)
            flux_gal_new[i,:] = fgal(wave[ind_data])
        for i in range(flux_qso.shape[0]):
            fqso = interpolate.interp1d(wave_qso[ind_qso], flux_qso[0,ind_qso], bounds_error = False, fill_value = 0)
            flux_qso_new[i,:] = fqso(wave[ind_data])


        wave_new = wave[ind_data]
        flux_new = flux[ind_data]
        err_new = err[ind_data]

        flux_temp=np.vstack((flux_gal_new[0:npca_gal,:],flux_qso_new[0:npca_qso,:]))
        res=np.linalg.lstsq(flux_temp.T, flux_new)[0]

        host_flux = np.dot(res[0:npca_gal],flux_temp[0:npca_gal])
        qso_flux = np.dot(res[npca_gal:],flux_temp[npca_gal:])

        data_cube = np.vstack((wave_new,flux_new,err_new,host_flux,qso_flux))

        ind_f4200 = np.where( (wave_new > 4160.) & (wave_new < 4210.), True,False )
        frac_host_4200 = np.sum(host_flux[ind_f4200])/np.sum(flux_new[ind_f4200])
        ind_f5100 = np.where( (wave_new > 5080.) & (wave_new < 5130.), True,False )
        frac_host_5100 = np.sum(host_flux[ind_f5100])/np.sum(flux_new[ind_f5100])


        return data_cube #,frac_host_4200,frac_host_5100

    
    def _DoContiFit(self,wave,flux,err,ra,dec,plateid,mjd,fiberid):
        """Fit the continuum with PL, Polynomial, UV/optical FeII, Balmer continuum"""
        global fe_uv,fe_op
        fe_uv = np.genfromtxt(self.path+'fe_uv.txt') 
        fe_op = np.genfromtxt(self.path+'fe_optical.txt') 
        
        #do continuum fit--------------------------
        window_all = np.array([ [1150., 1170.], [1275., 1290.], [1350., 1360.], [1445., 1465.],\
                            [1690., 1705.], [1770., 1810.], [1970., 2400.], [2480., 2675.],\
                            [2925., 3400.], [3775., 3832.], [4000., 4050.], [4200., 4230.],\
                            [4435., 4640.], [5100., 5535.], [6005., 6035.], [6110., 6250.],\
                            [6800., 7000.], [7160., 7180.], [7500., 7800.], [8050., 8150.] ])

        tmp_all = np.array([np.repeat(False,len(wave))]).flatten()
        for jj in range(len(window_all)):
            tmp = np.where( (wave > window_all[jj,0]) & (wave < window_all[jj,1]), True, False)
            tmp_all = np.any([tmp_all, tmp], axis = 0)


        if wave[tmp_all].shape[0] < 10:
            print( 'Continuum fitting pixel < 10.  ')

        # set initial paramiters for continuum
        if self.initial_guess is not None:
            pp0 = self.initial_guess
        else:
            pp0 = np.array([0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 15000., 0.5, 0., 0., 0.])

        
        conti_fit = kmpfit.Fitter(residuals = self._residuals, data = (wave[tmp_all], flux[tmp_all], err[tmp_all])) 
        tmp_parinfo = [{'limits':(0.,10.**10)},{'limits':(1200.,10000.)},{'limits':(-0.01,0.01)},\
                       {'limits':(0.,10.**10)},{'limits':(1200.,10000.)},{'limits':(-0.01,0.01)},\
                       {'limits':(0.,10.**10)},{'limits':(-5.,3.)},\
                       {'limits':(0.,10.**10)},{'limits':(10000.,50000.)},{'limits':(0.1,2.)},\
                        None,None,None]
        conti_fit.parinfo = tmp_parinfo
        conti_fit.fit(params0 = pp0)



        #Perform one iteration to remove 3sigma pixel below the first continuum fit 
        #to avoid the continuum windows falls within a BAL trough
        if self.rej_abs == True:
            if self.poly == True:
                tmp_conti = conti_fit.params[6]*(wave[tmp_all]/3000.0)**conti_fit.params[7] +f_poly_conti(wave[tmp_all],conti_fit.params[11:])
            else:
                tmp_conti = conti_fit.params[6]*(wave[tmp_all]/3000.0)**conti_fit.params[7]
            ind_noBAL = ~np.where( (flux[tmp_all] < tmp_conti-3.*err[tmp_all]) & (wave[tmp_all] <3500.) ,True,False)
            f = kmpfit.Fitter(residuals = self._residuals, data = (wave[tmp_all][ind_noBAL], smooth(flux[tmp_all][ind_noBAL],10), err[tmp_all][ind_noBAL]))
            conti_fit.parinfo = tmp_parinfo
            conti_fit.fit(params0 = pp0)
        



        #calculate continuum luminoisty
        L = self._L_conti(wave,conti_fit.params)

        # calculate MC err
        if self.MC == True and self.n_trails > 0:
            conti_para_std,all_L_std = self._conti_mc(self.wave[tmp_all], self.flux[tmp_all], self.err[tmp_all],pp0,conti_fit.parinfo,self.n_trails)


        # get conti result -----------------------------
        if self.MC == False:
            self.conti_result = np.array([ra,dec,self.plateid,self.mjd,self.fiberid,self.z,self.SN_ratio_conti,conti_fit.params[1],conti_fit.params[4],conti_fit.params[6],conti_fit.params[7],\
                               conti_fit.params[11],conti_fit.params[12],conti_fit.params[13],L[0],L[1],L[2]])
            self.conti_result_name = np.array(['ra','dec','plateid','MJD','fiberid','redshift','SN_ratio_conti','Fe_uv_FWHM','Fe_op_FWHM','PL_norm','PL_slope',\
                          'POLY_a','POLY_b','POLY_c','L1350','L3000','L5100'])

        else:
            self.conti_result = np.array([ra,dec,plateid,mjd,fiberid,self.z,self.SN_ratio_conti,conti_fit.params[1],conti_para_std[1],conti_fit.params[4],conti_para_std[4],conti_fit.params[6],conti_para_std[6],conti_fit.params[7],conti_para_std[7],\
                               conti_fit.params[11],conti_para_std[11],conti_fit.params[12],conti_para_std[12],conti_fit.params[13],conti_para_std[13],L[0],all_L_std[0],L[1],all_L_std[1],L[2],all_L_std[2]])
            self.conti_result_name = np.array(['ra','dec','plateid','MJD','fiberid','redshift','SN_ratio_conti','Fe_uv_FWHM','Fe_uv_FWHM_err','Fe_op_FWHM','Fe_op_FWHM_err','PL_norm','PL_norm_err','PL_slope',\
                          'PL_slope_err','POLY_a','POLY_a_err','POLY_b','POLY_b_err','POLY_c','POLY_c_err','L1350','L1350_err','L3000','L3000_err','L5100','L5100_err'])
        
        self.conti_fit = conti_fit 
        self.tmp_all = tmp_all
        
        
        #save different models--------------------
        f_fe_mgii_model = Fe_flux_mgii(wave,conti_fit.params[0:3])
        f_fe_balmer_model = Fe_flux_balmer(wave,conti_fit.params[3:6])
        f_pl_model = conti_fit.params[6]*(wave/3000.0)**conti_fit.params[7]
        f_bc_model = balmer_conti(wave, conti_fit.params[8:11])
        f_poly_model = f_poly_conti(wave, conti_fit.params[11:])
        f_conti_model = f_pl_model + f_fe_mgii_model + f_fe_balmer_model + f_poly_model+ f_bc_model
        line_flux = flux - f_conti_model
        
        self.f_conti_model = f_conti_model
        self.f_bc_model = f_bc_model
        self.f_fe_uv_model = f_fe_mgii_model
        self.f_fe_op_model = f_fe_balmer_model
        self.f_pl_model = f_pl_model
        self.f_poly_model = f_poly_model
        self.line_flux = line_flux
        self.PL_poly_BC = f_pl_model+f_poly_model+f_bc_model
        
        return self.conti_result,self.conti_result_name

    
    def _L_conti(self,wave,pp):
        """Calculate continuum Luminoisity at 1350,3000,5100A"""
        conti_flux = pp[6]*(wave/3000.0)**pp[7]+f_poly_conti(wave, pp[11:])
        #plt.plot(wave,conti_flux)
        L = np.array([])
        for LL in zip([1350.,3000.,5100.]):
            if wave.max() > LL[0] and wave.min() < LL[0]: 
                L_tmp = np.asarray([np.log10(LL[0]*flux2L(conti_flux[np.where( abs(wave - LL[0]) < 5.,True,False)].mean(), self.z) )])
            else:
                L_tmp = np.array([-1.])
            L = np.concatenate([L, L_tmp]) # save log10(L1350,L3000,L5100)
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
        f_Fe_MgII = Fe_flux_mgii(xval, pp[0:3])      # iron flux for MgII line region
        f_Fe_Balmer = Fe_flux_balmer(xval, pp[3:6])  # iron flux for balmer line region
        f_pl = pp[6]*(xval/3000.0)**pp[7]             # power-law continuum
        f_conti_BC = balmer_conti(xval, pp[8:11])    # Balmer continuum
        f_poly = f_poly_conti(xval, pp[11:])        # polynormal conponent for reddened spectra


        if self.Fe_uv_op == True and self.poly == False and self.BC == False :
            yval = f_pl + f_Fe_MgII + f_Fe_Balmer 
        elif self.Fe_uv_op == True and self.poly == True and self.BC == False:
            yval = f_pl + f_Fe_MgII + f_Fe_Balmer + f_poly 
        elif self.Fe_uv_op == True and self.poly == False and self.BC == True :
            yval = f_pl + f_Fe_MgII + f_Fe_Balmer + f_conti_BC 
        elif self.Fe_uv_op == False and self.poly == True and self.BC == False :
            yval = f_pl + f_poly
        elif self.Fe_uv_op == False and self.poly == False and self.BC == False :
            yval = f_pl 
        elif self.Fe_uv_op == False and self.poly == False and self.BC == True :
            yval = f_pl  + f_Fe_Balmer + f_conti_BC 
        elif self.Fe_uv_op == True and self.poly == True and self.BC == True :
            yval = f_pl + f_Fe_MgII + f_Fe_Balmer + f_poly + f_conti_BC 
        elif self.Fe_uv_op == False and self.poly == True and self.BC == True :
            yval = f_pl  + f_Fe_Balmer + f_poly + f_conti_BC 
        else:
            raise RuntimeError('No this option for Fe_uv_op, poly and BC!')
        return yval
    
    def _residuals(self, pp, data):
        """Continual residual function used in kmpfit"""
        xval, yval, weight = data
        return (yval - self._f_conti_all(xval,pp))/weight
    
    
    #---------MC error for continuum parameters-------------------
    def _conti_mc(self,x,y,err,pp0,pp_limits,n_trails):
        """Calculate the continual parameters' Monte carlo errrors"""
        all_para = np.zeros(len(pp0)*n_trails).reshape(len(pp0),n_trails)
        all_L = np.zeros(3*n_trails).reshape(3,n_trails)
        all_para_std = np.zeros(len(pp0))
        all_L_std = np.zeros(3)
        for tra in range(n_trails):
            flux = y + np.random.randn(len(y))*err
            conti_fit = kmpfit.Fitter(residuals = self._residuals, data = (x, flux, err), maxiter = 50)
            conti_fit.parinfo = pp_limits
            conti_fit.fit(params0 = pp0)
            all_para[:,tra] = conti_fit.params

            all_L[:,tra] = np.asarray(self._L_conti(x,conti_fit.params))
        for st in range(len(pp0)):
            all_para_std[st] = all_para[st,:].std() 
        for ll in range(3):    
            all_L_std[ll] = all_L[ll,:].std()
        return all_para_std,all_L_std
    
    
    #line function-----------
    def _DoLineFit(self,wave,line_flux,err,f):
        """Fit the emission lines with Gaussian profile """

        #remove abosorbtion line in emission line region
        # remove the pixels below continuum 
        ind_neg_line = ~np.where((((wave > 2700.) & (wave <2900.)) | ((wave > 1700.) & (wave <1970.))|\
                               ((wave > 1500.) & (wave <1700.)) | ((wave > 1290.) & (wave <1450.))|\
                               ((wave > 1150.) & (wave <1290.))) & (line_flux < -err),True,False) 


        #read line parameter
        linepara = fits.open(self.path+'qsopar.fits')
        linelist = linepara[1].data
        self.linelist = linelist
        
        ind_kind_line = np.where((linelist['lambda'] > wave.min()) & (linelist['lambda'] < wave.max()), True, False) 
        if ind_kind_line.any() == True :
            #sort complex name with line wavelength
            uniq_linecomp,uniq_ind = np.unique(linelist['compname'][ind_kind_line], return_index = True)
            uniq_linecomp_sort = uniq_linecomp[linelist['lambda'][ind_kind_line][uniq_ind].argsort()]
            ncomp = len(uniq_linecomp_sort)
            compname = linelist['compname']
            allcompcenter = np.sort(linelist['lambda'][ind_kind_line][uniq_ind])
           

            # loop over each complex and fit n lines simutaneously

            comp_result = np.array([])
            comp_result_name = np.array([])
            gauss_result = np.array([])
            gauss_result_name = np.array([])
            all_comp_range = np.array([])
            fur_result = np.array([])
            fur_result_name = np.array([])

            for ii in range(ncomp):
                compcenter = allcompcenter[ii]
                ind_line = np.where(linelist['compname'] == uniq_linecomp_sort[ii], True, False) # get line index
                nline_fit = np.sum(ind_line) # n line in one complex
                linelist_fit = linelist[ind_line]
                ngauss_fit = np.asarray(linelist_fit['ngauss'], dtype = int) #n gauss in each line

                #for iitmp in range(nline_fit):   # line fit together
                comp_range = [linelist_fit[0]['minwav'],linelist_fit[0]['maxwav']]   # read complex range from table
                all_comp_range = np.concatenate([all_comp_range, comp_range])

                #----tie lines--------
                self._do_tie_line(linelist,ind_line)

                #get the pixel index in complex region and remove negtive abs in line region 
                ind_n = np.where( (wave > comp_range[0]) & (wave < comp_range[1]) &(ind_neg_line == True) ,True,False)

                if np.sum(ind_n) > 10:
                    #call kmpfit for lines
                    
                    line_fit = self._do_line_kmpfit(linelist,line_flux,ind_line,ind_n,nline_fit,ngauss_fit)


                    # calculate MC err
                    if self.MC == True and self.n_trails > 0:
                        
                        all_para_std,fwhm_std,sigma_std,ew_std,peak_std,area_std = \
                        self._line_mc(np.log(wave[ind_n]), line_flux[ind_n], err[ind_n], self.line_fit_ini, self.line_fit_par, self.n_trails,compcenter)


                    #----------------------get line fitting results----------------------
                    #complex parameters
                    comp_result_tmp = np.array([[linelist['compname'][ind_line][0]],[line_fit.status],[line_fit.chi2_min],\
                               [line_fit.rchi2_min],[line_fit.niter],[line_fit.dof]]).flatten()
                    comp_result_name_tmp = np.array([str(ii+1)+'_complex_name', str(ii+1)+'_line_status',str(ii+1)+\
                                                   '_line_min_chi2',str(ii+1)+'_line_red_chi2',str(ii+1)+'_niter',str(ii+1)+'_ndof'])
                    comp_result = np.concatenate([comp_result,comp_result_tmp])
                    comp_result_name = np.concatenate([comp_result_name,comp_result_name_tmp])

                    #gauss result -------------

                    gauss_tmp = np.array([])
                    gauss_name_tmp = np.array([])

                    for gg in range(len(line_fit.params)):   
                        gauss_tmp = np.concatenate([gauss_tmp,np.array([line_fit.params[gg]])])
                        if self.MC == True:
                            gauss_tmp = np.concatenate([gauss_tmp,np.array([all_para_std[gg]])]) 
                    gauss_result = np.concatenate([gauss_result,gauss_tmp])

                    #gauss result name -----------------
                    for n in range(nline_fit):
                        for nn in range(int(ngauss_fit[n])):
                            line_name = linelist['linename'][ind_line][n]+'_'+str(nn+1)
                            if self.MC == True and self.n_trails > 0:
                                gauss_name_tmp_tmp = [line_name+'_scale',line_name+'_scale_err',line_name+'_centerwave',\
                                                    line_name+'_centerwave_err',line_name+'_sigma',line_name+'_sigma_err']
                            else:
                                gauss_name_tmp_tmp = [line_name+'_scale',line_name+'_centerwave',line_name+'_sigma']
                            gauss_name_tmp = np.concatenate([gauss_name_tmp,gauss_name_tmp_tmp])
                    gauss_result_name = np.concatenate([gauss_result_name,gauss_name_tmp])


                    #further line parameters ----------
                    fur_result_tmp = np.array([])
                    fur_result_name_tmp = np.array([])
                    fwhm,sigma,ew,peak,area = self.line_prop(compcenter,line_fit.params,'broad')
                    br_name = uniq_linecomp_sort[ii]

                    if self.MC == True:
                        fur_result_tmp = np.array([fwhm,fwhm_std,sigma,sigma_std,ew,ew_std,peak,peak_std,area,area_std])
                        fur_result_name_tmp = np.array([br_name+'_whole_br_fwhm',br_name+'_whole_br_fwhm_err',br_name+'_whole_br_sigma',br_name+'_whole_br_sigma_err',\
                                                        br_name+'_whole_br_ew',br_name+'_whole_br_ew_err',\
                                                    br_name+'_whole_br_peak',br_name+'_whole_br_peak_err',br_name+'_whole_br_area',br_name+'_whole_br_area_err']) 
                    else:
                        fur_result_tmp = np.array([fwhm,sigma,ew,peak,area])
                        fur_result_name_tmp = np.array([br_name+'_whole_br_fwhm',br_name+'_whole_br_sigma',br_name+'_whole_br_ew',br_name+'_whole_br_peak',br_name+'_whole_br_area'])
                    fur_result = np.concatenate([fur_result,fur_result_tmp])
                    fur_result_name = np.concatenate([fur_result_name,fur_result_name_tmp])

                else:
                    print("less than 10 pixels in line fitting!")
                    
            line_result = np.concatenate([comp_result,gauss_result,fur_result])
            line_result_name = np.concatenate([comp_result_name,gauss_result_name,fur_result_name])

        else:
            line_result = np.array([])
            line_result_name = np.array([])
            print("No line to fit! Pleasse set Line_fit to FALSE or enlarge wave_range!")
        
        self.gauss_result = gauss_result
        self.line_result = line_result
        self.line_result_name = line_result_name
        self.ncomp = ncomp
        self.line_flux = line_flux
        self.all_comp_range = all_comp_range
        self.uniq_linecomp_sort = uniq_linecomp_sort
        return self.line_result,self.line_result_name

    def _do_line_kmpfit(self,linelist,line_flux,ind_line,ind_n,nline_fit,ngauss_fit):
        """The key function to do the line fit with kmpfit"""
        line_fit = kmpfit.Fitter(self._residuals_line , data = (np.log(self.wave[ind_n]), line_flux[ind_n], self.err[ind_n]))  #fitting wavelength in ln space
        line_fit_ini = np.array([])
        line_fit_par = np.array([])
        for n in range(nline_fit):
            for nn in range(ngauss_fit[n]):
                #set up initial parameter guess
                line_fit_ini0 = [0.,np.log(linelist['lambda'][ind_line][n]),linelist['inisig'][ind_line][n]]
                line_fit_ini = np.concatenate([line_fit_ini,line_fit_ini0])
                #set up parameter limits
                lambda_low = np.log(linelist['lambda'][ind_line][n])-linelist['voff'][ind_line][n]
                lambda_up = np.log(linelist['lambda'][ind_line][n])+linelist['voff'][ind_line][n]
                sig_low = linelist['minsig'][ind_line][n]
                sig_up = linelist['maxsig'][ind_line][n]
                line_fit_par0 = [{'limits':(0.,10.**10)},{'limits':(lambda_low,lambda_up)},{'limits':(sig_low,sig_up)}]
                line_fit_par = np.concatenate([line_fit_par,line_fit_par0]) 

        line_fit.parinfo = line_fit_par
        line_fit.fit(params0 = line_fit_ini)
        line_fit.params = self.newpp
        self.line_fit = line_fit
        self.line_fit_ini = line_fit_ini
        self.line_fit_par = line_fit_par
        return line_fit
    
    def _do_tie_line(self, linelist, ind_line):
        """Tie line's central"""
        #--------------- tie parameter-----------
        ind_tie = np.where( linelist['vindex'][ind_line] > 0 ,True,False )
        ind_tie_1 = np.where( linelist['findex'][ind_line] == 1.,True,False)
        ind_tie_2 = np.where( linelist['findex'][ind_line] == 2.,True,False)

        #tie_value = linelist['fvalue'][ind_line][ind_tie]
        ind_tie_vindex = np.array([])
        ind_tie_findex1 = np.array([])
        ind_tie_findex2 = np.array([])

        # get index of vindex windex in initial parameters
        for iii in range(len(ind_tie)):
            if ind_tie[iii] == True:
                ind_tie_vindex = np.concatenate([ind_tie_vindex, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        self.delta_lambda = (np.log(linelist['lambda'][ind_line][ind_tie])-np.log(linelist['lambda'][ind_line][0]))[1:]
        self.ind_tie_windex = ind_tie_vindex+1
        self.ind_tie_vindex = ind_tie_vindex
        # get index of findex for 1&2 case in initial parameters
        for iii_1 in range(len(ind_tie_1)): 
            if  ind_tie_1[iii_1] == True:
                ind_tie_findex1 = np.concatenate([ind_tie_findex1, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii_1])*3)])])
                
        for iii_2 in range(len(ind_tie_2)): 
            if  ind_tie_2[iii_2] == True:
                ind_tie_findex2 = np.concatenate([ind_tie_findex2, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii_2])*3)])])
        #get tied fvalue for case 1 and case 2
        if np.sum(ind_tie_1) > 0:
            self.fvalue_factor_1 = linelist['fvalue'][ind_line][ind_tie_1][1]/\
            linelist['fvalue'][ind_line][ind_tie_1][0]
        if np.sum(ind_tie_2) > 0:
            self.fvalue_factor_2 = linelist['fvalue'][ind_line][ind_tie_2][1]/\
            linelist['fvalue'][ind_line][ind_tie_2][0]
            
        
        self.ind_tie_findex1 = ind_tie_findex1
        self.ind_tie_findex2 = ind_tie_findex2
    
    
    #---------MC error for emission line parameters-------------------
    def _line_mc(self,x,y,err,pp0,pp_limits,n_trails,compcenter):
        """calculate the Monte Carlo errror of line parameters"""
        all_para_1comp = np.zeros(len(pp0)*n_trails).reshape(len(pp0),n_trails)
        all_para_std = np.zeros(len(pp0))
        all_fwhm = np.zeros(n_trails)
        all_sigma = np.zeros(n_trails)
        all_ew = np.zeros(n_trails)
        all_peak = np.zeros(n_trails)
        all_area = np.zeros(n_trails)
     
        for tra in range(n_trails):
            flux = y + np.random.randn(len(y))*err
            line_fit = kmpfit.Fitter(residuals = self._residuals_line, data = (x, flux, err), maxiter = 50)
            line_fit.parinfo = pp_limits
            line_fit.fit(params0 = pp0)
            all_para_1comp[:,tra] = line_fit.params

            #further line properties
            all_fwhm[tra],all_sigma[tra],all_ew[tra],all_peak[tra],all_area[tra] = self.line_prop(compcenter,line_fit.params,'broad')

        for st in range(len(pp0)):
            all_para_std[st] = all_para_1comp[st,:].std()

        return all_para_std,all_fwhm.std(),all_sigma.std(),all_ew.std(),all_peak.std(),all_area.std()

    #-----line properties calculation function--------
    def line_prop(self,compcenter,pp,linetype):
        """
        Calculate the further results for the broad component in emission lines, e.g., FWHM, sigma, peak, line flux
        The compcenter is the theortical vacuum wavelength for the broad compoenet.
        """
        pp = pp.astype(float)
        if linetype == 'broad':
            ind_br = np.repeat(np.where(pp[2::3] > 0.0017,True,False),3)
           
        elif linetype == 'narrow':
            ind_br = np.repeat(np.where(pp[2::3] < 0.0017,True,False),3)
            
        else:
            raise RuntimeError("line type should be 'broad' or 'narrow'!")
            
                
        ind_br[9:] = False # to exclude the broad OIII and broad He II
        
        p = pp[ind_br]
        del pp 
        pp = p


        c = 299792.458 # km/s
        n_gauss = int(len(pp)/3)
        if n_gauss == 0:
            fwhm,sigma,ew,peak,area = 0.,0.,0.,0.,0.
        else:
            cen = np.zeros(n_gauss)
            sig = np.zeros(n_gauss)

            for i in range(n_gauss):
                cen[i] = pp[3*i+1]
                sig[i] = pp[3*i+2]


            #print cen,sig,area
            left = min(cen-3*sig)
            right = max(cen+3*sig)
            disp = 1.e-4
            npix = int((right-left)/disp)


            xx = np.linspace(left, right, npix)
            yy = manygauss(xx,pp)

            ff = interpolate.interp1d(np.log(self.wave),self.PL_poly_BC, bounds_error = False, fill_value = 0)
            contiflux = ff(xx)


            #find the line peak location
            ypeak = yy.max()
            ypeak_ind= np.argmax(yy)
            peak = np.exp(xx[ypeak_ind])

            #find the FWHM in km/s
            #take the broad line we focus and ignore other broad components such as [OIII], HeII


            if n_gauss > 3:
                spline = interpolate.UnivariateSpline(xx, manygauss(xx,pp[0:9])-np.max(manygauss(xx,pp[0:9]))/2, s = 0)
            else:
                spline = interpolate.UnivariateSpline(xx, yy-np.max(yy)/2, s = 0)
            if len(spline.roots()) > 0:
                fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()
                fwhm = abs(np.exp(fwhm_left)-np.exp(fwhm_right))/compcenter*c

                #calculate the total broad line flux
                area = (np.exp(xx)*yy*disp).sum()

                #calculate the line sigma and EW in normal wavelength
                lambda0 = 0.
                lambda1 = 0.
                lambda2 = 0.
                ew = 0
                for lm in range(npix):
                    lambda0 = lambda0 + manygauss(xx[lm],pp)*disp*np.exp(xx[lm])
                    lambda1 = lambda1 + np.exp(xx[lm])*manygauss(xx[lm],pp)*disp*np.exp(xx[lm])
                    lambda2 = lambda2 + np.exp(xx[lm])**2*manygauss(xx[lm],pp)*disp*np.exp(xx[lm])

                    ew = ew + abs(manygauss(xx[lm],pp)/contiflux[lm])*disp*np.exp(xx[lm])

                sigma = np.sqrt(lambda2/lambda0-(lambda1/lambda0)**2) / compcenter*c
            else:
                fwhm,sigma,ew,peak,area = 0.,0.,0.,0.,0.

        return fwhm,sigma,ew,peak,area
    
    def _residuals_line(self, pp, data):
        "The line residual function used in kmpfit"
        xval, yval, weight = data

        # ------tie parameter------------
        if len(self.ind_tie_vindex) > 1 :

            for xx in range(len(self.ind_tie_vindex)-1):
                if self.tie_lambda == True:
                    pp[int(self.ind_tie_vindex[xx+1])] = pp[int(self.ind_tie_vindex[0])]+self.delta_lambda[xx]
                if self.tie_width == True:
                    pp[int(self.ind_tie_windex[xx+1])] = pp[int(self.ind_tie_windex[0])]

        if len(self.ind_tie_findex1) > 0 and self.tie_flux_1 == True:
            pp[int(self.ind_tie_findex1[1])] = pp[int(self.ind_tie_findex1[0])] * self.fvalue_factor_1
        if len(self.ind_tie_findex2) > 0 and self.tie_flux_2 == True:
            pp[int(self.ind_tie_findex2[1])] = pp[int(self.ind_tie_findex2[0])] * self.fvalue_factor_2
        #---------------------------------

        #restore parameters
        self.newpp = pp.copy()
        return (yval - manygauss(xval,pp))/weight

    def _SaveResult(self, conti_result, conti_result_name, line_result, line_result_name, save_fits_path, save_fits_name):
        """Save all data to fits"""
        all_result = np.concatenate([conti_result,line_result])
        all_result_name = np.concatenate([conti_result_name,line_result_name])

        t = Table(all_result, names = (all_result_name))
        t.write(save_fits_path+save_fits_name+'.fits', format = 'fits', overwrite = True)
        
    def _PlotFig(self, ra, dec, z, wave, flux, err, decomposition_host, linefit, tmp_all, gauss_result, f_conti_model, conti_fit,\
               all_comp_range, uniq_linecomp_sort, line_flux, save_fig_path):
        """Plot the results"""

        self.PL_poly = conti_fit.params[6]*(wave/3000.0)**conti_fit.params[7]+f_poly_conti(wave, conti_fit.params[11:])
       
        matplotlib.rc('xtick', labelsize = 20) 
        matplotlib.rc('ytick', labelsize = 20) 
        #plot the first subplot
        fig = plt.figure(figsize = (15,8))
        plt.subplots_adjust(wspace = 3., hspace = 0.2)
        ax = plt.subplot(2,6,(1,6))
        if self.ra == -999. or self.dec == -999.:
            plt.title(str(self.sdss_name)+'   z = '+str(z), fontsize  = 20)
        else:
            plt.title('ra,dec = ('+str(ra)+','+str(dec)+')   '+str(self.sdss_name)+'   z = '+str(z), fontsize = 20)
            
        plt.plot(self.wave_prereduced,self.flux_prereduced,'k',label = 'data')
       
        if decomposition_host == True and self.decomposed == True:
            plt.plot(wave,self.qso+self.host,'pink', label = 'host+qso temp')
            plt.plot(wave,flux,'grey', label = 'data-host')
            plt.plot(wave,self.host,'purple', label = 'host')
        else:
            host = self.flux_prereduced.min()

        plt.scatter(wave[tmp_all],np.repeat(self.flux_prereduced.max()*1.05,len(wave[tmp_all])), color = 'grey', marker = 'o', alpha = 0.5)  # plot continuum region
       

        if linefit == True:
            if self.MC == True:
                
                for p in range(int(len(gauss_result)/6)):
                    if self.CalFWHM(gauss_result[2+p*6],gauss_result[4+p*6]) < 1200.:
                        color = 'g'
                    else:
                        color = 'r'
                    plt.plot(wave,manygauss(np.log(wave),gauss_result[::2][p*3:(p+1)*3])+f_conti_model, color = color)
                plt.plot(wave,manygauss(np.log(wave),gauss_result[::2])+f_conti_model,'b', label = 'line', lw = 2)
            else:
                
                for p in range(int(len(gauss_result)/3)):
                    if self.CalFWHM(gauss_result[3*p+1],gauss_result[3*p+2] ) < 1200.:
                        color = 'g'
                    else:
                        color = 'r'
                    plt.plot(wave,manygauss(np.log(wave),gauss_result[p*3:(p+1)*3])+f_conti_model, color = color)
                plt.plot(wave,manygauss(np.log(wave),gauss_result)+f_conti_model,'b', label = 'line', lw = 2)
        plt.plot([0,0],[0,0],'r', label = 'line br')
        plt.plot([0,0],[0,0],'g', label = 'line na')
        plt.plot(wave,f_conti_model,'c',lw=2, label = 'FeII')
        if self.BC == True:
            plt.plot(wave,self.f_pl_model+self.f_poly_model+self.f_bc_model,'y',lw = 2, label = 'BC')
        plt.plot(wave,conti_fit.params[6]*(wave/3000.0)**conti_fit.params[7]+f_poly_conti(wave, conti_fit.params[11:]), color = 'orange', lw = 2, label = 'conti')
        if self.decomposed == False:
            self.host = self.flux_prereduced.min()
        plt.ylim(min(self.host.min(),flux.min())*0.9,self.flux_prereduced.max()*1.1)

        
        if self.plot_legend == True:
            plt.legend(loc = 'best', frameon = False, fontsize = 10)
       
        
        #plot line name--------
        if self.plot_line_name == True:
            line_cen = np.array([6564.60,   6549.85,  6585.27,  6718.29,  6732.66,  4862.68,  5008.24,  4687.02,\
                   4341.68,   3934.78,  3728.47,  3426.84,  2798.75,  1908.72,  1816.97,\
                   1750.26,   1718.55,  1549.06,  1640.42,  1402.06,  1396.76,  1335.30,\
                   1215.67])
   
            line_name = np.array(['',  '', 'Ha+NII', '', 'SII6718,6732', 'Hb', '[OIII]',\
                    'HeII4687','Hr','CaII3934', 'OII3728', 'NeV3426', 'MgII','CIII]',\
                    'SiII1816', 'NIII1750', 'NIV1718', 'CIV', 'HeII1640','',\
                    'SiIV+OIV', 'CII1335','Lya'])
        
            for ll in range(len(line_cen)):
                if  wave.min() < line_cen[ll] < wave.max():
                    plt.plot([line_cen[ll],line_cen[ll]],[min(self.host.min(),flux.min()),self.flux_prereduced.max()*1.1],'k:')
                    plt.text(line_cen[ll]+10,1.*self.flux_prereduced.max(),line_name[ll], rotation = 90, fontsize = 15)
 
    
        plt.xlim(wave.min(),wave.max())
       
        if linefit == True:
            #plot subplot from 2 to N
            for c in range(self.ncomp):
                if self.ncomp == 4:
                    axn = plt.subplot(2,12,(12+3*c+1,12+3*c+3))
                if self.ncomp == 3:
                    axn = plt.subplot(2,12,(12+4*c+1,12+4*c+4))
                if self.ncomp == 2:
                    axn = plt.subplot(2,12,(12+6*c+1,12+6*c+6))
                if self.ncomp == 1:
                    axn = plt.subplot(2,12,(13,24))
                plt.plot(wave,self.line_flux,'k')



                if self.MC == True:
                    for p in range(int(len(gauss_result)/6)):
                        if self.CalFWHM(gauss_result[2+p*6],gauss_result[4+p*6] ) < 1200.:
                            color = 'g'
                        else:
                            color = 'r'
                        plt.plot(wave,manygauss(np.log(wave),gauss_result[::2][p*3:(p+1)*3]), color = color)
                    plt.plot(wave,manygauss(np.log(wave),gauss_result[::2]),'b')
                else:
                    for p in range(int(len(gauss_result)/3)):
                        if self.CalFWHM(gauss_result[3*p+1],gauss_result[3*p+2] ) < 1200.:
                            color = 'g'
                        else:
                            color = 'r'
                        plt.plot(wave,onegauss(np.log(wave),gauss_result[p*3:(p+1)*3]), color = color)
                    plt.plot(wave,manygauss(np.log(wave),gauss_result),'b')
                plt.xlim(all_comp_range[2*c:2*c+2])
                f_max = line_flux[np.where( (wave > all_comp_range[2*c] ) & ( wave < all_comp_range[2*c+1] ),True,False)].max()
                f_min = line_flux[np.where( (wave > all_comp_range[2*c] ) & ( wave < all_comp_range[2*c+1] ),True,False)].min()
                plt.ylim(f_min*0.9,f_max*1.1)
                axn.set_xticks([all_comp_range[2*c],np.round((all_comp_range[2*c]+all_comp_range[2*c+1])/2,-1),all_comp_range[2*c+1]])
                plt.text(0.02,0.9, uniq_linecomp_sort[c], fontsize = 20, transform = axn.transAxes)
        
        if linefit == True:    
            plt.text(0.4,-1.4, r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20, transform = ax.transAxes)
            plt.text(-0.1,0.5, r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)',fontsize = 20, transform = ax.transAxes, rotation = 90)
        else:
            plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20)
            plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize = 20)
        
        if self.save_fig == True:
            plt.savefig(save_fig_path+self.sdss_name+'.eps')
            
    def CalFWHM(self,logwave,logsigma):
        """transfer the logFWHM to normal frame"""
        return (np.exp(logwave+logsigma)-np.exp(logwave))/np.exp(logwave)*300000.*2.35

    def Onegauss(self, xval, pp):
        """The single Gaussian model used to fit the emission lines 
        Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave
        """

        term1 = np.exp( - (xval - pp[1])**2 / (2. * pp[2]**2) )
        yval = pp[0] * term1 / (np.sqrt(2.*np.pi) * pp[2])
        return yval

    def Manygauss(self, xval, pp):
        """The multi-Gaussian model used to fit the emission lines, it will call the onegauss function"""
        ngauss = int(pp.shape[0]/3)
        if ngauss != 0:
            yval = 0.
            for i in range(ngauss):
                yval = yval + onegauss(xval, pp[i*3:(i+1)*3])
            return yval
    
    



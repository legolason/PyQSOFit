# ppxf to fit the host component from PyQSOFit
# combine two exmaples from ppxf, population_gas_sdss and kinematics_sdss
# Version 1.2
# 10/27/2022

import glob, os
from os import path
from time import clock
from astropy.io import fits
from scipy import ndimage
import numpy as np
import ppxf as ppxf_package
from astropy import units as u
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import matplotlib.pyplot as plt
import ppxf.miles_util as lib
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")


class PPXF():
    def __init__(self):
        zzz=0

    def population_gas(self, wv_rest, fx,err, wv_min = None, wv_max = None):

        ppxf_dir = path.dirname(path.realpath(lib.__file__))

        if wv_min is None:
            wv_min = np.min(wv_rest)
        if wv_max is None:
            wv_max = np.max(wv_rest)

        mask = np.where((wv_rest > wv_min) & (wv_rest < wv_max) & (err < 100) & (fx > 0), True, False)   

        wave = wv_rest[mask]
        galaxy = fx[mask]/np.median(fx[mask])   # Normalize spectrum to avoid numerical issues
        # The SDSS wavelengths are in vacuum, while the MILES ones are in air.
        # For a rigorous treatment, the SDSS vacuum wavelengths should be
        # converted into air wavelengths and the spectra should be resampled.
        # To avoid resampling, given that the wavelength dependence of the
        # correction is very weak, I approximate it with a constant factor.
        #
        wave *= np.median(util.vac_to_air(wave)/wave)

        # The noise level is chosen to give Chi^2/DOF=1 without regularization (regul=0).
        # A constant noise is not a bad approximation in the fitted wavelength
        # range and reduces the noise in the fit.
        #
        noise = np.full_like(galaxy, err[mask].mean())  # Assume constant noise per pixel here

        # The velocity step was already chosen by the SDSS pipeline
        # and we convert it below to km/s
        #
        c = 299792.458  # speed of light in km/s
        #velscale = c*np.log(wave[1]/wave[0])  # eq.(8) of Cappellari (2017)
        velscale = c*np.diff(np.log(wave[[0, -1]]))/(wave.size - 1) # 69 km/s for SDSS
        FWHM_gal = 2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.
        z = 1e-5
        #------------------- Setup templates -----------------------

        pathname = ppxf_dir + '/miles_models/Eun1.30Z*.fits'

        # The templates are normalized to the V-band using norm_range. In this way
        # the weights returned by pPXF represent V-band light fractions of each SSP.
        miles = lib.miles(pathname, velscale, FWHM_gal, norm_range=[5070, 5950])

        # The stellar templates are reshaped below into a 2-dim array with each
        # spectrum as a column, however we save the original array dimensions,
        # which are needed to specify the regularization dimensions
        #
        reg_dim = miles.templates.shape[1:]
        stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

        # See the pPXF documentation for the keyword `regul`.
        # A regularization error of a few percent is a good start
        regul_err = 0.02

        # Estimate the wavelength fitted range in the rest frame.
        lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + z)

        # Construct a set of Gaussian emission line templates.
        # The `emission_lines` function defines the most common lines, but additional
        # lines can be included by editing the function in the file ppxf_util.py.
        gas_templates, gas_names, line_wave = util.emission_lines(
            miles.ln_lam_temp, lam_range_gal, FWHM_gal,
            tie_balmer=False, limit_doublets=False)

        # Combines the stellar and gaseous templates into a single array.
        # During the pPXF fit they will be assigned a different kinematic
        # COMPONENT value
        #
        templates = np.column_stack([stars_templates, gas_templates])

        #-----------------------------------------------------------

        vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
        start1 = [vel, 180.]     # (km/s), starting guess for [V, sigma]

        n_temps = stars_templates.shape[1]
        n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
        n_balmer = len(gas_names) - n_forbidden

        # Assign component=0 to the stellar templates, component=1 to the Balmer
        # gas emission lines templates and component=2 to the forbidden lines.
        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        gas_component = np.array(component) > 0  # gas_component=True for gas templates

        # Fit (V, sig, h3, h4) moments=4 for the stars
        # and (V, sig) moments=2 for the two gas kinematic components
        moments = [4, 2, 2]

        # Adopt the same starting value for the stars and the two gas components
        start = [start1, start1, start1]

        # If the Balmer lines are tied one should allow for gas reddeining.
        # The gas_reddening can be different from the stellar one, if both are fitted.
        gas_reddening = 0 if False else None

        # Here the actual fit starts.
        #
        # IMPORTANT: Ideally one would like not to use any polynomial in the fit
        # as the continuum shape contains important information on the population.
        # Unfortunately this is often not feasible, due to small calibration
        # uncertainties in the spectral shape. To avoid affecting the line strength of
        # the spectral features, we exclude additive polynomials (degree=-1) and only use
        # multiplicative ones (mdegree=10). This is only recommended for population, not
        # for kinematic extraction, where additive polynomials are always recommended.
        #
        t = clock()

        pp = ppxf(templates, galaxy, noise, velscale, start,
                  moments=moments, degree=-1, mdegree=10,
                  lam=wave, lam_temp=miles.lam_temp,
                  regul=1/regul_err, reg_dim=reg_dim,
                  component=component, gas_component=gas_component,
                  gas_names=gas_names, gas_reddening=gas_reddening,quiet=True)

        '''
        # When the two Delta Chi^2 below are the same, the solution
        # is the smoothest consistent with the observed spectrum.
        print(f"Desired Delta Chi^2: {np.sqrt(2*galaxy.size):#.4g}")
        print(f"Current Delta Chi^2: {(pp.chi2 - 1)*galaxy.size:#.4g}")
        print(f"Elapsed time in pPXF: {(clock() - t):.2f}")

        light_weights = pp.weights[~gas_component]      # Exclude weights of the gas templates
        light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
        light_weights /= light_weights.sum()            # Normalize to light fractions

        # Given that the templates are normalized to the V-band, the pPXF weights
        # represent v-band light fractions and the computed ages and metallicities
        # below are also light weighted in the V-band.
        miles.mean_age_metal(light_weights)

        # For the M/L one needs to input fractional masses, not light fractions.
        # For this, I convert light-fractions into mass-fractions using miles.flux
        mass_weights = light_weights/miles.flux
        mass_weights /= mass_weights.sum()              # Normalize to mass fractions
        miles.mass_to_light(mass_weights, band="r")
        '''

        sigma = np.round(pp.sol[0][1],1)
        sigmaerr = np.round(pp.error[0][1]*np.sqrt(pp.chi2), 1)

        # Plot fit results for stars and gas.
        plt.clf()
        ax=plt.subplot(111)
        pp.plot()
        plt.text(0.02, 0.85,r'$\sigma$='+str(sigma)+'$\pm$'+str(sigmaerr)+r'$\rm \ km\ s^{-1}$', 
                         transform=ax.transAxes,fontsize=20)

        # Plot stellar population mass-fraction distribution
        #plt.subplot(212)
        #miles.plot(light_weights)
        #plt.tight_layout()
        #print(pp.error,np.sqrt(pp.chi2) )
        return pp
    
    
    def kinematics(self, wv_rest, fx, err, wdisp = None, wv_min = None, wv_max = None, qso_width_mask = False):

        ppxf_dir = path.dirname(path.realpath(util.__file__))

        if wv_min is None:
            wv_min = np.min(wv_rest)
        if wv_max is None:
            wv_max = np.max(wv_rest)

        mask = np.where((wv_rest > wv_min) & (wv_rest < wv_max) & (err < 100) & (fx > 0), True, False)    

        log_wv_rest = np.log10(wv_rest)
        

        galaxy = fx[mask]/np.median(fx[mask])     # Normalize spectrum to avoid numerical issues
        ln_lam_gal = log_wv_rest[mask]*np.log(10)         # Convert lg --> ln
        lam_gal = np.exp(ln_lam_gal)



        d_ln_lam_gal = np.diff(ln_lam_gal[[0, -1]])/(ln_lam_gal.size -1)  # Use full lam range for accuracy

        c = 299792.458                              # speed of light in km/s
        velscale = c*d_ln_lam_gal                   # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
        noise = np.full_like(galaxy, err[mask].mean())        # Assume constant noise per pixel here

        # Compue instrumental FWHM for every pixel in Angstrom
        dlam_gal = np.diff(lam_gal)                 # Size of every pixel in Angstrom
        dlam_gal = np.append(dlam_gal, dlam_gal[-1])
        if wdisp is None:
            wdisp = np.ones_like(wv_rest[mask])            # Intrinsic dispersion of every pixel, in pixels units
        else:
            wdisp = wdisp[mask] 

        fwhm_gal = 2.355*wdisp*dlam_gal             # Resolution FWHM of every pixel, in Angstroms
        redshift = 0    
    
        # If the galaxy is at significant redshift, it may be easier to bring the
        # galaxy spectrum roughly to the rest-frame wavelength, before calling
        # pPXF (See Sec.2.4 of Cappellari 2017). In practice there is no
        # need to modify the spectrum in any way, given that a red shift
        # corresponds to a linear shift of the log-rebinned spectrum.
        # One just needs to compute the wavelength range in the rest-frame
        # and adjust the instrumental resolution of the galaxy observations.
        # This is done with the following three commented lines:
        #
        #lam_gal = lam_gal/(1 + z)     # Compute approximate restframe wavelength
        #fwhm_gal = fwhm_gal/(1 + z)   # Adjust resolution in Angstrom
        #z = 0                         # Spectrum is now in rest-frame

        # Read the list of filenames from the Single Stellar Population library
        # by Vazdekis (2016, MNRAS, 463, 3409) http://miles.iac.es/. A subset
        # of the library is included for this example with permission
        vazdekis = glob.glob(ppxf_dir + '/miles_models/Eun1.30Z*.fits')
        fwhm_tem = 2.51 # Vazdekis+16 spectra have a constant resolution FWHM of 2.51A.

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the SDSS galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        #
        hdu = fits.open(vazdekis[0])
        ssp = hdu[0].data
        h2 = hdu[0].header

        # The E-Miles templates span a large wavelength range. To save some
        # computation time I truncate the spectra to a similar range as the galaxy.
        lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
        good_lam = (lam_temp > 2000) & (lam_temp < 1e4)
        lam_temp = lam_temp[good_lam]
        lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]

        sspNew, ln_lam_temp = util.log_rebin(lamRange_temp, ssp[good_lam], velscale=velscale)[:2]
        templates = np.empty((sspNew.size, len(vazdekis)))

        # Interpolates the galaxy spectral resolution at the location of every pixel
        # of the templates. Outside the range of the galaxy spectrum the resolution
        # will be extrapolated, but this is irrelevant as those pixels cannot be
        # used in the fit anyway.
        fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)
        
        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the SDSS and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> SDSS
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        #
        # In the line below, the fwhm_dif is set to zero when fwhm_gal < fwhm_tem.
        # In principle it should never happen and a higher resolution template should be used.
        #
        
        fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
        if np.isnan(fwhm_dif[0]):
            fwhm_dif = 0
        
        sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels
        
        
        for j, fname in enumerate(vazdekis):
            hdu = fits.open(fname)
            ssp = hdu[0].data
            ssp = util.gaussian_filter1d(ssp[good_lam], sigma)  # perform convolution with variable sigma
            sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
            templates[:, j] = sspNew/np.median(sspNew[sspNew > 0]) # Normalizes templates
        
        width_qso = [800,800,2000,2000,3000,800,800,800,800,800,3000,800,800]
        if qso_width_mask == True:
            goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, redshift, width = width_qso)
        else:
            goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, redshift, width = 800)

        # Here the actual fit starts. The best fit is plotted on the screen.
        # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
        #
        c = 299792.458   # km/s
        vel = c*np.log(1 + redshift)   # eq.(8) of Cappellari (2017)
        start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
        t = clock()

        #bounds = [[-300, 300], [20, 100],[-0.3, 0.3], [-0.3, 0.3]]

        pp = ppxf(templates, galaxy, noise, velscale, start, bounds=None,
                  goodpixels=goodpixels, plot=True, moments=4, trig=1,
                  degree=20, lam=lam_gal, lam_temp=np.exp(ln_lam_temp),quiet = True)

        # The updated best-fitting redshift is given by the following
        # lines (using equation 8 of Cappellari 2017, MNRAS)
        vcosm = c*np.log(1 + redshift)  # This is the initial redshift estimate
        vpec = pp.sol[0]          # This is the fitted residual velocity
        vtot = vcosm + vpec       # I add the two velocities before computing z
        redshift_best = np.exp(vtot/c) - 1          # eq.(8) Cappellari (2017)
        errors = pp.error*np.sqrt(pp.chi2)          # Assume the fit is good
        redshift_err = np.exp(vtot/c)*errors[0]/c   # Error propagation

      
        sigma = np.round(pp.sol[1],1)
        sigmaerr = np.round(pp.error[1]*np.sqrt(pp.chi2), 1)

        plt.clf()
        ax=plt.subplot(111)
        pp.plot()
        plt.text(0.02, 0.85,r'$\sigma$='+str(sigma)+'$\pm$'+str(sigmaerr)+r'$\rm \ km\ s^{-1}$', 
                         transform=ax.transAxes,fontsize=20)
        return pp

    def set_mpl_style(fsize=20, tsize=18, tdir='in', major=5.0, minor=3.0, lwidth=1.8, lhandle=2.0):

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


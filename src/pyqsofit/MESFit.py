import glob, os, sys, timeit

import astropy.io.fits
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sfdmap2 import sfdmap
import pandas as pd
from scipy import interpolate
from lmfit import minimize, Parameters, report_fit
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from os import path
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
import warnings
import matplotlib.backends.backend_pdf as bpdf
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")


class MESFit():
    "Multi-epoch Spectral Fitting program"

    def __init__(self, path=None):

        self.path = path
        if self.path is None:
            raise RuntimeError("Wrong directory!")

    def Coadd_spec(self):
        count = 0
        wv = np.logspace(np.log10(3600.), np.log10(3600.) + 1.e-4 * 4649, 4650)
        # wave st = 3800,  step = 1.e-4, npixel = 4650
        fx_all = np.zeros_like(wv)
        err_all = np.zeros_like(wv)
        ivar_all = np.zeros_like(wv)
        count = np.zeros_like(wv)
        nepoch = 0
        for i in glob.glob(os.path.join(self.path, "*.fits")):
            spec = fits.open(i)
            nepoch = nepoch + 1
            if nepoch == 1:
                z = spec[2].data['z'][0]
            wave = 10 ** spec[1].data['loglam']
            flux = spec[1].data['flux']
            error = np.clip(np.sqrt(1. / spec[1].data['ivar']), 0, 10)
            f = interpolate.interp1d(wave, flux, bounds_error=False, fill_value=0.)
            fx = f(wv)  # new flux of each spectrum
            f = interpolate.interp1d(wave, error, bounds_error=False, fill_value=1.)
            err = f(wv)
            fx_all = fx_all + fx * (1. / err ** 2)
            err_all = err_all + err
            ivar_all = ivar_all + 1. / err ** 2
            ind = np.where(fx != 0., True, False)
            count[ind] = count[ind] + 1
        self.mean_wv = wv / (1. + z)
        self.mean_spec = fx_all / ivar_all  # Ivar weighted mean spec
        self.mean_err = err_all / count
        self.z = z
        self.nepoch = nepoch

        # ------rms----------
        count = np.zeros_like(wv)
        flux_all = np.zeros_like(self.mean_wv)
        err_all = np.zeros_like(self.mean_wv)
        for i in glob.glob(os.path.join(self.path, "*.fits")):
            spec = fits.open(i)
            wave = 10 ** spec[1].data['loglam']
            flux = spec[1].data['flux']
            error = np.clip(np.sqrt(1. / spec[1].data['ivar']), 0, 10)
            f = interpolate.interp1d(wave, flux, bounds_error=False, fill_value=self.mean_spec.mean())
            fx = f(wv)  # new flux of each spectrum
            f = interpolate.interp1d(wave, error, bounds_error=False, fill_value=self.mean_err.mean())
            err = f(wv)
            flux_all = flux_all + (fx - self.mean_spec) ** 2
            err_all = err_all + (err - self.mean_err) ** 2
            ind = np.where(fx != 0., True, False)
            count[ind] = count[ind] + 1
        self.rms_err = np.sqrt(err_all) / count
        self.rms_spec = np.sqrt(flux_all) / count

    def residuals_norm(self, p, mean_wv, mean_spec, flux_new):
        factor, sigma, shift = list(p.valuesdict().values())
        center = self.norm_line_wv
        wv_max = center + 40.
        wv_min = center - 40.
        ind = np.where((mean_wv < wv_max) & (mean_wv > wv_min), True, False)
        model = factor * (gaussian_filter1d(flux_new[ind], sigma=sigma) - shift)
        return (mean_spec[ind] - model)

    def normNL(self, norm_line_wv=5008.24, path=None, name='normNL', save=True, tol=1e-10):
        self.norm_line_wv = norm_line_wv
        plate_all = []
        mjd_all = []
        fiber_all = []
        factor_all = []
        sigma_all = []
        shift_all = []
        flux_all = []
        err_all = []
        f5100 = []
        f3000 = []
        f1350 = []
        c = 0
        wv = np.logspace(np.log10(3600.), np.log10(3600.) + 1.e-4 * 4649, 4650)
        for i in glob.glob(os.path.join(self.path, "*.fits")):
            c = c + 1
            spec = fits.open(i)
            if c == 1:
                z = spec[2].data['z'][0]
                ra = spec[2].data['plug_ra'][0]
                dec = spec[2].data['plug_dec'][0]
            plate = spec[2].data['plate'][0]
            mjd = spec[2].data['mjd'][0]
            fiber = spec[2].data['fiberid'][0]
            wave = 10 ** spec[1].data['loglam']
            flux = spec[1].data['flux']
            error = np.clip(np.sqrt(1. / spec[1].data['ivar']), 0, 10)

            f = interpolate.interp1d(wave, flux, bounds_error=False, fill_value=0.)
            flux_new = f(wv)
            f = interpolate.interp1d(wave, error, bounds_error=False, fill_value=0.)
            error_new = f(wv)

            # ----fitting-------
            p0 = [1., 1., 0.]
            fit_params = Parameters()
            fit_params.add('factor', value=p0[0], min=0.)
            fit_params.add('sigma', value=p0[1], min=0., max=10.)
            fit_params.add('shift', value=p0[2])
            fitobj = minimize(self.residuals_norm, fit_params, args=(self.mean_wv, self.mean_spec, flux_new),
                              calc_covar=False, xtol=tol, ftol=tol)
            params = list(fitobj.params.valuesdict().values())

            flux_norm = params[0] * (gaussian_filter1d(flux_new, params[1]) - params[2])
            err_norm = params[0] * (gaussian_filter1d(error_new, params[1]))  # did not add shift
            # ----save data-----
            plate_all.append(plate)
            mjd_all.append(mjd)
            fiber_all.append(fiber)
            factor_all.append(params[0])
            sigma_all.append(params[1])
            shift_all.append(params[2])
            flux_all.append(flux_norm)
            err_all.append(err_norm)
            ind = np.where((wave / (1 + z) < 5125) & (wave / (1 + z) > 5075), True, False)
            if np.sum(ind) > 0:
                f5100.append(flux[ind].mean())
            else:
                f5100.append(0)
            ind = np.where((wave / (1 + z) < 3060) & (wave / (1 + z) > 3000), True, False)
            if np.sum(ind) > 0:
                f3000.append(flux[ind].mean())
            else:
                f3000.append(0)
            ind = np.where((wave / (1 + z) < 1375) & (wave / (1 + z) > 1325), True, False)
            if np.sum(ind) > 0:
                f1350.append(flux[ind].mean())
            else:
                f1350.append(flux[ind].mean())

        plate_all = np.array(plate_all)
        mjd_all = np.array(mjd_all)
        fiber_all = np.array(fiber_all)
        factor_all = np.array(factor_all)
        sigma_all = np.array(sigma_all)
        shift_all = np.array(shift_all)
        f5100 = np.array(f5100)
        f3000 = np.array(f3000)
        f1350 = np.array(f1350)
        flux_all = np.array(flux_all)
        err_all = np.array(err_all)

        # sort all epoch with mjd
        indsort = np.argsort(mjd_all)
        if save == True:
            # -------save data-------------
            hdr1 = fits.Header()
            hdr1['RA'] = (ra, 'Plug_ra in degree')
            hdr1['DEC'] = (dec, 'Plug_dec in degree')
            hdr1['z'] = (np.round(z, 6), 'Redshift')
            hdr1['epoch'] = (c, 'Number of epoch')
            hdr1['sigma'] = ('', 'sigma in unit of (A)')
            hdr1['shift'] = ('', 'shift in unit of (erg/s/cm^2/A)')
            hdr1['flambda'] = (',''f5100,f3000,f1350 in unit of (erg/s/cm^2/A)')
            hdr1['EXTNAME'] = 'PARAMETERS'
            c1 = fits.Column(name='plate', array=plate_all[indsort], format='I')
            c2 = fits.Column(name='mjd', array=mjd_all[indsort], format='K')
            c3 = fits.Column(name='fiber', array=fiber_all[indsort], format='I')
            c4 = fits.Column(name='factor', array=factor_all[indsort], format='E')
            c5 = fits.Column(name='sigma', array=sigma_all[indsort], format='E')
            c6 = fits.Column(name='shift', array=shift_all[indsort], format='E')
            c7 = fits.Column(name='f5100', array=f5100[indsort], format='E')
            c8 = fits.Column(name='f3000', array=f3000[indsort], format='E')
            c9 = fits.Column(name='f1350', array=f1350[indsort], format='E')
            h = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9], header=hdr1)
            filename = os.path.join(path, name + '.fits')
            h.writeto(filename, overwrite=True)

            hdr2 = fits.Header()
            hdr2['EXTNAME'] = 'COADDSPEC'
            hdr2['COEEF0'] = (np.log10(3600.), 'Central wavelength (log10) of first pixel')
            hdr2['COEEF1'] = (0.0001, 'Log10 dispersion per pixel')
            hdr2['NPIXEL'] = (4650, 'Pixel number of wavelength')
            data = np.vstack((self.mean_spec, self.mean_err, self.rms_spec, self.rms_err))
            h = fits.BinTableHDU.from_columns([c1, c2, c3, c4], header=hdr2)
            fits.append(filename=filename, data=data, header=hdr2)

            hdr3 = fits.Header()
            hdr3['EXTNAME'] = 'FLUX'
            hdr3['COEEF0'] = (np.log10(3600.), 'Central wavelength (log10) of first pixel')
            hdr3['COEEF1'] = (0.0001, 'Log10 dispersion per pixel')
            hdr3['NPIXEL'] = (4650, 'Pixel number of wavelength')
            fits.append(filename=filename, data=flux_all[indsort, :], header=hdr3)
            hdr4 = fits.Header()
            hdr4['EXTNAME'] = 'ERROR'
            hdr4['COEEF0'] = (np.log10(3600.), 'Central wavelength (log10) of first pixel')
            hdr4['COEEF1'] = (0.0001, 'Log10 dispersion per pixel')
            hdr4['NPIXEL'] = (4650, 'Pixel number of wavelength')
            fits.append(filename=filename, data=err_all[indsort, :], header=hdr4)

    def fixNL(self, pardata: astropy.io.fits.hdu.table.BinTableHDU = None, fitdata: dict = None):
        '''
        For each line group, we fix the line sigma and scale of narrow lines to the results of mean spectra.
        :param pardata:
        :param fitdata:
        :return:
        '''
        comp_name = [nm for nm in fitdata.keys() if 'complex_name' in nm]
        available_group = [fitdata[comp_nm] for comp_nm in comp_name]
        na_line_idx = np.where((pardata.data['maxsig']<0.0018)&(np.isin(pardata.data['compname'], available_group)))
        for na_idx in np.squeeze(na_line_idx):
            line_name = pardata.data['linename'][na_idx]
            pardata.data['inisca'][na_idx] = fitdata[f'{line_name}_1_scale']
            pardata.data['inisig'][na_idx] = fitdata[f'{line_name}_1_sigma']
            pardata.data['vary'][na_idx] = 0.
        return pardata

    def ppxf_host(self, rest_wv, fx, error, redshift, wv_min=4125, wv_max=5350, plot=None, quiet=False,
                  mask_width=3000):
        "fit the host galaxy to get the stellar velocity dispersion. "

        ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
        # Only use the wavelength range in common between galaxy and stellar library.
        #
        mask = np.where((rest_wv > wv_min) & (rest_wv < wv_max) & (error < 100.), True, False)

        flux = fx[mask]
        galaxy = flux / np.median(flux)  # Normalize spectrum to avoid numerical issues
        wave = rest_wv[mask]
        err = error[mask]
        z = 0  # already rest frame so z =0

        # The SDSS wavelengths are in vacuum, while the MILES ones are in air.
        # For a rigorous treatment, the SDSS vacuum wavelengths should be
        # converted into air wavelengths and the spectra should be resampled.
        # To avoid resampling, given that the wavelength dependence of the
        # correction is very weak, I approximate it with a constant factor.
        #
        wave *= np.median(util.vac_to_air(wave) / wave)

        # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0).
        # A constant noise is not a bad approximation in the fitted wavelength
        # range and reduces the noise in the fit.
        #
        noise = np.full_like(galaxy, err.mean())  # Assume constant noise per pixel here

        # The velocity step was already chosen by the SDSS pipeline
        # and we convert it below to km/s
        #
        c = 299792.458  # speed of light in km/s
        velscale = c * np.log(wave[1] / wave[0])  # eq.(8) of Cappellari (2017)
        FWHM_gal = 2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.

        # ------------------- Setup templates -----------------------

        pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
        miles = lib.miles(pathname, velscale, FWHM_gal)

        # The stellar templates are reshaped below into a 2-dim array with each
        # spectrum as a column, however we save the original array dimensions,
        # which are needed to specify the regularization dimensions
        #
        reg_dim = miles.templates.shape[1:]
        stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

        # See the pPXF documentation for the keyword REGUL,
        regul_err = 0.013  # Desired regularization error

        # Construct a set of Gaussian emission line templates.
        # Estimate the wavelength fitted range in the rest frame.
        #
        lam_range_gal = np.array([np.min(wave), np.max(wave)]) / (1. + z)
        gas_templates, gas_names, line_wave = util.emission_lines(
            miles.ln_lam_temp, lam_range_gal, FWHM_gal,
            tie_balmer=False, limit_doublets=False)

        # Combines the stellar and gaseous templates into a single array.
        # During the PPXF fit they will be assigned a different kinematic
        # COMPONENT value
        #

        templates = np.column_stack([stars_templates, gas_templates])

        # -----------------------------------------------------------

        # The galaxy and the template spectra do not have the same starting wavelength.
        # For this reason an extra velocity shift DV has to be applied to the template
        # to fit the galaxy spectrum. We remove this artificial shift by using the
        # keyword VSYST in the call to PPXF below, so that all velocities are
        # measured with respect to DV. This assume the redshift is negligible.
        # In the case of a high-redshift galaxy one should de-redshift its
        # wavelength to the rest frame before using the line below as described
        # in PPXF_EXAMPLE_KINEMATICS_SAURON and Sec.2.4 of Cappellari (2017)
        #
        c = 299792.458
        dv = c * (miles.ln_lam_temp[0] - np.log(wave[0]))  # eq.(8) of Cappellari (2017)
        vel = c * np.log(1 + z)  # eq.(8) of Cappellari (2017)
        start = [vel, 180.]  # (km/s), starting guess for [V, sigma]

        n_temps = stars_templates.shape[1]
        n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
        n_balmer = len(gas_names) - n_forbidden

        # Assign component=0 to the stellar templates, component=1 to the Balmer
        # gas emission lines templates and component=2 to the forbidden lines.
        component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
        gas_component = np.array(component) > 0  # gas_component=True for gas templates

        # Fit (V, sig, h3, h4) moments=4 for the stars
        # and (V, sig) moments=2 for the two gas kinematic components
        moments = [4, 4, 4]

        # Adopt the same starting value for the stars and the two gas components
        start = [start, start, start]

        # If the Balmer lines are tied one should allow for gas reddeining.
        # The gas_reddening can be different from the stellar one, if both are fitted.
        # gas_reddening = 0 if tie_balmer else None

        # TODO: ppxf has update its version and delete the qso switch. We can only use the width to eliminate the
        #  mask range. We might consider to wright a function specifically masking quasar spectra.
        goodpixels = util.determine_goodpixels(np.log(wave), lam_range_gal, z, width=mask_width)
        # Here the actual fit starts.
        #
        # IMPORTANT: Ideally one would like not to use any polynomial in the fit
        # as the continuum shape contains important information on the population.
        # Unfortunately this is often not feasible, due to small calibration
        # uncertainties in the spectral shape. To avoid affecting the line strength of
        # the spectral features, we exclude additive polynomials (DEGREE=-1) and only use
        # multiplicative ones (MDEGREE=10). This is only recommended for population, not
        # for kinematic extraction, where additive polynomials are always recommended.
        #

        pp = ppxf(templates, galaxy, noise, velscale, start,
                  plot=False, moments=moments, degree=-1, mdegree=10, vsyst=dv,
                  lam=wave, clean=False, regul=1. / regul_err, reg_dim=reg_dim,
                  component=component, gas_component=gas_component,
                  gas_names=gas_names, goodpixels=goodpixels, quiet=quiet)

        weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
        weights = weights.reshape(reg_dim) / weights.sum()  # Normalized

        # calculate the band luminosity e.g., Lr
        sdss_filter = np.array([3543, 4770, 6231, 7625])
        band = np.array(['u', 'g', 'r', 'i'])
        filter_cen = sdss_filter[(np.abs(sdss_filter - (wave * (1 + redshift)).mean())).argmin()]
        bandn = band[(np.abs(sdss_filter - (wave * (1 + redshift)).mean())).argmin()]
        # assume the filter width ~ 1000A
        ind = np.where((wave * (1 + redshift) > filter_cen - 500) & (wave * (1 + redshift) < filter_cen + 500), True,
                       False)
        flux_band_mean = flux[ind][np.isfinite(flux[ind])].mean()
        flux_band_mean_up = (flux[ind] + err[ind])[np.isfinite(flux[ind])].mean()
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        DL = cosmo.luminosity_distance(redshift).value * 10 ** 6 * 3.08 * 10 ** 18  # unit cm
        Lband = np.log10(
            4 * np.pi * DL ** 2 * flux_band_mean * 1.e-17 * 1000)  # assume the band widths for sdss are ~ 1000A
        Lband_up = np.log10(4 * np.pi * DL ** 2 * flux_band_mean_up * 1.e-17 * 1000)

        # calculate the SFH and error
        mass_frac_tmp = np.asarray([np.sum(weights[0:3, :])])  # mass fraction below 10^8 yr
        mass_frac_tmp_delta = np.asarray([np.sum(weights[0:4, :]) - np.sum(weights[0:3, :])])
        ml_tmp = np.asarray([miles.mass_to_light(weights, band=bandn)])

        sigma = np.round(pp.sol[0][1], 1)
        sigmaerr = np.round(pp.error[0][1] * np.sqrt(pp.chi2), 1)
        stellar_mass = np.round(10 ** Lband / (3.8 * 10 ** 33) / ml_tmp, 2)  # M_sun
        stellar_mass_err = np.round(10 ** Lband_up / (3.8 * 10 ** 33) / ml_tmp - stellar_mass, 2)
        SFR = np.round(stellar_mass * mass_frac_tmp / 10 ** 8, 2)  # M_sun/yr
        SFR_err = np.round(stellar_mass * mass_frac_tmp_delta / 10 ** 8, 2)  # M_sun/yr

        # --------plot figure-----
        if plot == True:
            fig = plt.figure(figsize=(15, 6))
            ax = plt.subplot(111)
            pp.plot()
            plt.text(0.02, 0.85, r'$\sigma$=' + str(sigma) + '$\pm$' + str(sigmaerr) + r'$\rm{~km\ s^{-1}}$',
                     transform=ax.transAxes, fontsize=20)
            # plt.subplot(212)
            # miles.plot(weights)
            plt.tight_layout()
            # plt.savefig('')
        # ------save data------

        self.data_host = np.array([np.squeeze(data) for data in
                                   [sigma, sigmaerr, stellar_mass, stellar_mass, stellar_mass_err, SFR, SFR_err]])
        self.data_host_name = np.array(['sigma', 'sigmaerr', 'stellar_mass', 'stellar_mass_err', 'SFR', 'SFR_err'])

    def corr(self, w1, f1, err1, w_ref, f_ref, err_ref, wv_min=None, wv_max=None, n=15, plot=True):
        # do the cross-correlation between two epochs of spectra.

        # interpolate the epoch1 to epoch2(reference)
        f = interpolate.interp1d(w1, f1, bounds_error=False, fill_value=0.)
        fx = f(w_ref)
        f = interpolate.interp1d(w1, err1, bounds_error=False, fill_value=0.)
        err = f(w_ref)

        # wavelength range used to do the cross corelation
        ind = np.where((w_ref > wv_min) & (w_ref < wv_max), True, False)

        # do the chi2 minimization between two epochs
        chi2_all = []
        for i in range(int(-n), int(n) + 1):
            chi2 = np.array(np.sum((pd.DataFrame(fx[ind]).shift(i) - pd.DataFrame(f_ref[ind])) ** 2 /
                                   (pd.DataFrame(err[ind]).shift(i) ** 2 + pd.DataFrame(err_ref[ind]) ** 2))[0])
            chi2_all.append(chi2)

        x = np.array(range(int(-n), int(n) + 1))
        y = np.array(chi2_all)
        idx = np.argmin(y)
        xx = np.linspace(-n, n, 2 * n * 10)  # resolution = 0.2 pexel
        t, c, k = interpolate.splrep(x, y, s=0, k=5)
        spline = interpolate.BSpline(t, c, k, extrapolate=True)
        chi2_min, shift = spline(xx).min(), xx[np.argmin(spline(xx))]

        # get the 2.6 sigma error w.r.t. chi2_min+6.63 see Guo et al. 2019a
        sp = interpolate.UnivariateSpline(xx, spline(xx) - (chi2_min + 6.63), s=0)
        left = sp.roots()[np.where((sp.roots() < shift), True, False)].max()
        right = sp.roots()[np.where((sp.roots() > shift), True, False)].min()

        if plot == True:
            plt.figure(figsize=(15, 4))
            plt.subplots_adjust(wspace=0.3)
            plt.subplot(121)
            plt.plot(w_ref[ind], f_ref[ind])
            plt.plot(w_ref[ind], fx[ind])
            plt.xlabel(r'${\rm Wavelength}\ (\AA)$')
            plt.ylabel(r'${\rm Arbitrary}\ Flux $')

            plt.subplot(122)
            plt.plot(x * 69, y, 'bo', label='Original points')
            plt.plot(xx * 69, spline(xx), 'r', label='BSpline')
            plt.plot([left * 69, right * 69], [chi2_min + 6.63, chi2_min + 6.63], 'r-')
            plt.xlabel(r'${\rm Velocity}\ (km\ s^{-1} )$')
            plt.ylabel(r'$\chi^2$')
        return np.round(shift, 1), np.round(chi2_min, 1), np.round(left, 1), np.round(right, 1)

    def flux2L(self, flux, z):
        """Transfer flux to luminoity assuming a flat Universe"""
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        DL = cosmo.luminosity_distance(z).value * 10 ** 6 * 3.08 * 10 ** 18  # unit cm
        L = np.array(flux) * 1.e-17 * 4. * np.pi * DL ** 2  # erg/s/A
        return L

    def s2f(self, array):
        "string to float"
        return np.array([float(i) for i in array])

    def interp(self, wave, value, z=0., fill_value=0.):
        "interprate to standard wavelenth into rest or obs frame"
        wv = np.logspace(np.log10(3600.), np.log10(3600.) + 1.e-4 * 4649, 4650) / (1. + z)
        f = interpolate.interp1d(wave, value, bounds_error=False, fill_value=fill_value)
        return f(wv)

    def CalLineSN(self, w, f, line_min, line_max, conti):
        indline = np.where((w < line_max) & (w > line_min), True, False)
        indconti = np.where((w < conti + 25) & (w > conti - 25), True, False)
        return f[indline].mean() / f[indconti].std()

    def Calwidth(self, wave, flux, wv_min, wv_max):
        "Return line fwhm and sigma in unit of km/s"

        c = 299792.458  # km/s
        ind = np.where((wave < wv_max) & (wave > wv_min), True, False)

        # calculate the line FWHM (width)
        peak = wave[ind][np.argmax(flux[ind])]
        sp = interpolate.UnivariateSpline(wave[ind],
                                          flux[ind] - ((flux[ind].max() - flux[ind].min()) / 2. + flux[ind].min()), s=0)
        fwhm_left, fwhm_right = sp.roots().min(), sp.roots().max()
        fwhm = np.round(abs(fwhm_left - fwhm_right) / peak * c, 1)

        # calculate the line sigma (width)
        wave_cen = np.sum(wave[ind] * flux[ind]) / np.sum(flux[ind])
        sigma = np.round(np.sqrt(np.sum(wave[ind] ** 2 * flux[ind]) / np.sum(flux[ind]) - wave_cen ** 2) / peak * c, 1)
        return fwhm, sigma

    def Plotline(self, wave, ymin, ymax):
        line_cen = np.array([6564.60, 6549.85, 6585.27, 6718.29, 6732.66, 4862.68, 5008.24, 4687.02, \
                             4341.68, 3934.78, 3728.47, 3426.84, 2798.75, 1908.72, 1816.97, \
                             1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1215.67])
        line_name = np.array(['', '', 'Ha+[NII]', '', '[SII]', 'Hb', '[OIII]', \
                              'HeII', 'Hr', 'CaII', '[OII]', 'NeV', 'MgII', 'CIII]', \
                              'SiII', 'NIII]', 'NIV]', 'CIV', 'HeII', '', \
                              'SiIV+OIV]', 'Lya'])
        for ll in range(len(line_cen)):
            if wave.min() < line_cen[ll] < wave.max():
                plt.plot([line_cen[ll], line_cen[ll]], [ymin, ymax * 1], 'k:')
                plt.text(line_cen[ll] + 10, 0.9 * ymax, line_name[ll], rotation=90, fontsize=15)

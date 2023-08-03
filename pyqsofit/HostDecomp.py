# -*-coding:utf-8 -*-

"""
# Path       : 20230710PyQSOFit/pyqsofit
# File       : PriorDecomp.py
# Time       ：2023/7/27 15:47
# Author     ：Wenke Ren
# version    ：python 3.10
# Description：Use the PCA data to decompose the spectra with a priori
"""
import os, glob
from astropy.io import fits
import pandas as pd
import numpy as np
from lmfit import Minimizer, Parameters
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear
import astropy.units as u
import astropy.constants as const

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util


class QSO_PCA():
    def __init__(self, template_path, n_template=5, template_name=None):
        self.flux_qso = np.array([])
        self.wave_qso = np.array([])
        self.n_template = 0
        self.norm_factor = 1
        self.template_path = template_path
        self._read_PCA(template_path, n_template, template_name)

    def _read_PCA(self, template_path, n_template=5, template_name=None):
        if template_name is None:
            template_name = 'global'
        qso_fits = fits.open(os.path.join(template_path, f'qso_eigenspec_Yip2004_{template_name}.fits'))
        qso = qso_fits[1].data
        wave_qso = qso['wave'].flatten()
        flux_qso = (qso['pca'][0].T / np.abs(np.mean(qso['pca'][0], axis=1))).T
        flux_qso = flux_qso.astype('float64')
        self.n_template = n_template
        self.wave_qso = wave_qso.astype('float64')
        flux_qso = flux_qso[0:n_template, :]
        # self.norm_factor = 1 / np.median(np.abs(flux_qso))
        self.flux_qso = flux_qso * self.norm_factor
        self.vel_disp = 69.  # in km/s
        return wave_qso, flux_qso

    def interp_data(self, wave_exp):

        wave_in = self.wave_qso
        flux_in = self.flux_qso
        wave_exp = np.array(wave_exp, dtype='float64')
        # c0 = const.c.to(u.km / u.s).value

        if np.max(wave_in) <= np.max(wave_exp) or np.min(wave_in) >= np.min(wave_exp):
            raise ValueError(f'The length of template is not long enough to cover the wave range you give. '
                             f'The range of template is [{np.min(wave_in)}, {np.max(wave_in)}], while the '
                             f'range of wave input is [{np.min(wave_exp)}, {np.max(wave_exp)}]')

        if np.ndim(flux_in) == 1:
            f_flux = interp1d(wave_in, flux_in, bounds_error=False, fill_value=0)
            flux_out = f_flux(wave_exp)

        elif np.ndim(flux_in) == 2:
            n_spec = np.shape(flux_in)[0]
            flux_out = np.zeros((n_spec, len(wave_exp)))
            for i in range(n_spec):
                f_flux = interp1d(wave_in, flux_in[i], bounds_error=False, fill_value=0)
                flux_out[i] = f_flux(wave_exp)
        else:
            raise IndexError('The dimension of the flux_in is not correct.')
        self.wave_qso_model = wave_exp
        self.flux_qso_model = flux_out
        return flux_out


class host_template():
    def __init__(self, template_path, n_template=5, template_type='PCA'):
        self.flux_gal = np.array([])
        self.wave_gal = np.array([])
        self.vel_disp = 0
        self.n_template = n_template
        self.norm_factor = 1
        self.template_path = template_path
        self.template_type = template_type
        if template_type == 'PCA':
            self._read_PCA(template_path)
        elif template_type == 'BC03':
            self._read_BC03(template_path)
        elif template_type == 'indo19':
            self._read_INDO(template_path)
        elif template_type == 'indo50':
            self._read_INDO(template_path)
        elif template_type == 'M09_17':
            self._read_M09(template_path)
        elif template_type == 'MILES17':
            self._read_MILES(template_path)
        else:
            raise FileNotFoundError('Such template is not prepared!')

    def _read_PCA(self, template_path):
        galaxy = fits.open(os.path.join(template_path, 'gal_eigenspec_Yip2004.fits'))
        gal = galaxy[1].data
        wave_gal = gal['wave'].flatten()
        flux_gal = (gal['pca'][0].T / np.abs(np.mean(gal['pca'][0], axis=1))).T
        flux_gal = flux_gal.astype('float64')
        self.wave_gal = wave_gal.astype('float64')
        if self.n_template < len(flux_gal):
            flux_gal = flux_gal[0:self.n_template, :]
        else:
            self.n_template = len(flux_gal)
        # self.norm_factor = 1 / np.median(np.abs(flux_gal))
        self.flux_gal = flux_gal * self.norm_factor
        self.vel_disp = 69.  # in km/s
        return wave_gal, flux_gal

    def _read_BC03(self, template_path):
        wave_gal = np.array([])
        flux03 = []
        bc03_file_names = glob.glob(os.path.join(template_path, '*.gz'))
        bc03_idx = np.array([os.path.split(nm)[1].split('_')[0] for nm in bc03_file_names], dtype='int')
        bc03_file_names = np.array(bc03_file_names)[np.argsort(bc03_idx)]
        for i, f in enumerate(bc03_file_names):
            gal_temp = np.genfromtxt(f)
            wave_gal = gal_temp[:, 0]
            flux03.append(gal_temp[:, 1])
        flux_gal = np.array(flux03, dtype='float64').reshape(len(bc03_file_names), -1)
        if self.n_template < len(flux_gal):
            flux_gal = flux_gal[0:self.n_template, :]
        else:
            self.n_template = len(flux_gal)
        self.wave_gal = wave_gal.astype('float64')
        self.norm_factor = 1 / np.median(np.abs(flux_gal))
        self.flux_gal = flux_gal * self.norm_factor
        self.vel_disp = 75.  # in km/s
        return wave_gal, flux_gal

    def _read_INDO(self, template_path):
        flux_temp = np.array([])
        wave_gal = np.array([])
        cc = 0
        for i in glob.glob(template_path + '/*.fits'):
            cc = cc + 1
            gal_temp = fits.open(i)
            h2 = gal_temp[0].header
            wave_gal = np.array(h2['CRVAL1'] + h2['CDELT1'] * np.arange(h2['NAXIS1']))
            flux_temp = np.concatenate(
                (flux_temp, gal_temp[0].data / np.abs(np.mean(gal_temp[0].data))))
        flux_gal = np.array(flux_temp, dtype='float64').reshape(cc, -1)
        self.n_template = cc
        self.wave_gal = wave_gal.astype('float64')
        self.norm_factor = 1 / np.median(np.abs(flux_gal))
        self.flux_gal = flux_gal * self.norm_factor
        self.vel_disp = 33.7  # in km/s
        return wave_gal, flux_gal

    def _read_M09(self, template_path):
        flux_temp = np.array([])
        wave_gal = np.array([])
        cc = 0
        for i in glob.glob(template_path + '/*.csv'):
            cc = cc + 1
            gal_temp = pd.read_csv(i, comment='#', names=['T', 'Z', 'lambda', 'flux'])
            wave_gal = np.array(gal_temp.loc[:, 'lambda'])
            flux_temp = np.concatenate(
                (flux_temp, gal_temp.loc[:, 'flux'] / np.abs(np.mean(gal_temp.loc[:, 'flux']))))
        flux_gal = np.array(flux_temp, dtype='float64').reshape(cc, -1)
        self.n_template = cc
        self.wave_gal = wave_gal.astype('float64')
        self.norm_factor = 1 / np.median(np.abs(flux_gal))
        self.flux_gal = flux_gal * self.norm_factor
        self.vel_disp = 60  # in km/s TO BE DECIDED!!!!!
        return wave_gal, flux_gal

    def _read_MILES(self, template_path):
        flux_temp = np.array([])
        wave_gal = np.array([])
        cc = 0
        for i in glob.glob(template_path + '/*.csv'):
            cc = cc + 1
            gal_temp = pd.read_csv(i, comment='#', names=['lambda', 'flux'], delim_whitespace=True)
            wave_gal = np.array(gal_temp.loc[:, 'lambda'])
            flux_temp = np.concatenate(
                (flux_temp, gal_temp.loc[:, 'flux'] / np.abs(np.mean(gal_temp.loc[:, 'flux']))))
        flux_gal = np.array(flux_temp, dtype='float64').reshape(cc, -1)
        self.n_template = cc
        self.wave_gal = wave_gal.astype('float64')
        self.norm_factor = 1 / np.median(np.abs(flux_gal))
        self.flux_gal = flux_gal * self.norm_factor
        self.vel_disp = 62.6  # in km/s
        return wave_gal, flux_gal

    def interp_data(self, wave_exp, broaden: bool = False, shift: float = 0, sigma: float = 0):
        '''
        Interpolate the galaxy template and find the interpolation flux at give wavelength. We also provide
        broaden methods which allow user to convolve a gaussian core before the interpolation. However,
        considering most template are not evenly sampled, we have to interpolate twice if the broaden method
        is used, which might introduce more biases.
        :param wave_exp:
        :param broaden:
        :param shift: float, in km/s
        :param sigma: float, in km/s
        :return:
        '''
        wave_in = self.wave_gal
        flux_in = self.flux_gal
        wave_exp = np.array(wave_exp, dtype='float64')
        c0 = const.c.to(u.km / u.s).value

        if broaden is True:
            loglam = np.linspace(np.log10(np.min(wave_in)), np.log10(np.max(wave_in)), len(wave_in))
            delta_ll = np.median(loglam[1:] - loglam[:-1])

            spec_R = c0 * (10 ** delta_ll - 1)
            if spec_R < sigma:
                xx_lim = np.abs(shift) + sigma * 3
                loglam_shift = np.log10(shift / c0 + 1)
                loglam_sigma = np.log10(sigma / c0 + 1)
                conv_xx = np.arange(-xx_lim, xx_lim, delta_ll)
                gaussian_func = np.exp(-(conv_xx - loglam_shift) ** 2 / (2. * loglam_sigma ** 2)) / (
                        np.sqrt(2. * np.pi) * loglam_sigma)
                conv_core = gaussian_func / np.sum(gaussian_func)

                flux_interp = interp1d(wave_in, flux_in, bounds_error=False, fill_value=0)(
                    10 ** loglam)
                flux_broaden = np.convolve(flux_interp, conv_core, mode='same')
                wave_in = 10 ** loglam
                flux_in = flux_broaden
            else:
                wave_in = wave_in / (1 + sigma / c0)

        if np.max(wave_in) <= np.max(wave_exp) or np.min(wave_in) >= np.min(wave_exp):
            raise ValueError(f'The length of template is not long enough to cover the wave range you give. '
                             f'The range of template is [{np.min(wave_in)}, {np.max(wave_in)}], while the '
                             f'range of wave input is [{np.min(wave_exp)}, {np.max(wave_exp)}]')

        if np.ndim(flux_in) == 1:
            f_flux = interp1d(wave_in, flux_in, bounds_error=False, fill_value=0)
            flux_out = f_flux(wave_exp)

        elif np.ndim(flux_in) == 2:
            n_spec = np.shape(flux_in)[0]
            flux_out = np.zeros((n_spec, len(wave_exp)))
            for i in range(n_spec):
                f_flux = interp1d(wave_in, flux_in[i], bounds_error=False, fill_value=0)
                flux_out[i] = f_flux(wave_exp)
        else:
            raise IndexError('The dimension of the flux_in is not correct.')
        self.wave_gal_model = wave_exp
        self.flux_gal_model = flux_out
        return flux_out


def _na_mask(wave, flux, err):
    """
    Mask the narrow line region
    :param wave:
    :param flux:
    :param err:
    :return:
    """
    window_idx = np.ones_like(wave, dtype='bool')
    mask_window = np.array([[3717., 3737.], [4091., 4111.], [4330., 4350.], [4852., 4872.],
                            [4950., 4970.], [5000., 5020.], [6540., 6590.], [6710., 6740.]])
    for sub_win in mask_window:
        window_idx = window_idx & np.where((wave > sub_win[0]) & (wave < sub_win[1]), False, True)
    wave_fit, flux_fit, err_fit = wave[window_idx], flux[window_idx], err[window_idx]

    return wave_fit, flux_fit, err_fit


class Linear_decomp():
    def __init__(self, wave, flux, err, n_gal, n_qso, path, host_type='PCA', qso_type='global', na_mask=False):
        self.wave = wave
        self.flux = flux
        self.err = err
        self.n_gal = n_gal
        self.n_qso = n_qso

        path2prior = os.path.join(path, 'pca/prior')
        path2qso = os.path.join(path, 'pca/Yip_pca_templates')
        if host_type == 'PCA':
            path2host = os.path.join(path, 'pca/Yip_pca_templates')
        elif host_type == 'BC03':
            path2host = os.path.join(path, 'bc03')
        else:
            path2host = path

        self.qso_tmp = QSO_PCA(path2qso, n_qso, template_name=qso_type)
        self.gal_tmp = host_template(path2host, n_gal, template_type=host_type)

        # Get the shortest wavelength range
        wave_min = np.max([np.min(wave), np.min(self.qso_tmp.wave_qso), np.min(self.gal_tmp.wave_gal)])
        wave_max = np.min([np.max(wave), np.max(self.qso_tmp.wave_qso), np.max(self.gal_tmp.wave_gal)])
        ind_data = np.where((wave > wave_min) & (wave < wave_max), True, False)
        self.wave, self.flux, self.err = wave[ind_data], flux[ind_data], err[ind_data]

        if na_mask == True:
            self.wave_fit, self.flux_fit, self.err_fit = _na_mask(self.wave, self.flux, self.err)
        else:
            self.wave_fit, self.flux_fit, self.err_fit = self.wave, self.flux, self.err

        self.qso_datacube = self.qso_tmp.interp_data(self.wave_fit)
        self.gal_datacube = self.gal_tmp.interp_data(self.wave_fit)

        self.n_qso = self.qso_tmp.n_template
        self.n_gal = self.gal_tmp.n_template

        if self.n_qso != n_qso:
            raise ValueError(f'The number of qso template is not correct. '
                             f'You ask for {n_qso} template, while the maximum number of qso template is {self.n_qso}')
        if self.n_gal != n_gal:
            raise ValueError(f'The number of gal template is not correct. '
                             f'You ask for {n_gal} template, while the maximum number of gal template is {self.n_gal}')

    def auto_decomp(self):
        flux_temp = np.vstack((self.qso_datacube, self.gal_datacube)).T

        # set the bondage of fitting
        if self.gal_tmp.template_type.lower() == 'pca':
            bounds = (-np.inf, np.inf)
        else:
            bounds = (
                np.concatenate([-np.ones(self.qso_tmp.n_template) * np.inf, np.zeros(self.gal_tmp.n_template)]),
                np.concatenate([np.ones(self.qso_tmp.n_template) * np.inf, np.ones(self.gal_tmp.n_template) * np.inf])
            )

        self.result_params = lsq_linear(flux_temp, self.flux_fit, bounds=bounds)['x']
        qso_par = self.result_params[:self.n_qso]
        gal_par = self.result_params[self.n_qso:]

        qso_flux = self.qso_model(qso_par, self.wave)
        gal_flux = self.gal_model(gal_par, self.wave)

        # Calculate the host galaxy fraction at 4200 and 5100
        frac_host_4200 = -1.
        frac_host_5100 = -1.

        ind_f4200 = np.where((self.wave > 4160.) & (self.wave < 4210.), True, False)
        if np.sum(ind_f4200) > 10:
            frac_host_4200 = np.sum(gal_flux[ind_f4200]) / np.sum(self.flux[ind_f4200])

        ind_f5100 = np.where((self.wave > 5080.) & (self.wave < 5130.), True, False)
        if np.sum(ind_f5100) > 10:
            frac_host_5100 = np.sum(gal_flux[ind_f5100]) / np.sum(self.flux[ind_f5100])

        data_cube = np.vstack((self.wave, self.flux, self.err, gal_flux, qso_flux))

        return data_cube, frac_host_4200, frac_host_5100

    def qso_model(self, param: list = None, wave=None):
        if param is None:
            param = np.array(self.result_params)[:self.n_qso]
        if wave is None:
            return np.dot(np.array(param), self.qso_datacube)
        else:
            return np.dot(np.array(param), self.qso_tmp.interp_data(wave))

    def gal_model(self, param: list = None, wave=None):
        if param is None:
            param = np.array(self.result_params)[self.n_qso:]
        if wave is None:
            return np.dot(np.array(param), self.gal_datacube)
        else:
            return np.dot(np.array(param), self.gal_tmp.interp_data(wave))


class Prior_decomp():
    def __init__(self, wave, flux, err, n_gal, n_qso, path, host_type='PCA', qso_type='CZBIN1', na_mask=True,
                 fh_ini_list=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        # (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        # (0.2, 0.4, 0.6, 0.8)
        path2prior = os.path.join(path, 'pca/prior')
        path2qso = os.path.join(path, 'pca/Yip_pca_templates')
        path2host = os.path.join(path, 'pca/Yip_pca_templates')

        GAL_prior_name = 'GAL_pp_prior.csv'
        QSO_prior_name = f'QSO_pp_prior_{qso_type}.csv'
        self.fh_ini_list = fh_ini_list

        self.qso_tmp = QSO_PCA(path2qso, n_qso, template_name=qso_type)
        self.gal_tmp = host_template(n_template=n_gal, template_path=path2host, template_type=host_type)

        # Get the shortest wavelength range
        wave_min = np.max([np.min(wave), np.min(self.qso_tmp.wave_qso), np.min(self.gal_tmp.wave_gal)])
        wave_max = np.min([np.max(wave), np.max(self.qso_tmp.wave_qso), np.max(self.gal_tmp.wave_gal)])
        ind_data = np.where((wave > wave_min) & (wave < wave_max), True, False)
        self.wave, self.flux, self.err = wave[ind_data], flux[ind_data], err[ind_data]

        if na_mask == True:
            self.wave_fit, self.flux_fit, self.err_fit = _na_mask(self.wave, self.flux, self.err)
        else:
            self.wave_fit, self.flux_fit, self.err_fit = self.wave, self.flux, self.err

        self.qso_datacube = self.qso_tmp.interp_data(self.wave_fit)
        self.gal_datacube = self.gal_tmp.interp_data(self.wave_fit)
        self.n_qso = self.qso_tmp.n_template
        self.n_gal = self.gal_tmp.n_template

        self.qso_prior = np.array([[0, 1]] * self.n_qso)
        self.gal_prior = np.array([[0, 1]] * self.n_gal)

        self.read_prior(path2prior, QSO_prior_name, GAL_prior_name)

    def read_prior(self, prior_loc, QSO_prior_name='QSO_pp_prior.csv', GAL_prior_name='GAL_pp_prior.csv'):
        self.qso_prior = self._read_prior(os.path.join(prior_loc, QSO_prior_name), self.n_qso)
        self.gal_prior = self._read_prior(os.path.join(prior_loc, GAL_prior_name), self.n_gal)

    def auto_decomp(self, reg_factor=0.2):
        rchi2_list = []
        result_list = []
        for fh_ini in self.fh_ini_list:
            init_params = self.initial_params(fh_ini)
            fit_result, rchi2 = self.decompose(reg_factor, init_params)
            rchi2_list.append(rchi2)
            result_list.append(fit_result)

        best_fit = result_list[np.argmin(rchi2_list)]
        qso_par = best_fit[:self.n_qso]
        gal_par = best_fit[self.n_qso:]

        qso_flux = self.qso_model(qso_par, self.wave)
        gal_flux = self.gal_model(gal_par, self.wave)

        # Calculate the host galaxy fraction at 4200 and 5100
        frac_host_4200 = -1.
        frac_host_5100 = -1.

        ind_f4200 = np.where((self.wave > 4160.) & (self.wave < 4210.), True, False)
        if np.sum(ind_f4200) > 10:
            frac_host_4200 = np.sum(gal_flux[ind_f4200]) / np.sum(self.flux[ind_f4200])

        ind_f5100 = np.where((self.wave > 5080.) & (self.wave < 5130.), True, False)
        if np.sum(ind_f5100) > 10:
            frac_host_5100 = np.sum(gal_flux[ind_f5100]) / np.sum(self.flux[ind_f5100])

        data_cube = np.vstack((self.wave, self.flux, self.err, gal_flux, qso_flux))

        return data_cube, frac_host_4200, frac_host_5100

    def initial_params(self, fh_ini=None):
        if fh_ini is None: fh_ini = self.fh_ini_list[0]
        spec_level = np.median(self.flux_fit)
        qso_norm_factor = (1 - fh_ini) * spec_level / np.median(self.qso_datacube[0])
        gal_norm_factor = fh_ini * spec_level / np.median(self.gal_datacube[0])
        init_params = Parameters()
        for i in range(self.n_qso):
            init_params.add(f'QSO_{i}', value=self.qso_prior[i, 0] * qso_norm_factor)
        for i in range(self.n_gal):
            init_params.add(f'GAL_{i}', value=self.gal_prior[i, 0] * gal_norm_factor)

        return init_params

    def decompose(self, reg_factor, init_params, non_negative=False):
        if non_negative is True:
            init_params['QSO_0'].min = 0
            init_params['GAL_0'].min = 0
        spec_fitter = Minimizer(self._residuals, init_params,
                                fcn_args=(self.flux_fit, self.err_fit, reg_factor))
        fit_result = spec_fitter.leastsq(ftol=1.e-8, xtol=1.e-8)
        result_params = fit_result.params

        par_list = np.array(list(result_params.valuesdict().values()))
        qso_par = par_list[:self.n_qso]
        gal_par = par_list[self.n_qso:]

        qso_flux = self.qso_model(qso_par)
        gal_flux = self.gal_model(gal_par)
        model_flux = qso_flux + gal_flux

        rchi2 = np.sum(((self.flux_fit - model_flux) / self.err_fit) ** 2) / (
                len(self.flux_fit) - self.n_qso - self.n_gal)

        return par_list, rchi2

    def _residuals(self, param: Parameters, yval, err, reg_factor):
        chi_array = self._get_diff(param, yval, err)
        penalty = reg_factor * np.sqrt(self._get_prior_diff(param)) * np.sum(chi_array ** 2) / len(yval)
        # print(np.mean(chi_array + penalty))
        return chi_array + penalty

    def _read_prior(self, path2prior, n_pp):
        prior = np.array(pd.read_csv(path2prior))
        return prior[:n_pp]

    def qso_model(self, param: list = None, wave=None):
        if param is None:
            param = np.array(list(self.result_params.valuesdict().values()))[:self.n_qso]
        if wave is None:
            return np.dot(np.array(param), self.qso_datacube)
        else:
            return np.dot(np.array(param), self.qso_tmp.interp_data(wave))

    def gal_model(self, param: list = None, wave=None):
        if param is None:
            param = np.array(list(self.result_params.valuesdict().values()))[self.n_qso:]
        if wave is None:
            return np.dot(np.array(param), self.gal_datacube)
        else:
            return np.dot(np.array(param), self.gal_tmp.interp_data(wave))

    def cal_model(self, param: Parameters, wave=None):
        par_list = np.array(list(param.valuesdict().values()))
        qso_par = par_list[:self.n_qso]
        gal_par = par_list[self.n_qso:]
        return self.qso_model(qso_par, wave) + self.gal_model(gal_par, wave)

    def _get_diff(self, param: Parameters, yval, err):
        return (yval - self.cal_model(param)) / err

    def _get_prior_diff(self, param: Parameters):
        par_list = np.array(list(param.valuesdict().values()))
        qso_par = par_list[:self.n_qso]
        gal_par = par_list[self.n_qso:]

        w_qso = (qso_par / qso_par[0] - self.qso_prior[:, 0]) / self.qso_prior[:, 1]
        w_gal = (gal_par / gal_par[0] - self.gal_prior[:, 0]) / self.gal_prior[:, 1]

        return (np.sum(w_qso ** 2) + np.sum(w_gal ** 2)) / (self.n_qso + self.n_gal - 2)


def ppxf_kinematics(wave, flux, err, path, fit_range=(3900, 5350)):
    ppxf_dir = os.path.dirname(os.path.realpath(util.__file__))

    redshift = 0
    lam_gal = wave

    fit_cut = np.where((lam_gal >= fit_range[0]) & (lam_gal <= fit_range[1]), True, False)
    lam_gal = lam_gal[fit_cut]

    galaxy = flux[fit_cut]
    norm_factor = np.median(galaxy)
    galaxy = galaxy / norm_factor
    ln_lam_gal = np.log(lam_gal)
    d_ln_lam_gal = np.diff(ln_lam_gal[[0, -1]]) / (ln_lam_gal.size - 1)
    c = 299792.458
    velscale = c * d_ln_lam_gal
    velscale = velscale.item()
    noise = err[fit_cut] / np.median(galaxy)
    noise[noise == 0] = np.median(noise)

    dlam_gal = np.diff(lam_gal)
    dlam_gal = np.append(dlam_gal, dlam_gal[-1])
    fwhm_gal = 2.76

    valdes = glob.glob(os.path.join(path, 'indo/*.fits'))
    fwhm_tem = 1.35

    hdu = fits.open(valdes[0])
    ssp = hdu[0].data
    h2 = hdu[0].header

    lam_temp = h2['CRVAL1'] + h2['CDELT1'] * np.arange(h2['NAXIS1'])
    good_lam = (lam_temp > 3800.) & (lam_temp < 5500.)
    lam_temp = lam_temp[good_lam]
    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]

    sspNew, ln_lam_temp = util.log_rebin(lamRange_temp, ssp[good_lam], velscale=velscale)[:2]
    templates = np.empty((sspNew.size, len(valdes)))

    fwhm_dif = np.sqrt((fwhm_gal ** 2 - fwhm_tem ** 2))
    sigma = fwhm_dif / 2.355 / h2['CDELT1']  # Sigma difference in pixels

    for j, fname in enumerate(valdes):
        hdu = fits.open(fname)
        ssp = hdu[0].data
        ssp = util.gaussian_filter1d(ssp[good_lam], sigma)  # perform convolution with variable sigma
        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates[:, j] = sspNew / np.median(sspNew[sspNew > 0])  # Normalizes templates

    goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, redshift)
    line_mask = np.argwhere(
        ((lam_gal > 4760) & (lam_gal < 5020))).flatten()
    # | ((lam_gal > 3955) & (lam_gal < 3985))
    goodpixels = goodpixels[~np.isin(goodpixels, line_mask)]

    c = 299792.458  # km/s
    vel = c * np.log(1 + redshift)  # eq.(8) of Cappellari (2017)
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]

    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodpixels, plot=False, moments=2, trig=1,
              degree=20, lam=lam_gal, lam_temp=np.exp(ln_lam_temp), quiet=True)

    return np.array([pp.sol[1], pp.error[1] * np.sqrt(pp.chi2), pp.sol[0], pp.chi2])

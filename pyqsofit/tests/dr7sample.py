import glob, os,sys,timeit
import matplotlib
import numpy as np
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
import matplotlib.backends.backend_pdf as bpdf

from astroquery.sdss import SDSS
from astropy import coordinates as coords

warnings.filterwarnings("ignore")

# Use custom matplotlib style to make Yue happy
QSOFit.set_mpl_style()

# Show the versions so we know what works
import astropy
import lmfit
import pyqsofit
print(astropy.__version__)
print(lmfit.__version__)
print(pyqsofit.__version__)

import emcee # optional, for MCMC
print(emcee.__version__)

path_ex = os.path.abspath('pyqsofit/tests') # The absolute path to the example directory 

print(path_ex)
# Setup the parameter file

"""
Create parameter file
lambda    complexname  minwav maxwav linename ngauss inisca minsca maxsca inisig minsig maxsig voff vindex windex findex fvalue vary
"""

newdata = np.rec.array([
(6564.61, r'H$\alpha$', 6400, 6800, 'Ha_br',   2, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.015, 0, 0, 0, 0.05 , 1),
(6564.61, r'H$\alpha$', 6400, 6800, 'Ha_na',   1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.01,  1, 1, 0, 0.002, 1),
(6549.85, r'H$\alpha$', 6400, 6800, 'NII6549', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 1, 0.001, 1),
(6585.28, r'H$\alpha$', 6400, 6800, 'NII6585', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 1, 0.003, 1),
(6718.29, r'H$\alpha$', 6400, 6800, 'SII6718', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 2, 0.001, 1),
(6732.67, r'H$\alpha$', 6400, 6800, 'SII6732', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 2, 0.001, 1),

(4862.68, r'H$\beta$', 4640, 5100, 'Hb_br',     2, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.01, 0, 0, 0, 0.01 , 1),
(4862.68, r'H$\beta$', 4640, 5100, 'Hb_na',     1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1),
(4960.30, r'H$\beta$', 4640, 5100, 'OIII4959c', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1),
(5008.24, r'H$\beta$', 4640, 5100, 'OIII5007c', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.004, 1),
#(4960.30, r'H$\beta$', 4640, 5100, 'OIII4959w',   1, 0.1, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.001, 1),
#(5008.24, r'H$\beta$', 4640, 5100, 'OIII5007w',   1, 0.1, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.002, 1),
#(4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.005, 0, 0, 0, 0.001, 1),
#(4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_na', 1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 0.005, 1, 1, 0, 0.001, 1),

#(3934.78, 'CaII', 3900, 3960, 'CaII3934' , 2, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 99, 0, 0, -0.001, 1),

#(3728.48, 'OII', 3650, 3800, 'OII3728', 1, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 1, 1, 0, 0.001, 1),
    
#(3426.84, 'NeV', 3380, 3480, 'NeV3426',    1, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 0, 0, 0, 0.001, 1),
#(3426.84, 'NeV', 3380, 3480, 'NeV3426_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025,   0.02,   0.01, 0, 0, 0, 0.001, 1),

(2798.75, 'MgII', 2700, 2900, 'MgII_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.0017, 0, 0, 0, 0.05, 1),
(2798.75, 'MgII', 2700, 2900, 'MgII_na', 2, 0.1, 0.0, 1e10, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002, 1),

(1908.73, 'CIII', 1700, 1970, 'CIII_br',   2, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01, 1),
#(1908.73, 'CIII', 1700, 1970, 'CIII_na',   1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
#(1892.03, 'CIII', 1700, 1970, 'SiIII1892', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
#(1857.40, 'CIII', 1700, 1970, 'AlIII1857', 1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
#(1816.98, 'CIII', 1700, 1970, 'SiII1816',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
#(1786.7,  'CIII', 1700, 1970, 'FeII1787',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
#(1750.26, 'CIII', 1700, 1970, 'NIII1750',  1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),
#(1718.55, 'CIII', 1700, 1900, 'NIV1718',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),

(1549.06, 'CIV', 1500, 1700, 'CIV_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.015, 0, 0, 0, 0.05 , 1),
(1549.06, 'CIV', 1500, 1700, 'CIV_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01,  1, 1, 0, 0.002, 1),
#(1640.42, 'CIV', 1500, 1700, 'HeII1640',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
#(1663.48, 'CIV', 1500, 1700, 'OIII1663',    1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.008, 1, 1, 0, 0.002, 1),
#(1640.42, 'CIV', 1500, 1700, 'HeII1640_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),
#(1663.48, 'CIV', 1500, 1700, 'OIII1663_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),

#(1402.06, 'SiIV', 1290, 1450, 'SiIV_OIV1', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
#(1396.76, 'SiIV', 1290, 1450, 'SiIV_OIV2', 1, 0.1, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
#(1335.30, 'SiIV', 1290, 1450, 'CII1335',   1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),
#(1304.35, 'SiIV', 1290, 1450, 'OI1304',    1, 0.1, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),

(1215.67, 'Lya', 1150, 1290, 'Lya_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.02, 0, 0, 0, 0.05 , 1),
(1215.67, 'Lya', 1150, 1290, 'Lya_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01, 0, 0, 0, 0.002, 1)],

formats = 'float32,      a20,  float32, float32,      a20,  int32, float32, float32, float32, float32, float32, float32, float32,   int32,  int32,  int32,   float32, int32',
names  =  ' lambda, compname,   minwav,  maxwav, linename, ngauss,  inisca,  minsca,  maxsca,  inisig,  minsig,  maxsig,  voff,     vindex, windex,  findex,  fvalue,  vary')

# Header
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'

# Can be set to negative for absorption lines if you want
hdr['inisca'] = 'Initial guess of line scale [in ??]'
hdr['minsca'] = 'Lower range of line scale [??]'
hdr['maxsca'] = 'Upper range of line scale [??]'

hdr['inisig'] = 'Initial guess of linesigma [in lnlambda]'
hdr['minsig'] = 'Lower range of line sigma [lnlambda]'  
hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'

hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'

hdr['vary'] = 'Whether or not to vary the line parameters (set to 0 to fix the line parameters to initial values)'

# Save line info
hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')
hdu.writeto(os.path.join(path_ex, 'qsopar.fits'), overwrite=True)

from astropy.table import Table
print(Table(newdata))

# Download the SDSS DR7 sample
import urllib.request
import gzip

cat_file_name = 'dr7_bh_Nov19_2013.fits.gz'

if not os.path.exists(cat_file_name):
    urllib.request.urlretrieve(f'http://quasar.astro.illinois.edu/BH_mass/data/catalogs/{cat_file_name}', cat_file_name)

# Fit a sample of nqsofit spectra
nqsofit = 200

# Line properties
line_calc_names = ['CIV_br', 'MgII_br', 'Hb_br']
line_shen_names = ['CIV', 'BROAD_MGII', 'BROAD_HB']
line_fwhms = np.zeros((nqsofit, len(line_calc_names)))
line_Ls = np.zeros((nqsofit, len(line_calc_names)))
Mis = np.zeros(nqsofit)
zs = np.zeros(nqsofit)

start = timeit.default_timer()

with gzip.open(cat_file_name, 'rb') as f:
    
    spec_all = fits.open(f)
    
    len_specs = len(spec_all[1].data)
    range_rand = np.random.choice(range(len_specs), nqsofit, replace=False)

    print(spec_all.info())
    print(repr(spec_all[1].header))
    
    # Catalog properties
    Mi_all = spec_all[1].data['MI_Z2']
    z_all = spec_all[1].data['REDSHIFT']
    line_fwhm_all = np.zeros_like(line_fwhms)
    line_L_all = np.zeros_like(line_Ls)
    
    # Emission line loop
    for j, line in enumerate(line_shen_names):
        line_fwhm_all[:,j] = spec_all[1].data[f'FWHM_{line}'][range_rand]
        line_L_all[:,j] = spec_all[1].data[f'LOGL_{line}'][range_rand]
    
    # Save as single PDF
    with bpdf.PdfPages(os.path.join(path_ex, 'fit_results.pdf')) as pdf:

        # For each spectrum
        for i, ind in enumerate(range_rand):

            plate = spec_all[1].data['PLATE'][ind]
            mjd = spec_all[1].data['MJD'][ind]
            fiber = spec_all[1].data['FIBER'][ind]
            ra = spec_all[1].data['RA'][ind]
            dec = spec_all[1].data['DEC'][ind]
            Mi = spec_all[1].data['MI_Z2'][ind]
            
            # Query the spectrum
            pos = coords.SkyCoord(ra, dec, unit='deg')
            xid = SDSS.query_region(pos, spectro=True)
            
            if xid is None:
                continue
            
            mask = (xid['plate'] == plate) & (xid['mjd'] == mjd) & (xid['fiberID'] == fiber)
            print(xid[mask])
            sp = SDSS.get_spectra(matches=xid[mask])
            print(sp)
            
            data = sp[0]

            # Requried
            lam = 10**data[1].data['loglam']        # OBS wavelength [A]
            flux = data[1].data['flux']             # OBS flux [erg/s/cm^2/A]
            err = 1/np.sqrt(data[1].data['ivar'])   # 1 sigma error
            z = data[2].data['z'][0]                # Redshift

            q = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plate, mjd=mjd, fiberid=fiber,  path=path_ex)

            # Do the fitting
            q.Fit(param_file_name='qsopar.fits', name=None, nsmooth=1, and_or_mask=False, deredden=True, reject_badpix=True,
                  wave_mask=None, decompose_host=True, use_ppxf=False, BC03=True, Mi=None, host_line_mask=True, npca_gal=10, npca_qso=5,
                  Fe_uv_op=True, poly=True, BC=False, rej_abs=False, MC=False, nsamp=200, initial_guess=None, n_pix_min_host=np.inf,
                  linefit=True, save_result=False, plot_fig=True, save_fig=False, plot_line_name=True, plot_legend=True)
            
            # Emission line loop
            for j, line in enumerate(line_calc_names):
                
                print(line)
                
                if line.endswith('_br'):
                    line_type = 'broad'
                else:
                    line_type = 'narrow'
            
                # Get the line properties
                fwhm, sigma, ew, peak, area = q.line_prop_from_name(line, line_type)

                print(f"Broad {line}:")
                print("FWHM (km/s)", np.round(fwhm, 1))
                print("Sigma (km/s)", np.round(sigma, 1))
                print("EW (A)", np.round(ew, 1))
                print("Peak (A)", np.round(peak, 1))
                print("area (10^(-17) erg/s/cm^2)", np.round(area, 1))
                print("")
                
                line_fwhms[i, j] = fwhm
                line_Ls[i, j] = np.log10(q.flux2L(area))
                Mis[i] = Mi
                zs[i] = z
                

            # Stack the results into 1 file
            pdf.savefig(q.fig)
            
    # End open file
            
# Summary figures

def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets)**2).mean())

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.scatter(z_all, Mi_all, color='k', marker='.', alpha=0.01)
ax.scatter(zs, Mis, color='r', marker='o', alpha=0.5)
ax.set_title(pyqsofit.__version__)

ax.set_xlabel('Redshift', fontsize=20)
ax.set_ylabel(r'$M_i(z=2)$', fontsize=20)

ax.set_xlim([0, 5.5])
ax.set_ylim([-22, -30])

fig.tight_layout()
fig.savefig(os.path.join(path_ex, 'Mi_z.pdf'), dpi=300)

# Emission line loop
for j, line in enumerate(line_calc_names):
        
    # Summary figures
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    
    x = np.linspace(1000, 12000)
    axs[0].plot(x, x, color='gray')
    
    x = np.linspace(41, 48)
    axs[1].plot(x, x, color='gray', zorder=-1)
    
    axs[0].scatter(line_fwhm_all[:, j], line_fwhms[:, j], color='r', marker='o', alpha=0.5)
    rmse_fwhm = rmse(line_fwhm_all[:, j], line_fwhms[:, j])
    print(f'RMSE FWHM = {rmse_fwhm}')
    axs[0].set_xlim([1000, 12000])
    axs[0].set_ylim([1000, 12000])
    
    axs[1].scatter(line_L_all[:, j], line_Ls[:, j], color='r', marker='o', alpha=0.5)
    rmse_L = rmse(line_fwhm_all[:, j], line_fwhms[:, j])
    print(f'RMSE L = {rmse_L}')
    axs[1].set_xlim([41, 46])
    axs[1].set_ylim([41, 46])
    
    axs[0].set_title(line)
    axs[0].set_xlabel(f'FWHM v{pyqsofit.__version__}', fontsize=20)
    axs[0].set_ylabel(f'FWHM Shen 2011', fontsize=20)
    
    axs[1].set_title(line)
    axs[1].set_xlabel(f'log L v{pyqsofit.__version__}', fontsize=20)
    axs[1].set_ylabel(f'log L Shen 2011', fontsize=20)
    
    fig.tight_layout()
    fig.savefig(os.path.join(path_ex, f'{line}.pdf'), dpi=300)

end = timeit.default_timer()
print('finshed in '+str(round(end-start))+'s')

## i
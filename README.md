# PyQSOFit
## A code to fit the spectrum of quasar  
### example.ipynb is a demo for the quick start !!!

We provide a brief guide of the Python QSO fitting code (PyQSOFit) to measure spectral properties of SDSS quasars. The code is currently transferred from Yue's IDL to Python. The package includes the main routine, Fe II templates, an input line-fitting parameter list, host galaxy templates, and dust reddening map to extract spectral measurements from the raw fits. Monte Carlo estimation of the measurement uncertainties of the fitting results can be conducted with the same fitting code. 

The code takes an input spectrum (observed-frame wavelength, flux density and error arrays) and the redshift as input parameters, performs the fitting in the restframe, and outputs the best-fit parameters and quality-checking plots to the paths specified by the user. 

The code uses an input line-fitting parameter list to specify the fitting range and parameter constraints of the individual emission line components. An example of such a file is provided in the example.ipynb. Within the code, the user can switch on/off components to fit to the pseudo-continuum. For example, for some objects the UV/optical Fe II emission cannot be well constrained and the user may want to exclude this component in the continuum fit. The code is highly flexible and can be modified to meet the specific needs of the user.

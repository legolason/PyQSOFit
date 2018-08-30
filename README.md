# PyQSOFit
A code to fit the spectrum of quasar

We here provide a brief user’s guide to the spectral fitting code we used to measure spectral properties of SDSS-RM quasars, which is a general-purpose code for quasar spectral fits. The code is currently written in Python. The package includes the main routine, Fe II templates, an input line-fitting parameter file, and ancillary routines used to extract spectral measurements from the fits. Monte Carlo estimation of the measurement uncertainties of the fitting results can be conducted with the same fitting code. 

The code takes an input spectrum (observed-frame wavelength, flux density and error arrays) and the redshift as input parame- ters, performs the fitting in the restframe, and outputs the best-fit parameters and quality-checking plots to the paths specified by the user. 

The code uses an input line-fitting parameter file to specify the fitting range and parameter constraints of the individual emission line components. An example of such a file is provided in the package. Within the code, the user can switch on/off components to fit to the pseudo-continuum. For example, for some objects the UV/optical Fe II emission cannot be well constrained and the user may want to exclude this component in the continuum fit. The code is highly flexible and can be modified to meet the specific needs of the user.

## PyQSOFit: A code to fit the spectrum of quasar  

__See the [example](https://nbviewer.org/github/legolason/PyQSOFit/blob/master/example/example.ipynb) demo notebook for a quick start tutorial__

We provide a brief guide of the Python QSO fitting code (PyQSOFit) to measure spectral properties of SDSS quasars. The code was originally translated from Yue Shen's IDL code to Python. The package includes the main routine, Fe II templates, an input line-fitting parameter list, host galaxy templates, and dust reddening map to extract spectral measurements from the raw fits. Monte Carlo estimation of the measurement uncertainties of the fitting results can be conducted with the same fitting code. 

The code takes an input spectrum (observed-frame wavelength, flux density and error arrays) and the redshift as input parameters, performs the fitting in the restframe, and outputs the best-fit parameters and quality-checking plots to the paths specified by the user. 

The code uses an input line-fitting parameter list to specify the fitting range and parameter constraints of the individual emission line components. An example of such a file is provided in the ``example.ipynb``. Within the code, the user can switch on/off components to fit to the pseudo-continuum. For example, for some objects the UV/optical Fe II emission cannot be well constrained and the user may want to exclude this component in the continuum fit. The code is highly flexible and can be modified to meet the specific needs of the user.

Use this code at your own risk, as we are not responsible for any errors. But if you report bugs, it will be greatly appreciated.

## Install

v1.1 (stable): https://github.com/legolason/PyQSOFit/releases/tag/v1.1

## Known problem

1. Misidentification of line component, between narrow and broad. The code is now using an absolute criteria to determine the broad & narrow emission line component for now (1200 km/s, which is equivalent to sigma=0.0016972 for our model parameters). The fitting limit of the `lmfit` is not exact enough when the parameter hit the boundary. In some cases, the fitted line width of narrow lines will be a little larger than 1200 km/s and being plotted as broad lines and vice versa. We encourage user to leave a gap between the lower width limit of broad lines and higher width limit of narrow lines during their initial setting. This inconvenience will be improved in our following updates.
2. The results fits file might have different name from the given spectrum. It is known there are a very few spectra with MJD documented in their fits files which are different from that used in their file names. For example, for spectrum `spec-0389-51795-0575.fits`, the MJD given in its name is 51795 while the MJD documented in its fits file is 51794. If one follows our example and put the MJD directly from spectral data into our fitting procedure, our code will name the results file differently from that of the spectra file.

## Cite this code

> The preferred citation for this code is Guo, Shen & Wang (2018), ascl:1809:008\
> @misc{2018ascl.soft09008G,\
> author = {{Guo}, H. and {Shen}, Y. and {Wang}, S.},\
> title = "{PyQSOFit: Python code to fit the spectrum of quasars}",\
> keywords = {Software },\
> howpublished = {Astrophysics Source Code Library},\
> year = 2018,\
> month = sep,\
> archivePrefix = "ascl",\
> eprint = {1809.008},\
> adsurl = {[http://adsabs.harvard.edu/abs/2018ascl.soft09008G}](http://adsabs.harvard.edu/abs/2018ascl.soft09008G%7D),\
> adsnote = {Provided by the SAO/NASA Astrophysics Data System}\
> }

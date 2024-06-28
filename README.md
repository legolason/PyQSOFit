## PyQSOFit: A code to fit the spectrum of quasar  

__See the [example](https://nbviewer.org/github/legolason/PyQSOFit/blob/master/example/example.ipynb) demo notebook for a quick start tutorial__

We provide a brief guide of the Python QSO fitting code (PyQSOFit) to measure spectral properties of SDSS quasars. The code was originally translated from Yue Shen's IDL code to Python. The package includes the main routine, Fe II templates, an input line-fitting parameter list, host galaxy templates, and dust reddening map to extract spectral measurements from the raw fits. Monte Carlo estimation of the measurement uncertainties of the fitting results can be conducted with the same fitting code. 

The code takes an input spectrum (observed-frame wavelength, flux density and error arrays) and the redshift as input parameters, performs the fitting in the restframe, and outputs the best-fit parameters and quality-checking plots to the paths specified by the user. 

The code uses an input line-fitting parameter list to specify the fitting range and parameter constraints of the individual emission line components. An example of such a file is provided in the ``example.ipynb``. Within the code, the user can switch on/off components to fit to the pseudo-continuum. For example, for some objects the UV/optical Fe II emission cannot be well constrained and the user may want to exclude this component in the continuum fit. The code is highly flexible and can be modified to meet the specific needs of the user.

Use this code at your own risk, as we are not responsible for any errors. But if you report bugs, it will be greatly appreciated.

## Install

v1.1 (stable): https://github.com/legolason/PyQSOFit/releases/tag/v1.1

## Known problem

1. During the line fitting procedure, some emission lines need less Gaussian models than provided to be well fitted. In these circumstances, the code will give two (or more) exactly the same Gaussian models to fit the data instead of one. This problem will not affect the physical measurements but only wierd in the QA image. One can use BIC to judge the proper number of Gaussian models for each line referencing to our `example.ipynb`.
2. Since the host decomposition hiring BC03 template is non-negative linear fitting, the host component will not stay at zero even if no hosts are detected or the decomposition is failed. We suggest user to determine the reliability of the host decomposition through QA image and the host fraction parameters f_host. (i.e., f_host>0.05 from (Shen et al. 2019)[https://doi.org/10.1088/0004-637X/805/2/96])

## Cite this code


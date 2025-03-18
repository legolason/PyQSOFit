## PyQSOFit: A code to fit the spectrum of quasar  

### Getting Started

__See the [example](https://nbviewer.org/github/legolason/PyQSOFit/blob/master/example/example.ipynb) demo notebook for a quick start tutorial__

We provide a brief guide of the Python QSO fitting code (PyQSOFit) to measure spectral properties of SDSS quasars. The code was originally translated from Yue Shen's IDL code to Python. The package includes the main routine, Fe II templates, an input line-fitting parameter list, host galaxy templates, and dust reddening map to extract spectral measurements from the raw fits. Monte Carlo or MCMC estimation of the measurement uncertainties of the fitting results can be conducted with the same fitting code. 

The code takes an input spectrum (observed-frame wavelength, flux density and error arrays) and the redshift as input parameters, performs the fitting in the restframe, and outputs the best-fit parameters and quality-checking plots to the paths specified by the user. 

The code uses an input line-fitting parameter list to specify the fitting range and parameter constraints of the individual emission line components. An example of such a file is provided in the ``example.ipynb``. Within the code, the user can switch on/off components to fit to the pseudo-continuum. For example, for some objects the UV/optical Fe II emission cannot be well constrained and the user may want to exclude this component in the continuum fit. The code is highly flexible and can be modified to meet the specific needs of the user.

Use this code at your own risk, as we are not responsible for any errors. But if you report bugs, it will be greatly appreciated.

### Install

Bleeding edge:

``git clone https://github.com/legolason/PyQSOFit``<br>
``cd PyQSOFit``<br>
``python -m pip install .``<br>

Pip install (stable version):

``pip install pyqsofit`` (coming soon)

### Cite

Preferred code citation: [Guo, H., Shen, Y., Wang, S. 2018, ascl:1809.008](https://ui.adsabs.harvard.edu/abs/2018ascl.soft09008G/abstract).

Please also cite: [Shen, Y. et al. 2019, ApJS, 241, 34S](https://ui.adsabs.harvard.edu/abs/2019ApJS..241...34S/abstract)

If using new host decompistion tools (`host_prior=True`), please cite: [Ren, W. et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240617598R/abstract)

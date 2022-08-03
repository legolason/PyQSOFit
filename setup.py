#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
#if "release" in sys.argv[-1]:
#    os.system("python setup.py sdist")
#    os.system("python setup.py bdist_wheel")
#    os.system("twine upload dist/*")
#    os.system("rm -rf dist/lightkurve*")
#    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('pyqsofit/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='PyQSOFit',
      version=__version__,
      description="A code to fit the spectrum of quasar.",
      long_description=long_description,
      author='Hengxiao Guo',
      author_email='hengxiaoguo@gmail.com',
      url='https://github.com/legolason/PyQSOFit',
      license='GNU General Public License v3.0',
      package_dir={'pyqsofit': 'pyqsofit'},
      package_data={'pyqsofit': ['fe_uv.txt','fe_optical.txt',
                    'bc03/*.spec.gz',
                    'pca/Yip_pca_templates/*.fits',
                    'sfddata/*.fits']
                   },
      #data_files=[('bc03',['bc03/*.spec.gz']), ('pca/Yip_pca_templates', ['pca/Yip_pca_templates/*.fits']), ('stddata',['stddata/*.fits'])],
      packages=['pyqsofit'],
      install_requires=install_requires,
      include_package_data=True,
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
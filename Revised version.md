## Revised version by Wenke Ren

### Bug Fixing

1. In function `QSOFit._DoLineFit()`, if there is no line to fit while user wanna a figure to plot, an error would be raised due to the undefined `self.gauss_result` et al.
2. The parameter setting in `example.ipynb` constrained the core and the wing component of [OIII] doublet to a single same values instead of two values respectively. 
3. The function `QSOFit._do_tie_line()` could not full fill the function described in `generate_par_file()` in `example.ipynb`. I split the tie procedure of windex to make sure the tie of line width independent from velocity offset.
4. Adjust the code style according to [google python style guide](https://google.github.io/styleguide/pyguide.html).

### New Features

1. Regulate the form of the output data instead of a string for all.
2. Add the normalization factor and velocity shift of FeII template and the normalization factor and $\tau_e$ of Balmer continuum into output, so that one can rebuild the fitting model acconting the output result.
3. FeII flux calculation. An optional function which is compatible with former programs.
4. **(High Risk): **Overall re-wright the `QSOFit._PlotFig()` for the convenience of modification and add a $\chi^2$ parameter on the left top.

4. Move all function inside the class `QSOFit`.


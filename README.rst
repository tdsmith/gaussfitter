Gaussfitter
===========

This is a fork of Adam Ginsburg's gaussfitter.py tool that supports fitting to masked arrays.
Fitting to masked arrays is somewhat slower than fitting to unmasked arrays.

This code is taken from agpy, where it has resided for a long time and has had
a long, glorious history.

In short: This is a small toolkit for fitting 2D gaussians.  It makes use of
mpfit.py by Sergei Koposov
(https://code.google.com/p/astrolibpy/source/browse/), and a modified version
of his code is included (by necessity) here.  It is modified primarily to
remove a scipy dependency.

Examples to come!  PRs welcome!

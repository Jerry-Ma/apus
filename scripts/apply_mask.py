#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-06-13 13:51
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
apply_mask.py

Apply mask to images
"""

from astropy.io import fits
import numpy as np

if __name__ == "__main__":
    import sys
    in_, wht, out = sys.argv[1:4]
    hlin = fits.open(in_)
    hlwht = fits.open(wht)
    for i in range(len(hlin)):
        badmask = hlwht[i].data == 0
        hlin[i].data[badmask] = np.nan
    hlin.writeto(out, clobber=True)

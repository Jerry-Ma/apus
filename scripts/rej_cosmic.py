#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-06-11 14:29
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
rej_cosmic.py

L.A cosmic wrapper for cosmic ray removal
"""

import numpy as np
from _lacosmicx import lacosmicx
from astropy.io import fits


conf = dict(
    sigclip=4.5, sigfrac=0.3, objlim=5.0, satlevel=50000.0,
    pssl=0.0, niter=4, sepmed=True, cleantype='medmask',
    fsmode='median',
        )
conf_cfht = dict(
    gain=1.585, readnoise=5,
        )


if __name__ == "__main__":
    import sys
    in_ = sys.argv[1]
    bpm = sys.argv[2]
    out = sys.argv[3]
    debug = "[la cosmic] {0} -> {1}".format(in_, out)
    print debug

    kwargs = dict(conf, **conf_cfht)
    hdulist = fits.open(in_)
    bpmlist = fits.open(bpm)
    for i in range(1, len(hdulist)):
        print "working on ext {0}".format(i)
        mask = 1 - bpmlist[i].data.astype(np.uint8)
        _, clean = lacosmicx(hdulist[i].data, inmask=mask,
                             **kwargs)
        hdulist[i].data = clean / conf_cfht['gain']
    hdulist.writeto(out, clobber=True)

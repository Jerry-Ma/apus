#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-06-11 20:36
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
create_refcat.py

Create reference catalog from ascii table
"""

from astropy.io import fits
import numpy as np
# import astromatic_wrapper as aw
from astropy.table import Table
from astropy.table import Column
import sys
# import os


if __name__ == "__main__":
    band = sys.argv[1]
    lolim, uplim = map(float, sys.argv[2:4])
    in_ = sys.argv[4]
    inwcs = sys.argv[5]
    out = sys.argv[6]
    # inwcs = '{0}.wcs'.format(os.path.splitext(in_)[0])
    tbl = Table.read(in_, format='ascii.commented_header')
    obsdate = Column([2000.2586] * len(tbl), name='OBSDATE')
    tbl.add_column(obsdate)
    tbl.add_column(Column([0.05, ] * len(tbl), name='emag'))
    tbl['ra_error'] = tbl['ra_error'] / 3600.
    tbl['dec_error'] = tbl['dec_error'] / 3600.

    mag = 'phot_g_mean_mag'
    emag = 'emag'

    key = ['ra', 'dec', 'ra_error', 'dec_error', mag, emag, 'OBSDATE']
    outkey = ['X_WORLD', 'Y_WORLD', 'ERRA_WORLD', 'ERRB_WORLD',
              'MAG', 'MAGERR', 'OBSDATE']
    outfmt = ['1D', '1D', '1E', '1E', '1E', '1E', '1D']
    outunt = ['deg', 'deg', 'deg', 'deg', 'mag', 'mag', 'yr']

    tbl = tbl[key]
    for k, o in zip(key, outkey):
        if k != o:
            tbl.rename_column(k, o)
    mlolim = 0.001
    muplim = np.log10(np.e) * 2.5 / 10.
    tbl = tbl[(tbl['MAG'] > lolim) & (tbl['MAG'] < uplim) &
              (tbl['MAGERR'] > mlolim) & (tbl['MAGERR'] < muplim)
              ]
    # create date hdu
    datahdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=k, format=f, array=tbl[k], unit=u)
         for k, f, u in zip(outkey, outfmt, outunt)])
    datahdu.header['EXTNAME'] = 'LDAC_OBJECTS'
    # create imhead hdu
    with open(inwcs, 'r') as fo:
        tblhdr = np.array([''.join(fo.readlines()), ])
    len_ = len(tblhdr[0])
    col = fits.Column(name='Field Header Card',
                      array=tblhdr, format='{0:d}A'.format(len_))
    cols = fits.ColDefs([col])
    headhdu = fits.BinTableHDU.from_columns(cols)
    headhdu.header['TDIM1'] = '(80, {0})'.format(len_ / 80)
    headhdu.header['EXTNAME'] = 'LDAC_IMHEAD'
    hdulist = fits.HDUList([fits.PrimaryHDU(), headhdu, datahdu])
    hdulist.writeto(out, clobber=True)
    # hdulist = fits.open('./sdssref_template.cat')
    # print hdulist[-1].data.dtype
    # print len(tbl)
    # print thl[-1].data.dtype
    # hdulist[-1].data = thl[-1].data.astype(hdulist[-1].data.dtype)
    # hdulist.writeto(out, clobber=True)

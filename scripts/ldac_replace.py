#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-08-11 09:21
# Python Version :  2.7.12
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
ldac_replace.py
"""

# import numpy as np
from astropy.table import Table
from astropy.io import fits


def read_table(fname):
    hdulist = fits.open(fname)
    exts = range(2, len(hdulist), 2)
    ret = []
    for i in exts:
        # print "read ext {0}".format(i / 2)
        ret.append(Table.read(hdulist[i]))
    hdulist.close()
    return ret


def read_ascii_table(fname, n_ext):
    tbl = Table.read(fname, format='ascii.commented_header')
    ret = []
    for ext in range(1, n_ext + 1):
        print "extracting ext {0}".format(ext)
        ret.append(tbl[tbl['EXT_NUMBER'] == ext])
    return ret


def get_ldac_object(tbl):
    outkey = ['X_WORLD', 'Y_WORLD', 'ERRA_WORLD', 'ERRB_WORLD',
              'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
              'ERRTHETAWIN_IMAGE',
              'FLUX_AUTO', 'FLUXERR_AUTO', 'FLAGS']
    outfmt = ['1D', '1D', '1E', '1E',
              '1D', '1D', '1D', '1D', '1D',
              '1D', '1D', '1I']
    outunt = ['deg', 'deg', 'deg', 'deg',
              'deg', 'deg', 'deg', 'deg', 'deg',
              'deg', 'deg', None]
    # TODO figure out why is the sex catalog column name truncated
    fix_colname = None if 'ERRAWIN_IMAGE' in tbl.colnames else 15
    datahdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=k, format=f, array=tbl[k[:fix_colname]], unit=u)
         for k, f, u in zip(outkey, outfmt, outunt)])
    datahdu.header['EXTNAME'] = 'LDAC_OBJECTS'
    return datahdu


def save_mef(fname, replace, outname):
    hdulist = fits.open(fname)
    n_ext = len(hdulist) / 2
    print "number of extensions {0}".format(n_ext)
    meftbl = read_ascii_table(replace, n_ext)
    for i, j in enumerate(range(2, len(hdulist), 2)):
        # thl = aw.utils.ldac.convert_table_to_ldac(meftbl[i])
        ldac_object = get_ldac_object(meftbl[i])
        hdulist[j] = ldac_object
    # ret = None
    # for tbl in meftbl:
    #     hdulist = aw.utils.ldac.convert_table_to_ldac(tbl)
    #     if ret is None:
    #         ret = hdulist
    #     else:
    #         ret.extend(hdulist[1:])
    # ret.writeto(fname, clobber=True)
    hdulist.writeto(outname, clobber=True)


if __name__ == "__main__":
    import sys
    try:
        radius = float(sys.argv[-1])
        sys.argv.pop()
    except ValueError:
        radius = 1.5
    in_file, replace, out = sys.argv[1:]
    print "[ldac replace] {0}".format(in_file)
    save_mef(in_file, replace, out)

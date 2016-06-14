#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-04-18 23:46
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
ldac_subtract.py
"""

import numpy as np
import astromatic_wrapper as aw
from astropy.coordinates import SkyCoord
from astropy import units as u
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


def get_radec_colname(tbl):
    colnames = map(str.upper, tbl.colnames)
    for colname in colnames:
        if 'ALPHA_J2000' == colname and 'DELTA_J2000' in colnames:
            ra, dec = 'ALPHA_J2000', 'DELTA_J2000'
            break
        elif 'RA' == colname[:2] and 'DEC' + colname[2:] in colnames:
            ra, dec = colname, 'DEC' + colname[2:]
        else:
            raise ValueError('cannot guess RA Dec column for catalog'
                             'with header: {0}'.format(tbl.colnames))
    return ra, dec


def subtraction(in_, rejects, radius=1.5):
    in_ = read_table(in_)
    rejects = map(read_table, rejects)
    for i in range(len(in_)):
        # print "processing ext {0}".format(i + 1)
        if len(in_[i]) == 0:
            continue
        mask = np.ones_like(in_[i], dtype=bool)
        colra, coldec = get_radec_colname(in_[i])
        in_coord = SkyCoord(ra=in_[i][colra], dec=in_[i][coldec],
                            unit=u.degree)
        for rejtbl in rejects:
            if len(rejtbl[i]) == 0:
                continue
            colra_rej, coldec_rej = get_radec_colname(rejtbl[i])
            rej_coord = SkyCoord(ra=rejtbl[i][colra_rej],
                                 dec=rejtbl[i][coldec_rej],
                                 unit=u.degree)
            id_rej, id_in, _, _ = in_coord.search_around_sky(rej_coord,
                                                             radius * u.arcsec)
            mask[id_in] = False
        in_[i] = in_[i][mask]
    return in_


def save_mef(fname, meftbl, outname):
    hdulist = fits.open(fname)
    for i, j in enumerate(range(2, len(hdulist), 2)):
        thl = aw.utils.ldac.convert_table_to_ldac(meftbl[i])
        hdulist[j] = thl[-1]
    ret = None
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
    in_ = sys.argv[1]
    rejects = sys.argv[2:-1]
    out = sys.argv[-1]
    debug = "[ldac subtract] {0}".format(in_)
    for r in rejects:
        debug += ' - {0}'.format(r)
    debug += ' = {0}'.format(out)
    print debug
    save_mef(in_, subtraction(in_, rejects, radius=radius), out)

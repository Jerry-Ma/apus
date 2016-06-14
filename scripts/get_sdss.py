#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-06-11 17:11
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
get_sdss.py

retreive SDSS catalog for astrometry
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import fits
import os
import sys
from astroquery.sdss import SDSS


def cmp_ra(ra1, ra2):
    # west to east, shorted path
    if ra1 - ra2 > 180.:
        ra1 -= 360.
    elif ra1 - ra2 < -180.:
        ra1 += 360.
    return int(ra1 > ra2) - int(ra1 < ra2)


def merge_box(box1, box2):
    if box1 is None:
        return box2
    elif box2 is None:
        return box1
    else:
        l1, r1, b1, t1 = box1
        l2, r2, b2, t2 = box2
        ra = [l1, l2, r1, r2]
        ramin, _, _, ramax = sorted(ra, cmp=cmp_ra)
        dec = [b1, b2, t1, t2]
        decmin, _, _, decmax = sorted(dec)
        return ramin, ramax, decmin, decmax


def get_box(image, inst='cfht'):
    print 'image: {0}'.format(image)
    hdulist = fits.open(image)
    target = SkyCoord(hdulist[0].header['RA'], hdulist[0].header['DEC'],
                      unit=(u.hourangle, u.deg))
    hdulist.close()
    if inst == 'cfht':
        dra = ddec = 0.55
    else:
        raise ValueError("no instrument specification found")
    l = target.ra.degree - dra / np.cos(target.dec.radian)
    if l < 0:
        l = 360. + l
    r = target.ra.degree + dra / np.cos(target.dec.radian)
    if r > 360.:
        r = r - 360.
    t = target.dec.degree - ddec
    b = target.dec.degree + ddec
    print "image bbox: {0}, {1} {2}, {3}".format(l, r, t, b)
    return [l, r, t, b], \
           [target.ra.degree, target.dec.degree, dra * 2, ddec * 2]


sql_query = """\
SELECT ra,dec,raErr,decErr, u, err_u, g, err_g, r, err_r, i, err_i, z, err_z
FROM Star
WHERE
ra BETWEEN %(min_ra)f and %(max_ra)f
AND dec BETWEEN %(min_dec)f and %(max_dec)f
AND ((flags_r & 0x10000000) != 0)
AND ((flags_r & 0x8100000c00a4) = 0)
AND (((flags_r & 0x400000000000) = 0) or (err_%(band)s <= 0.2))
AND (((flags_r & 0x100000000000) = 0) or (flags_r & 0x1000) = 0)
"""
# AND %(band)s BETWEEN %(min_mag)f and %(max_mag)f

if __name__ == "__main__":

    # band = sys.argv[1]
    in_ = sys.argv[1:-1]
    out = sys.argv[-1]
    outreg = '{0}.reg'.format(os.path.splitext(out)[0])
    outasc = '{0}.asc'.format(os.path.splitext(out)[0])
    outwcs = '{0}.wcs'.format(os.path.splitext(out)[0])
    box = None
    with open(outreg, 'w') as fo:
        fo.write('global color=red\n')
        for image in in_:
            ibox, rbox = get_box(image)
            fo.write('fk5; box({0},{1},{2},{3}, 0)\n'.format(*rbox))
            box = merge_box(box, ibox)
        # get final box
        cra = (box[0] + box[1]) / 2
        cdec = (box[2] + box[3]) / 2
        width = (box[1] - box[0]) * np.cos(cdec * np.pi / 180.)
        height = box[3] - box[2]
        fo.write('fk5; box({0},{1},{2},{3}, 0) # color=cyan'.format(
            cra, cdec, width, height))

    kwargs = {
            "min_ra": box[0], "max_ra": box[1],
            "min_dec": box[2], "max_dec": box[3],
            "min_mag": 17.5, "max_mag": 22.5,
            "band": 'r',
            }
    fsql = sql_query % kwargs
    print fsql
    stdstar = SDSS.query_sql(fsql, data_release=12)
    print stdstar
    stdstar.write(outasc, format='ascii.commented_header')
    # create dummy wcs
    hdulist = fits.open(in_[0])
    ow = wcs.WCS(hdulist[1].header)
    hdulist.close()
    ps = abs(ow.pixel_scale_matrix[0, 0])
    ncol = width / ps
    nrow = height / ps
    ow.wcs.crpix = [ncol / 2, nrow / 2]
    # ow.wcs.cdelt = np.diag(ow.pixel_scale_matrix)
    ow.wcs.crval = [cra, cdec]
    ow.wcs.ctype = ["RA---STG", "DEC--STG"]
    header = ow.to_header()
    # for key in ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']:
    #     header[key.replace('PC', 'CD')] = header[key]
    header['CD1_1'] = ps
    header['CD2_2'] = -ps
    header['CD1_2'] = 0
    header['CD2_1'] = 0
    keepkeys = ['EQUINOX', 'RADESYS',
                'CTYPE1', 'CUNIT1', 'CRVAL1', 'CRPIX1', 'CD1_1', 'CD1_2',
                'CTYPE2', 'CUNIT2', 'CRVAL2', 'CRPIX2', 'CD2_1', 'CD2_2',
                ]
    for key in header.keys():
        if key not in keepkeys:
            del header[key]
    header.insert(0, ('EXTEND', True))
    header.insert(0, ('NAXIS2', int(nrow + 0.5)))
    header.insert(0, ('NAXIS1', int(ncol + 0.5)))
    header.insert(0, ('NAXIS', 2))
    header.insert(0, ('BITPIX', 0))
    header.insert(0, ('SIMPLE', True))
    with open(outwcs, 'w') as fo:
        fo.write(header.tostring())

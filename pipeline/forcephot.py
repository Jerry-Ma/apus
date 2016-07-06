#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-06-15 14:16
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
forcephot.py

Perform forced photometry at given source list using SExtractor
"""

import os
import re

jobkey = 'force_phot'
jobdir = jobkey
confdir = os.path.join(jobdir, 'config')

inputs = ["/home/ma/Codes/Hermes_dr2/L2-COSMOS/L2-COSMOS_image_SMAP250_DR2.fits", ]
input_reg = r'.+/L2-(?P<imgkey>\w+)_image_SMAP(?P<band>\d+)_.+\.fits'
input_fmt = os.path.join(jobdir, '{imgkey[0]}{band[0]}.sci.fits')

srclist = '../test/srclist.asc'
srclist_fmt = os.path.join(jobdir, '{imgkey:s}{band:s}.asc')
weight_fmt = os.path.join(jobdir, '{imgkey:s}{band:s}.wht.fits')

# extra = [
#         ]
# extra_reg = r'.+/(?P<exkey>[A-Za-z]+)(\w+)?\..+'
# extra_fmt = os.path.join(confdir, '{exkey[0]}{ext[0]}')


# function that takes input, and returns (extra, extra_link_name)
def get_srclist(image):
    imgkey = re.match(input_fmt, image).group('imgkey')
    imgband = re.match(input_fmt, image).group('band')
    return (srclist, srclist_fmt.format(imgkey=imgkey, band=imgband))
per_input_extras = [get_srclist, ]


# am_sharedir = None
# am_diagdir = None
# am_resampdir = None
am_params = {
    'sex': {
        'CATALOG_TYPE': 'ASCII_HEAD',
        'MAG_ZEROPOINT': '0',
        'DETECT_MINAREA': '1',
        'DETECT_MAXAREA': '1',
        'DETECT_THRESH': '1.5',
        'DEBLEND_MINCONT': '1.0',
        'FILTER': 'N',
        'CLEAN': 'N',
        },
    }

# ------------------- #
# PIPELINE DEFINITION #
# ------------------- #
imreg = r'.+/(?P<imgkey>[^_]+)'
imglob = '*.sci.fits'

t00 = dict(
        name='create detect image',
        func='./scripts/create_pseudo_image {in} {out}',
        type_='collate',
        in_=os.path.join(jobdir, imglob),
        reg=imreg + r'\.sci\.fits',
        out=os.path.join(jobdir, '{imgkey[0]}.det.fits'),
        )
t01 = dict(
        name='sex',
        func='sex',
        type_='transform',
        in_=t00['name'],
        reg=imreg + r'\.det\.fits',
        add_inputs=[os.path.join(jobdir, r'{imgkey[0]}.sci.fits'),
                    os.path.join(jobdir, r'{imgkey[0]}.wht.fits'), ],
        params={'WEIGHT_TYPE': 'MAP_RMS', },
        out=os.path.join(jobdir, '{imgkey[0]}.cat.fits'),
        )
t02 = dict(
        name='to ascii',
        func='./scripts/ldac_to_asc.sh {in} {out}',
        type_='transform',
        in_=t01['name'],
        reg=imreg + r'\.cat\.fits',
        out=os.path.join(jobdir, r'{imgkey[0]}.asci'),
        )
t20 = dict(
        name='scamp 1st pass',
        func='scamp_1st',
        type_='collate',
        in_=[t11['name'], t01['name']],  # image, refcat
        reg='.+',
        params={'ASTREF_CATALOG': 'FILE', },
        out=os.path.join(jobdir, 'mergedref.cat'),
        outkey='MERGEDOUTCAT_NAME',
        )
t21 = dict(
        name='scamp 2nd pass',
        func='scamp_2nd',
        type_='collate',
        reg='.+',
        in_=[t11['name'], t20['name']],
        params={'ASTREF_CATALOG': 'FILE', },
        out=None,
        )
# t22 = dict(
#         name='softlink header',
#         func='ln -sf {in} {out}',
#         type_='transform',
#         follows=t21['name'],
#         in_=os.path.abspath(os.path.join(jobdir, 'astro_*p.head2')),
#         reg='.+/astro_' + imreg.replace('.fits', '.head2'),
#         out=os.path.join(jobdir, '{imgkey[0]}.head2'),
#         )
t30 = dict(
        name='swarp',
        func='swarp',
        type_='collate',
        follows=t21['name'],
        in_=os.path.join(jobdir, imglob),
        reg='.+/' + imreg,
        add_inputs=[os.path.join(jobdir, r'{imgkey[0]}.bpm.fits'),
                    os.path.join(jobdir, r'astro_{imgkey[0]}.head2'), ],
        out=[os.path.join(jobdir, 'coadd_{0}.fits'.format(jobkey)),
             os.path.join(jobdir, 'coadd_{0}.weight.fits'.format(jobkey)), ],
        params={'WEIGHT_TYPE': 'MAP_WEIGHT'}
        )
t31 = dict(
        name='fix nan pixel',
        func='./scripts/apply_mask.py {in} {out}',
        type_='transform',
        in_=t30['name'],
        reg='.+',
        out=os.path.join(jobdir, 'fixed_coadd_{0}.fits'.format(jobkey)),
        )

tlist = [t00, t01, t10, t11, t12, t20, t21, t30, t31]

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/ma/Codes')
    from apus import core
    core.bootstrap()

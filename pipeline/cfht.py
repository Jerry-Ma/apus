#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-05-23 19:08
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
cfht.py

Defines pipeline for reducing CFHT FLS data
"""

import os
import re
from astropy.io import fits


jobkey = 'cfht_fls_g'
jobdir = jobkey
confdir = os.path.join(jobdir, 'config')

inputs = ["../stackpipe_old/test_images/*.fits", ]
input_reg = r'.+/(?P<imgkey>\d+p)\.fits'
input_out = os.path.join(jobdir, '{imgkey[0]}.fits')

extra_dir = '/home/ma/Codes/CFHT/ancillary'
extra = [
        os.path.join(extra_dir, 'edgemask.fits'),
        # os.path.join(extra_dir, 'megacam.ret'),
        ]
extra_reg = r'.+/(?P<exkey>[A-Za-z]+)(\w+)?\..+'
extra_out = os.path.join(confdir, '{exkey[0]}{ext[0]}')


# function that takes input, and returns (extra, extra_link_name)
def get_weight(image):
    hdulist = fits.open(image, memmap=True)
    maskfile = hdulist[0].header['IMRED_MK'].split('[')[0] + '.fits'
    hdulist.close()
    # get link name
    imgkey = re.match(input_reg, image).group('imgkey')
    return os.path.join(extra_dir, maskfile), \
        os.path.join(jobdir, '{0}.bpm.fits'.format(imgkey))
per_input_extra = [get_weight, ]

# am_sharedir = None
# am_diagdir = None
# am_resampdir = None
refcatband = jobkey.rsplit('_', 1)[-1]
refcatlim = (-99, 99)
refcatkey = 'sdss'
am_zero = 25.
am_params = {
    'sex': {
        'MAG_ZEROPOINT': '{0:f}'.format(am_zero),
        },
    'sex_astro': {
        'CATALOG_TYPE': 'FITS_LDAC',
        'FILTER_NAME': 'gauss_5.0_9x9.conv',
        'DETECT_MINAREA': '3',
        'DETECT_THRESH': '5.0',
        },
    'scamp': {
        'MAGZERO_OUT': '{0:f}'.format(am_zero),
        'AHEADER_GLOBAL': 'megacam.ahead',
        'MOSAIC_TYPE': 'SAME_CRVAL',
        },
    'scamp_1st': {
        'MERGEDOUTCAT_TYPE': 'FITS_LDAC',
        'MERGEDOUTCAT_NAME': os.path.join(jobdir, 'mergedref.cat'),
        'AHEADER_SUFFIX': '.none',
        'HEADER_SUFFIX': '.head1',
        'XML_NAME': os.path.join(jobdir, 'scamp_1st.xml'),
        },
    'scamp_2nd': {
        'ASTREFCENT_KEYS': 'ALPHA_J2000,DELTA_J2000',
        'AHEADER_SUFFIX': '.none',
        'HEADER_SUFFIX': '.head2',
        'XML_NAME': os.path.join(jobdir, 'scamp_2nd.xml')
        },
    'swarp': {
        'HEADER_SUFFIX': '.head2',
        'WEIGHT_SUFFIX': '.none',
        'CENTER_TYPE': 'ALL',
        # 'CENTER': ' 17:16:12.744,+59:22:58.72',
        'PIXELSCALE_TYPE': 'MANUAL',
        'PIXEL_SCALE': '0.186',
        # 'PIXEL_SCALE': '6.0',
        # 'IMAGE_SIZE': '1971,1714'
        'XML_NAME': os.path.join(jobdir, 'swarp.xml')
        },
    }

# ------------------- #
# PIPELINE DEFINITION #
# ------------------- #
imreg = r'(?P<imgkey>\d+p)\.fits'
imglob = '[0-9]*p.fits'

t00 = dict(
        name='get sdss',
        func='./scripts/get_sdss.py {in} {out}',
        type_='collate',
        in_=os.path.join(jobdir, imglob),
        reg='.+/' + imreg,
        out=os.path.join(jobdir, '{0}.asc'.format(refcatkey)),
        )
t01 = dict(
        name='create refcat',
        func='./scripts/create_refcat.py {0} {1} {2} {{in}} {{out}}'.format(
            refcatband, *refcatlim),
        type_='transform',
        in_=t00['name'],
        reg='.+/(?P<name>.+)\.asc',
        out=os.path.join(jobdir, r'{name[0]}.fits'),
        )
t10 = dict(
        name='remove cosmic ray',
        func='./scripts/rej_cosmic.py {in} {out}',
        type_='transform',
        in_=os.path.join(jobdir, imglob),
        reg='.+/' + imreg,
        add_inputs=os.path.join(jobdir, r'{imgkey[0]}.bpm.fits'),
        out=os.path.join(jobdir, r'clean_{imgkey[0]}.fits'),
        )
t11 = dict(
        name='sex for astrometry',
        func='sex_astro',
        type_='transform',
        in_=t10['name'],
        reg='.+/clean_' + imreg,
        add_inputs=os.path.join(jobdir, r'{imgkey[0]}.bpm.fits'),
        params={'WEIGHT_TYPE': 'MAP_WEIGHT', },
        out=os.path.join(jobdir, r'astro_{imgkey[0]}.fits'),
        )
t12 = dict(
        name='to ascii',
        func='./scripts/ldac_to_asc.sh {in} {out}',
        type_='transform',
        in_=t11['name'],
        reg='.+/(?P<name>.+)\.fits',
        out=os.path.join(jobdir, r'{name[0]}.asc'),
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

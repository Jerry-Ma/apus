#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-05-21 17:14
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
env.py

Paths to the astromatic executables and related directories
"""

import os


class AmConfig(object):

    sexparam_default = [
        # coord
        'ALPHA_J2000', 'DELTA_J2000', 'X_IMAGE', 'Y_IMAGE',
        'NUMBER', 'EXT_NUMBER',
        # phot
        'MAG_AUTO', 'MAGERR_AUTO', 'MAG_APER', 'MAGERR_APER',
        'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER',
        'BACKGROUND', 'THRESHOLD',
        # scamp
        'XWIN_IMAGE', 'YWIN_IMAGE',
        'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE', 'ERRTHETAWIN_IMAGE',
        'FLAGS', 'FLAGS_WEIGHT', 'FLAGS_WIN',  # 'IMAFLAGS_ISO',
        'FLUX_RADIUS',
        # ref key
        'X_WORLD', 'Y_WORLD',
        'ERRA_WORLD', 'ERRB_WORLD', 'ERRTHETA_WORLD',
        # PSF shape
        'FWHM_WORLD', 'FWHM_IMAGE',
        'A_IMAGE', 'B_IMAGE',
        'THETA_IMAGE', 'ELLIPTICITY',
        'CLASS_STAR'
        ]
    sex_default = {
            'FILTER_NAME': 'default.conv',
            'STARNNW_NAME': 'default.nnw',
            'WRITE_XML': 'N',
            'BACKPHOTO_TYPE': 'LOCAL',
            'PIXEL_SCALE': 0,
            'HEADER_SUFFIX': '.none',
            'GAIN_KEY': 'bug_of_sex_219',
            'NTHREADS': 0,
            }
    scamp_default = {
            'CHECKPLOT_RES': '1024',
            'SAVE_REFCATALOG': 'Y',
            'WRITE_XML': 'N',
            }
    swarp_default = {
            'INTERPOLATE': 'N',
            'FSCALASTRO_TYPE': 'VARIABLE',
            'DELETE_TMPFILES': 'N',
            'NOPENFILES_MAX': '1000000',
            'WRITE_XML': 'N',
            }
    scratch_dir = '/tmp'
    path_prefix = '/usr'

    def __init__(self, **kwargs):
        """populate properties, with optional overrides from kwargs"""
        self.set_overrides(kwargs)

    def get(self, prop):
        return self.overrides.get(prop, getattr(self, prop))

    def set_overrides(self, overrides):
        self.overrides = overrides
        self.share_dir = os.path.join(self.get('path_prefix'), 'share')
        self.bin_dir = os.path.join(self.get('path_prefix'), 'bin')
        for i in ['bin', 'share']:
            for sname, lname in [('sex', 'sextractor'),
                                 ('scamp', 'scamp'),
                                 ('swarp', 'swarp')]:
                if sname == 'sex' and i == 'bin':
                    lname = 'sex'  # sextractor binary naming
                setattr(self, '{0}{1}'.format(sname, i),
                        os.path.join(self.get('{0}_dir'.format(i)), lname))


am = AmConfig()

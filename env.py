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


scratch = '/mnt/Scratch/swarp_resamp'

# astromatic
am = {
    'root': '/home/ma/Codes/astromatic',
    }
am['share'] = os.path.join(am['root'], 'share')
am['sexbin'] = os.path.join(am['root'], 'bin/sex')
am['scampbin'] = os.path.join(am['root'], 'bin/scamp')
am['swarpbin'] = os.path.join(am['root'], 'bin/swarp')
am['sexshare'] = os.path.join(am['share'], 'sextractor')
am['scampshare'] = os.path.join(am['share'], 'scamp')
am['swarpshare'] = os.path.join(am['share'], 'swarp')

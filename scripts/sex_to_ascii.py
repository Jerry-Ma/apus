#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-07-26 20:01
# Python Version :  2.7.12
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
sex_to_ascii.py
"""


if __name__ == "__main__":
    import sys
    from astropy.table import Table
    tbl = Table.read(sys.argv[1], format='ascii.sextractor')
    tbl.write(sys.argv[2], format='ascii.commented_header')

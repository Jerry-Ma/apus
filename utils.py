#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2015-12-11 23:27
# Python Version :  %PYVER%
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
utils.py

Provide various helper functions
"""

import re
import os
import glob
import logging
import logging.config


def update_progress(mesg, perc):
    print "\r{0:8s}: [{1:40s}] {2:.1f}%".format(
        mesg,
        '#' * int(perc * 40),
        perc * 100),


def alert(string):
    """highlight string in terminal"""
    return '\033[91m{0}\033[0m'.format(string)


def de_alert(string):
    """remove any escape sequence"""
    return re.sub(r'\033\[\d+m', '', string)


def get_or_create_dir(p):

    logger = logging.getLogger(__name__)
    if not os.path.isdir(p):
        os.makedirs(p)
        logger.info("+ {0:s}".format(p))
    return os.path.abspath(p)


def replace_dir(fname, dirname):
    bname = os.path.basename(fname)
    return os.path.join(dirname, bname)


def get_scamp_checkplot_name(dirname, prefix=''):
    scamp_checkplot = ['fgroups', 'distort', 'astr_interror2d',
                       'astr_interror1d', 'astr_referror2d',
                       'astr_referror1d', 'astr_chi2', 'psphot_error']
    return ','.join([os.path.join(dirname, '{0}_{1}'.format(prefix, i))
                     for i in scamp_checkplot])


def parse_astromatic_conf(*conf_files):
    params = {}
    for fname in conf_files:
        with open(fname, 'r') as fo:
            for ln in fo.readlines():
                ln = ln.strip()
                if ln == '' or ln.startswith('#'):
                    continue
                else:
                    key, rest = map(str.strip, ln.split(None, 1))
                    value = rest.split('#')[0]
                    params[key] = value
    return params


def dump_astromatic_conf(infile, outfile, clobber=False, **kwargs):

    logger = logging.getLogger(__name__)
    if os.path.isfile(outfile) and not clobber:
        raise ValueError("file exist:{0}".format(outfile))
    with open(outfile, 'w') as fo:
        for oln in infile.readlines():
            ln = oln.strip()
            if len(ln) == 0 or ln.startswith("#"):
                fo.write(oln)
                continue
            else:
                try:
                    keyval = map(str.strip, ln.split('#', 1))[0]
                except ValueError:
                    keyval = ln
                try:
                    key, val = map(str.strip, keyval.split(None, 1))
                except ValueError:
                    key, val = keyval, None
                newval = kwargs.get(key, None)
                if newval is None:
                    fo.write(oln)
                    continue
                else:
                    if val is None:
                        fo.write(oln.replace(key, '{0}  {1}'.format(
                            key, newval)))
                    else:
                        # only replace the value part
                        iv = oln.index(key) + len(key)
                        if '#' in oln:
                            jv = oln.index('#')
                        fo.write(oln[:iv] + oln[iv:jv].replace(val, newval) +
                                 oln[jv:])
    logger.info("+> {0:s}".format(outfile))
    return outfile


def dump_sex_param(infile, outfile, *args, **kwargs):

    logger = logging.getLogger(__name__)
    err = 'Not a valid output para: {0:s}'
    if os.path.isfile(outfile) and not kwargs.get('clobber', False):
        raise ValueError("file exist:{0}".format(outfile))
    content = infile.getvalue()
    # merge common with args: handle array-like keys
    re_key = re.compile('(\w+)\s*(\(\s*\d+\s*\))?')
    keys = []
    for arg in args:
        for key in arg:
            match = re.match(re_key, key.strip())
            if match is not None:
                _key = match.groups()[0]
                if _key not in content:
                    raise ValueError(err.format(key))
                if _key in keys:
                    keys.index(_key)
                    keys[keys.index(_key)] = key
                else:
                    keys.append(key)
            else:
                raise ValueError(err.format(key))
    logger.info('output params: {0}'.format(', '.join(keys)))
    with open(outfile, 'w') as fo:
        for key in keys:
            fo.write('{0:23s}  #\n'.format(key))
        fo.write('#' * 26 + '\n')
        fo.write(content)
    logger.info("+> {0:s}".format(outfile))
    return keys


def sorted_glob(pattern):

    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('([0-9]+)', s)]
    print pattern
    return sorted(glob.glob(pattern), key=alphanum_key)


def init_logging():

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'short': {
                'format': '[%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'short',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False
            },
        }
    })

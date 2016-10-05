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
import sys
import glob
import core
import inspect
import logging
import logging.config
from ruffus import needs_update_check_modify_time


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
                    value = map(str.strip, rest.split('#'))[0]
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


def join_tlist(*args):
    if len(args) == 2:
        p1, p2 = args
        return p1 + (dict(p2[0], follows=p1[-1]), ) + p2[1:]
    elif len(args) < 2:
        raise RuntimeError("need at least two tlists to join")
    else:
        return join_tlist(join_tlist(*args[:-1]), args[-1])


def get_main_config():
    '''useful when split config file to several modules'''
    return sys.modules['__main__']


def default_to_main_config(func):
    '''replace function args with main config keys if not specified'''
    def wrapped_func(*args, **kwargs):
        conf = get_main_config()
        args = list(args)
        fargs = inspect.getargspec(func).args
        while len(args) < len(fargs):
            args.append(None)
        for i, (a, f) in enumerate(zip(args, fargs)):
            if a is None and hasattr(conf, f):
                fargs[i] = getattr(conf, f)
            else:
                fargs[i] = a
        return func(*fargs)
    return wrapped_func


def tlist_wrapper(tlist, outglob, outreg):
    return ApusTaskList(tlist, outglob, outreg)


class ApusTaskList(object):
    '''provide interface for chaining multiple task lists'''

    def __init__(self, tlist, outglob, outreg):

        self.config = get_main_config()
        if hasattr(self.config, 'tlist'):
            head_follows = core.ensure_list(tlist[0].get('follows', None))
            head_follows.append(self.config.tlist[-1])
            head_task = dict(tlist[0], follows=head_follows)
            self.config.tlist.append(head_task)
            self.config.tlist.extend(tlist[1:])
        else:
            self.config.tlist = tlist
        self.outglob = outglob
        self.outreg = outreg

    def chain(self, tlist_func):
        other = tlist_func(self.outglob, self.outreg)
        return other


def apus_check_if_uptodate(*args, **kwargs):
    conf = get_main_config()
    return needs_update_check_modify_time(
            *args, job_history=conf.history_file, **kwargs)

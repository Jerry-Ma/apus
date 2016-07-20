#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-05-19 23:23
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
core.py

"""

from __future__ import print_function
import os
import re
import sys
import glob
import time
import logging
import StringIO
import subprocess
from functools import wraps    # enable pickling of decorator
from datetime import timedelta
# from collections import Iterable

import ruffus
import ruffus.cmdline as cmdline
from ruffus.proxy_logger import make_shared_logger_and_proxy
# from ruffus import mkdir
from ruffus import Pipeline
from ruffus import formatter
# from ruffus import add_inputs
from ruffus import output_from
from ruffus.ruffus_exceptions import error_ambiguous_task
import env
import utils
import func


def ensure_list(value, tuple_as_list=False):
    if tuple_as_list:
        listclass = (list, tuple)
        elemclass = (str, dict)
    else:
        listclass = list
        elemclass = (str, tuple, dict)
    if isinstance(value, elemclass) or callable(value):
        value = [value, ]
    elif value is None:
        value = []
    elif isinstance(value, listclass):
        if isinstance(value, tuple):
            value = list(value)
    else:
        raise RuntimeError("not able to ensure list type for {0}"
                           .format(value))
    return value


def ensure_args_as_list(*iargs, **ikwargs):
    """wrap string argument as list"""
    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            newargs = list(args)
            for i in iargs:
                newargs[i] = ensure_list(args[i], **ikwargs)
            return func(*newargs, **kwargs)
        return wrapped_func
    return wrapper


def unwrap_if_len_one(arg):
    return arg if len(arg) > 1 else arg[0]


class ApusConfig(object):
    """interfacing the input config module"""

    dirs = [
        ('jobdir', None), ('confdir', '{jobdir:s}'), ('diagdir', '{jobdir:s}'),
        ('task_io_default_dir', '{jobdir:s}')
        ]
    specs = [('jobkey', None), ('inputs', []), ('tlist', [])]
    runtime = [
        ('env_overrides', {}), ('slice_test', None),
        ('logger', None), ('logger_mutex', None),
        ('log_file', '{jobkey:s}.log'), ('history_file', '{jobkey:s}.ruffus')
        ]

    def __init__(self, config):
        missingkeyerror = "missing key in config module: {0}"
        for key, defval in self.dirs + self.specs + self.runtime:
            try:
                val = getattr(config, key)
            except AttributeError:
                if defval is None and key not in dict(self.runtime).keys():
                    raise RuntimeError(missingkeyerror.format(key))
                elif isinstance(defval, str):
                    val = defval.format(**self.__dict__)
                else:
                    val = defval
            setattr(self, key, val)

    def get_dirs(self):
        return list(set(getattr(self, k) for k in dict(self.dirs).keys()))

    def get_task_names(self):
        return [t['name'] for t in self.tlist]


def configure(config, args):
    """parse command line arguments, split the ruffus arguments from
    config moduel items, provide additional command line shortcuts

    :config: configuration module
    :args: arguments supplied in commandline
    :returns: ApusConfig object, argparse option object

    """
    apusconf = ApusConfig(config)
    parser = cmdline.get_argparse(description="""
+- Astronomy Pipeline Using ruffuS (script by Zhiyuan Ma) -+
""", version=ruffus.__version__)
    # additional options
    parser.add_argument('action', nargs='?', default='run',
                        choices=['init', 'run'],
                        help='the action to take: initialize the pipeline'
                             ' directory, or run the pipeline')
    parser.add_argument(
            '-r', '--redo-all', action='store_true',
            help='force redo all tasks')
    parser.add_argument(
            '-d', '--dry-run', action='store_true',
            help='perform low level dry run for debugging purpose')

    def input_slice(string):
        if '...' in string:
            return map(int, string.split('...'))
        elif string == 'none':
            return None
        else:  # single integer is slicing from 0
            return [0, int(string)]
    parser.add_argument(
            '-s', '--slice-test', nargs='+', default='none', type=input_slice,
            help='run a slice of inputs for testing purpose')
    parser.set_defaults(
            verbose=['0', ],
            log_file=apusconf.log_file,
            history_file=apusconf.history_file
            )
    option = parser.parse_args(args)
    # handle logger
    logger, logger_mutex = make_shared_logger_and_proxy(
            logger_factory, apusconf.jobkey, [option.log_file, option.verbose])
    apusconf.logger = logger
    apusconf.logger_mutex = logger_mutex
    # handle slice test: override expand glob and slice on creating task
    apusconf.slice_test = option.slice_test
    # handle dryrun
    apusconf.dry_run = option.dry_run

    return apusconf, option


def get_task_func(func):
    errmsg = 'not a meaningful task func value: {0}'.format(func)
    if isinstance(func, str):
        if func.lower() in ['sextractor', 'sex', 'scamp', 'swarp']:
            return astromatic_task
        else:
            command = func.split()
            if '{in}' in command or '{out}' in command:
                return subprocess_task
            else:
                raise RuntimeError(errmsg)
    elif callable(func):
        return callable_task
    else:
        raise RuntimeError(errmsg)


def aggregate_task_inputs(input_, tuple_num_elem=2):
    formatter_inputs = []
    simple_inputs = []
    generator_inputs = []
    for in_ in input_:
        if callable(in_):
            generator_inputs.append(in_)
        elif isinstance(in_, tuple) and len(in_) == tuple_num_elem:
            if callable(in_[0]):
                raise RuntimeError('generator input should not have formatter'
                                   ' {0}'.format(in_))
            formatter_inputs.append(in_)
        elif isinstance(in_, list):
            simple_inputs.extend(in_)
        elif isinstance(in_, (str, dict)):
            simple_inputs.append(in_)
        else:
            raise RuntimeError('invalid input {0}'.format(in_))
    if len(generator_inputs) > 1:
        raise RuntimeError('found multiple generator inputs {0}'.format(
            generator_inputs))
    return formatter_inputs, simple_inputs, generator_inputs


def sliced_glob(pattern, slice_ranges):
    ret = []
    files = glob.glob(pattern)
    for l, r in slice_ranges:
        ret.extend(files[l:r])
    return list(set(ret))


def create_ruffus_task(pipe, config, task, **kwargs):
    """create Ruffus task from the task dictionary and add to pipe

    keys: name, func, pipe, [in_, out]
    optional task keys: add_inputs, replace_inputs,
                        allow_slice, in_keys, out_keys,
    optional context keys: allow_finish_silently, dry_run
    """

    # validate the task dict first
    missingkeyerror = "missing key in task dict: {0}"
    for key in ['name', 'func', 'pipe', ['in_', 'out']]:
        if all(k not in task.keys() for k in ensure_list(key)):
            raise RuntimeError(missingkeyerror.format(key))
    config.__dict__.update(**kwargs)

    # context is parameters that are passed to the task function
    context_exclude_task_keys = [
            'pipe', 'in_', 'out', 'add_inputs', 'allow_slice']
    context_key_defaults = {
            'allow_finish_silently': False,
            'dry_run': config.dry_run,
            }
    context = {k: v for k, v in task.items() + context_key_defaults.items()
               if k not in context_exclude_task_keys}
    for key, defval in context_key_defaults.items():
        context[key] = task.get(key, defval)

    # process task dict
    task_name_list = config.get_task_names()

    pipe_func = getattr(pipe, task['pipe'])
    task_name = task['name']
    task_func = get_task_func(task['func'])

    task_args = []
    task_kwargs = {'name': task_name, 'task_func': task_func}

    # handle input
    formatter_inputs, simple_inputs, generator_inputs = aggregate_task_inputs(
            ensure_list(task['in_'], tuple_as_list=False))
    if len(generator_inputs) > 0:  # generator_inputs goes to unnamed argument
            task_args.extend(generator_inputs)
    # simple_inputs get common general formatter
    if len(simple_inputs) > 0:
        formatter_inputs.append((simple_inputs, r'.+'))
    # handle formatter_inputs
    task_inputs = []
    task_formatters = []
    for in_, reg in formatter_inputs:
        in_ = [i['name'] if isinstance(i, dict) else i
               for i in ensure_list(in_)]
        temp_in = []
        temp_reg = reg
        for i in in_:
            if i in task_name_list:
                temp_in.append(output_from(i))
                continue
            elif not os.path.isabs(i):  # prepend default io dir
                i = os.path.join(config.task_io_default_dir, i)
                temp_reg = r'(?:[^/]*/)*' + reg
            else:
                pass
            if re.search(r'[?*,\[\]{}]', i) is not None:
                # slice on flagged task with glob input
                if config.slice_test is not None \
                        and task.get('allow_slice', False):
                    config.logger.info(
                            'sliced glob input: {0} @ {1}'.format(
                                i, ', '.join(map(str, config.slice_test))))
                    sliced_files = sliced_glob(i, config.slice_test)
                    if len(sliced_files) == 0:
                        raise RuntimeError('no input left after slicing')
                    for f in sliced_files:
                        config.logger.info(' {0}'.format(f))
                    temp_in.extend(sliced_files)
                else:
                    config.logger.info('glob input: {0}'.format(i))
                    temp_in.append(i)
            else:
                config.logger.info('file input: {0}'.format(i))
                temp_in.append(i)
        task_inputs.append(temp_in)  # list of list
        task_formatters.append(temp_reg)  # list of regex
    if len(task_inputs) > 0:
        task_inputs = reduce(lambda a, b: a + b, task_inputs)  # flatten
        if len(task_inputs) > 0:
            task_kwargs['input'] = unwrap_if_len_one(task_inputs)
        if task['pipe'] != 'merge':  # require formatter for non-merge pipe
            task_kwargs['filter'] = formatter(*task_formatters)
    # handle additional inputs and replace_inputs
    for inkey in ['add_inputs', 'replace_inputs']:
        task_inkey = []
        for in_ in ensure_list(task.get(inkey, None)):
            if in_ in task_name_list:
                try:  # have to replace task name with task
                    in_, = pipe.lookup_task_from_name(in_, "__main__")
                except (ValueError, error_ambiguous_task):
                    pass
            if isinstance(in_, str) and not os.path.isabs(in_):
                in_ = os.path.join(config.task_io_default_dir, in_)
            task_inkey.append(in_)
        if len(task_inkey) > 0:
            task_kwargs[inkey] = tuple(task_inkey)  # ruffus req.
    # handle outputs
    task_output = []
    for out in ensure_list(task.get('out', None)):
        if not os.path.isabs(out):
            out = os.path.join(config.task_io_default_dir, out)
        task_output.append(out)
    if len(task_output) > 0:
        task_kwargs['output'] = unwrap_if_len_one(task_output)
    else:  # flag file for checkpointing
        task_kwargs['output'] = os.path.join(
            config.task_io_default_dir,
            task_name.replace(' ', '_') + '.success')
    # handle follows
    task_follows = [t['name'] if isinstance(t, dict) else t
                    for t in ensure_list(task.get('follows', []))]
    if 'follows' in context.keys():
        # for cleaner debug info
        context['follows'] = unwrap_if_len_one(task_follows)
    # handle context as extra
    task_extras = []
    for extra in ensure_list(task.get('extras', None)):
        if isinstance(extra, str) and not os.path.isabs(extra):
            task_extras.append(os.path.join(config.task_io_default_dir, extra))
        else:
            task_extras.append(extra)
    task_context = {
            'task': context,
            'logger': config.logger,
            'logger_mutex': config.logger_mutex,
            }
    task_extras.append(task_context)
    task_kwargs['extras'] = task_extras
    # create ruffus task
    ruffus_task = pipe_func(*task_args, **task_kwargs)
    # additional follows for astromatic task if no input config is supplied
    if task_func.__name__ == 'astromatic_task' and \
            'conf' not in ensure_list(task.get('in_keys', None)):
        pre_task = {
                'name': 'config {0}'.format(task_name),
                'func': func.dump_config_files,
                'pipe': 'originate',
                'out': os.path.join(config.confdir, 'conf.{0}'.format(
                        task_name.replace(' ', '_'))),
                'params': task['params']
                }
        task_follows.append(create_ruffus_task(
            pipe, config, pre_task, task_io_default_dir=''))
    if len(task_follows) > 0:
        ruffus_task.follows(*task_follows)
    # handle job_limit
    jobs_limit = task.get('jobs_limit', None)
    if jobs_limit is not None:
        ruffus_task.jobs_limit(jobs_limit)
    # add finish signal
    ruffus_task.posttask(task_finish_signal(task_name, config))
    return ruffus_task


def build_init_pipeline(config, option):
    """pipeline to prepare directories and symbolic links for inputs

    :config: configuration module
    :option: commandline argument
    :returns: pipeline object
    """
    pipe = Pipeline(name=config.jobkey + '.init')
    t00 = {'name': 'make dirs'}
    pipe.mkdir(config.get_dirs(), name=t00['name'])

    # rectify the inputs
    formatter_inputs, simple_inputs, _ = aggregate_task_inputs(
            ensure_list(config.inputs), tuple_num_elem=3)
    if len(formatter_inputs) + len(simple_inputs) == 0:
        raise RuntimeError('no input specified')
    # create tasks
    tlist = []
    for i, (in_, reg, out) in enumerate(formatter_inputs):
        name = 'link formatter inputs'
        if i > 0:
            name += ' {0}'.format(i + 1)
        tlist.append({
            'name': name,
            'func': func.create_symbolic_link,
            'pipe': 'transform',
            'in_': (in_, reg),
            'out': out,
            'allow_finish_silently': True,
            'allow_slice': True,
            'follows': t00,
            })
    if len(simple_inputs) > 0:
        tlist.append({
            'name': 'link simple inputs',
            'func': func.create_symbolic_link,
            'pipe': 'transform',
            'in_': simple_inputs,
            'out': os.path.join(config.jobdir, '{basename[0]}{ext[0]}'),
            'allow_finish_silently': True,
            'follows': t00,
            })
    # t03 = {
    #         'name': 'link per input extras',
    #         'func': func.create_symbolic_link,
    #         'type_': 'files',
    #         'follows': t00['name'],
    #         }
    # def gen_per_input_extra():
    #     for f in ensure_list(config.per_input_extra):
    #         for in_ in config.inputs:
    #             # @files doesn't propagate extra, so we pass them explicitly
    #             yield list(f(in_)) + [t03, config.logger,
    #                                   config.logger_mutex]
    # t03['in_'] = gen_per_input_extra
    for t in tlist:
        create_ruffus_task(pipe, config, t, task_io_default_dir='')

    return pipe


def build_pipeline(config, option):
    """assemble the job pipeline

    :config: configuration module
    :option: commandline argument
    :returns: pipeline object
    """
    pipe = Pipeline(name=config.jobkey)
    for task in config.tlist:
        create_ruffus_task(pipe, config, task)
    return pipe


def bootstrap():
    """entry point; parse command line argument, create pipeline object,
    and run it
    """
    print("+- APUS powered by Ruffus ver {0} -+".format(ruffus.__version__))
    config, option = configure(sys.modules['__main__'], sys.argv[1:])
    # set up env with overrides
    env.am.set_overrides(config.env_overrides)
    # check existence of the jobdir
    if option.action == 'run':
        if not os.path.exists(config.jobdir):
            raise RuntimeError('job directory does not exist,'
                               ' run init to create one')
        else:
            build_pipeline(config, option)
    elif option.action == 'init':
        option.history_file = option.history_file + '.init'
        build_init_pipeline(config, option)
    # handle redo-all
    if option.redo_all:
        task_list = ruffus.pipeline_get_task_names()
        option.forced_tasks.extend(task_list)
    if len(option.forced_tasks) > 0:
        for t in option.forced_tasks:
            config.logger.info("forced redo: {0}".format(utils.alert(t)))
    # mark the begin time
    config.timestamp = time.time()
    cmdline.run(option, checksum_level=1)


def logger_factory(logger_name, args):
    """provide logging modules, adapted from ruffus, with some small tweaks

    """
    log_file_name, verbose = args
    new_logger = logging.getLogger(logger_name)

    class debug_filter(logging.Filter):
        """ignore INFO messages"""
        def filter(self, record):
            return logging.INFO != record.levelno

    class NullHandler(logging.Handler):
        """for when there is no logging"""
        def emit(self, record):
            pass

    new_logger.setLevel(logging.DEBUG)
    has_handler = False

    # log to file if that is specified
    if log_file_name:
        handler = logging.FileHandler(log_file_name, delay=False)

        class stipped_down_formatter(logging.Formatter):
            def format(self, record):
                prefix = ""
                if not hasattr(self, "first_used"):
                    self.first_used = True
                    prefix = "\n" + self.formatTime(record, "%Y-%m-%d")
                    prefix += " %(name)s\n" % record.__dict__
                self._fmt = " %(asctime)s - %(levelname)-7s - %(message)s"
                old_msg = record.msg
                record.msg = utils.de_alert(record.msg)
                out = prefix + logging.Formatter.format(self, record)
                record.msg = old_msg
                return out
        handler.setFormatter(stipped_down_formatter(
            "%(asctime)s - %(name)s - %(levelname)6s - %(message)s", "%H:%M:%S"
            ))
        handler.setLevel(logging.DEBUG)
        new_logger.addHandler(handler)
        has_handler = True

    # log to stderr if verbose
    if verbose:
        stderrhandler = logging.StreamHandler(sys.stderr)
        stderrhandler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        stderrhandler.setLevel(logging.DEBUG)
        # if log_file_name:
        #     stderrhandler.addFilter(debug_filter())
        new_logger.addHandler(stderrhandler)
        has_handler = True

    # no logging
    if not has_handler:
        new_logger.addHandler(NullHandler())

    return new_logger


def task_finish_signal(task_name, config):
    """print task execution time upon finishing"""
    def ret_task_finish_signal():
        with config.logger_mutex:
            elapsed_time = timedelta(
                    seconds=time.time() - config.timestamp)
            config.logger.debug('task {0:^45s} finished @ {1:s}'
                                .format(utils.alert(task_name),
                                        utils.alert(elapsed_time))
                                )
    return ret_task_finish_signal


@ensure_args_as_list(0)
def touch_file(out_files):
    for out in out_files:
        with open(out, 'a'):
            os.utime(out, None)


@ensure_args_as_list(0)
def get_flagfile(out_files):
    if len(out_files) == 1 and '.success' in out_files[0]:
        flag = out_files[0]
        out_files = []
    else:
        flag = None
    return out_files, flag


def documented_subprocess_call(command, flag=None):
    def call(*args, **kwargs):
        # handle scamp refcatalog suffix
        if '-ASTREFCAT_NAME' in command:
            ikey = command.index('-ASTREFCAT_NAME') + 1
            refcatkey = command[ikey]
            refcatfiles = glob.glob(
                    "{0}_?{1}".format(*os.path.splitext(refcatkey)))
            if len(refcatfiles) == 0:
                refcatfiles = glob.glob(refcatkey)
            if len(refcatfiles) > 0:
                command[ikey] = ','.join(refcatfiles)
        log = func.get_log_func(**kwargs)
        # output = subprocess.check_output(command)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=1)
        has_output = False
        for ln in iter(proc.stdout.readline, b''):
            if ln:
                has_output = True
            log('debug', ln.strip('\n'))
        if proc.poll() is not None and proc.returncode != 0:
            # an error happened!
            err_msg = "%s\nsubprocess failed with code: %s" % (
                    proc.stderr.read(), proc.returncode)
            raise RuntimeError(err_msg)
        if flag is not None:
            with open(flag, 'w'):
                pass
        return has_output
        # return output
    # print(' '.join(command))
    call.__doc__ = 'subprocess: ' + ' '.join(command)
    return call


@ensure_args_as_list(0, 1)
def astromatic_task(in_files, out_files, *extras):
    """create astromatic command to run by subprocess"""
    in_files.extend(extras[:-1])
    context = extras[-1]
    task = dict(context['task'], func=get_astromatic_callable(
            in_files, out_files, context))
    return callable_task(in_files, out_files, dict(context, task=task))


def get_astromatic_inputs(inputs, in_keys):
    """identify the inputs by mapping to the in_keys"""

    # collate, no add_input: [(in1, in2, ...), ex1, ex2]
    # key: [key1, ekey1, ekey2]

    # collate, add_input: [((in1, add1, add2), (in2, add1, add2)), ex1, ex2]
    # key: [(key1, key2 key3), ekey1, ekey2]

    # transform, no add_input: [ in, ex1, ex2]
    # key: [key1, ekey1, ekey2]

    # transform, add_input: [(in, add1), ex1, ex2]
    # key: [(key1, key2 key3), ekey1, ekey2]

    # ret:
    #   [(key1, [in1, in2, .. ]), (key2, [add1_1, add1_2 ..])
    ret_keys = []
    ret_vals = []
    # first level zip for exkeys
    for key, val in zip(in_keys, inputs):
        if isinstance(key, tuple):  # deal with tuple keys:
            if all(isinstance(v, tuple) for v in val):
                val = zip(*val)  # tuple of tuple, need to be zipped
            if len(key) == len(val):
                ret_keys.extend(key)
                ret_vals.extend(val)
            else:
                raise RuntimeError(
                    "mismatched number of"
                    " keys ({0}) and inputs {1}".format(len(key), len(val)))
        else:
            ret_keys.append(key)
            ret_vals.append(val)
    # aggregate duplicated keys
    ag_keys = list(set(ret_keys))
    ag_vals = [[] for _ in range(len(ag_keys))]
    for i, key in enumerate(ret_keys):
        ag_vals[ag_keys.index(key)].append(ret_vals[i])
    for i, val in enumerate(ag_vals):
        # validate list-type keys
        if any([isinstance(v, tuple) for v in val]):
            if len(val) > 1:
                raise RuntimeError("list-type key {0} should only appear once"
                                   " in in_keys".format(ag_keys[i]))
            else:
                ag_vals[i] = val[0]
    return zip(ag_keys, ag_vals)


def get_astromatic_callable(in_files, out_files, context):
    """return the command for executing astromatic task"""
    task = context['task']
    # ensure there is a flag file at least for check pointing
    out_files, flag = get_flagfile(out_files)
    # get program
    program = [i for i in ['sex', 'scamp', 'swarp']
               if task['func'].lower().startswith(i)][0]
    # split up inputs types
    rectified_inputs = get_astromatic_inputs(in_files,
                                             task.get('in_keys', ['in', ]))
    command = [env.am.get('{0}bin'.format(program)), ]
    params = {}
    for key, val in rectified_inputs:
        # here val is a list of values
        if key == 'in':
            command.extend(val)
        elif key == 'conf':
            if len(val) < 1:
                raise RuntimeError('no configuration file specified for {0}'
                                   .format(task['func']))
            else:
                command.extend(['-c', val[0]])
                params.update(utils.parse_astromatic_conf(*val[1:]))
        else:  # values should be concat by comma
            params[key] = ','.join(val)
    params.update(task.get('params', {}))
    for key, val in params.items():
        command.extend(['-{0}'.format(key), '"{0}"'.format(val)])
    # handle outkeys
    default_outkeys = {
            'sex': ['CATALOG_NAME', ],
            'scamp': [],
            'swarp': ['IMAGEOUT_NAME', 'WEIGHTOUT_NAME']
            }
    out_keys = task.get('out_keys', default_outkeys[program])
    for i, key in enumerate(out_keys):
        command.extend(['-{0}'.format(key), out_files[i]])
    # handle out_files
    # if program == 'sex':
    #     outkey = ensure_list(task.get('outkey', 'CATALOG_NAME'))
    # elif program == 'scamp':
    #     outkey = ensure_list(task.get('outkey', {}))
    #     # handle output catalog name suffix
    #     if len(outkey) == 1 and outkey[0] == 'MERGEDOUTCAT_NAME':
    #         flag = out_files[0]
    # elif program == 'swarp':
    #     if len(out_files) == 1:
    #         out_files.append(out_files[0].replace('.fits', '.weight.fits'))
    #     outkey = ensure_list(task.get('outkey',
    #                          ['IMAGEOUT_NAME', 'WEIGHTOUT_NAME']))
    # for i, key in enumerate(outkey):
    #     command.extend(['-' + key, out_files[i]])
    # handle multiple in_files
    # scheme: [dectect] image [weight1, weight2, ...]
    # if program == 'sex':
    #     if params.get('WEIGHT_TYPE', 'NONE') != 'NONE':
    #         n_weight = len(params['WEIGHT_TYPE'].split(','))
    #         weight = ensure_list(params.pop('WEIGHT_IMAGE', []))
    #         while len(weight) < n_weight:
    #             weight.insert(0, in_files.pop())
    #         command.extend(['-WEIGHT_IMAGE', ','.join(weight)])
    # elif program == 'scamp':
    #     if params.get('ASTREF_CATALOG', None) is not None and \
    #             params['ASTREF_CATALOG'] == 'FILE':
    #         # the last input is the catalog key
    #         refcatkey = in_files.pop()
    #         command.extend(['-ASTREFCAT_NAME', refcatkey])
    # elif program == 'swarp':
    #     # input looks like [[], [], ...]
    #     # [img, wht, head]
    #     _in_files = zip(*in_files)
    #     if params.get('WEIGHT_TYPE', 'NONE') != 'NONE':
    #         img, wht = _in_files[:2]
    #         hdr = _in_files[2] if len(_in_files) > 2 else None
    #     else:
    #         img, wht = _in_files[0], None
    #         hdr = _in_files[1] if len(_in_files) > 1 else None
    #     in_files = img
    #     if wht is not None:
    #         command.extend(['-WEIGHT_IMAGE', ','.join(wht)])
    #     if hdr is not None:  # create sym link
    #         for (i, h) in zip(img, hdr):
    #             lnh = '{0}{1}'.format(os.path.splitext(i)[0],
    #                                   params.get('HEADER_SUFFIX', '.head'))
    #             subprocess.call(['ln', '-sf', os.path.abspath(h), lnh])
    # for k, v in params.items():
    #     command.extend(['-{0}'.format(k), v])
    # command.extend(in_files)
    return documented_subprocess_call(command, flag=flag)


@ensure_args_as_list(0, 1)
def subprocess_task(in_files, out_files, *extras):
    """run command as subprocess by replacing func key in task context
    with callable
    extras: extra inputs, task context
    """
    in_files.extend(extras[:-1])
    context = extras[-1]
    task = dict(context['task'], func=get_subprocess_callable(
            in_files, out_files, context))
    return callable_task(in_files, out_files, dict(context, task=task))


def get_subprocess_callable(in_files, out_files, context):
    if any(isinstance(i, tuple) for i in in_files):
        if len(in_files) > 1:
            raise RuntimeError("subprocess task should not have nested inputs")
        else:
            in_files = in_files[0]
    out_files, flag = get_flagfile(out_files)
    command = context['task']['func'].split()
    for key, value in zip(['{in}', '{out}'], [in_files, out_files]):
        if key in command:
            i = command.index(key)
            command[i:i + 1] = value
    # command = task['func'] + in_files + out_files
    return documented_subprocess_call(command, flag=flag)


@ensure_args_as_list(0, 1)
def callable_task(in_files, out_files, *extras):
    in_files.extend(extras[:-1])
    context = extras[-1]
    task, logger, logger_mutex = [
            context[i] for i in ['task', 'logger', 'logger_mutex']]
    func = task['func']
    if func.__doc__.startswith('subprocess: '):
        mesg = func.__doc__[len('subprocess: '):]
    else:
        mesg = "{0}({1}, {2})".format(func.__name__, in_files, out_files)
    if task.get('dry_run', False):
        output = '~dry~run~3~sec: {0}'.format(mesg)
        time.sleep(1)
        touch_file(out_files)
    else:
        with logger_mutex:
            logger.debug(mesg)
        kwargs = {'task': task, 'logger': logger, 'logger_mutex': logger_mutex}
        output = func(unwrap_if_len_one(in_files),
                      unwrap_if_len_one(out_files),
                      **kwargs)
    if not output and task.get('allow_finish_silently', False):
        pass
    else:
        output = "finished silently" if not output else "finished"
        with logger_mutex:
            logger.debug(output)


# OBSOLETE
def get_amprog(string):
    """naming convention of the astromatic task key: sex_suffix"""
    prog = string.split('_', 1)[0]
    return prog


def get_amsuffix(string):
    """naming convention of the astromatic task key: sex_suffix"""
    if '_' in string:
        return string.split('_', 1)[-1]
    else:
        return ""


def get_amconf(key):
    """naming convention of astromatic configuration file"""
    return 'conf.{0}'.format(key)


def am_config_default(prog):
    if prog == 'sex':
        return {'STARNNW_NAME': 'default.nnw',
                'WRITE_XML': 'N',
                'BACKPHOTO_TYPE': 'LOCAL',
                'PIXEL_SCALE': '0',
                'HEADER_SUFFIX': '.none'}
    elif prog == 'scamp':
        return {'CHECKPLOT_RES': '1024',
                'SAVE_REFCATALOG': 'Y',
                'WRITE_XML': 'Y',
                }
    elif prog == 'swarp':
        return {'INTERPOLATE': 'N',
                'FSCALASTRO_TYPE': 'VARIABLE',
                'DELETE_TMPFILES': 'N',
                'NOPENFILES_MAX': '1000000',
                }


def dump_configuration_files(flag_file, config):
    """Dump AstrOmatic configuration files

    :config: config object
    """

    logger, logger_mutex = config.logger, config.logger_mutex
    confdir = config.confdir
    am_diagdir = config.am_diagdir
    am_sharedir = config.am_sharedir
    am_resampdir = config.am_resampdir
    am_params = config.am_params
    # aggregate keys by program: sex -> sex_1, sex_2 ...
    # merge param to per_task_params
    global_params = {}
    per_task_params = {}
    has_per_task_params = dict(sex=0, scamp=0, swarp=0)
    for key in am_params.keys():
        program = get_amprog(key)
        if '_' in key:
            per_task_params[key] = am_params[key]
            _d = am_config_default(program)
            _d.update(**am_params.get(program, {}))
            _d.update(**per_task_params[key])
            per_task_params[key] = _d
            has_per_task_params[program] += 1
        else:
            global_params[key] = dict(am_config_default(program),
                                      **am_params[key])
    for k, v in has_per_task_params.items():
        if v == 0 and k in global_params.keys():
            per_task_params[k] = global_params[k]

    # create configuration files and set up the path
    for key, param in per_task_params.items():
        program = get_amprog(key)  # e.g. sex for sex_astro
        am_bin = config.get('{0}bin'.format(program), None)
        if am_bin is None:
            raise ValueError('path to executable for {0} is not defined'
                             .format(program))
        # default am_share directory
        if config.am_sharedir is None:
            am_sharedir = env.am.get('{0}share'.format(program))
        # generate filenames
        conffile = get_amconf(key)
        if program == 'sex':  # both conf and param
            paramfile = conffile + '.param'
            param['PARAMETERS_NAME'] = os.path.join(confdir, paramfile)
        if program == 'scamp':
            amsuffix = get_amsuffix(key)
            param['CHECKPLOT_NAME'] =\
                utils.get_scamp_checkplot_name(os.path.abspath(am_diagdir),
                                               prefix=amsuffix)
        elif program == 'swarp' and \
                param.get('RESAMPLE_DIR', None) is None:
            param['RESAMPLE_DIR'] = am_resampdir
        # use absolute dir whenever possible
        for k, v in param.items():
            if os.path.isfile(v):
                param[k] = os.path.abspath(v)
            elif os.path.isfile(os.path.join(am_sharedir, v)):
                param[k] = os.path.abspath(os.path.join(am_sharedir, v))
            elif os.path.isfile(os.path.join(confdir, v)):
                param[k] = os.path.abspath(os.path.join(confdir, v))
            else:
                param[k] = v
        # create config and param
        conffile = os.path.join(confdir, conffile)
        fo = StringIO.StringIO(subprocess.check_output([am_bin, '-dd']))
        utils.dump_astromatic_conf(fo, conffile, clobber=True, **param)
        with logger_mutex:
            logger.info('dump conf: {0}'.format(conffile))
            for k, v in param.items():
                logger.info('{0:>20s}: {1:s}'.format(k, v))
        fo.close()
        if program == 'sex':
            paramfile = os.path.join(confdir, paramfile)
            fo = StringIO.StringIO(subprocess.check_output([am_bin, '-dp']))
            utils.dump_sex_param(fo, paramfile, clobber=True)
            with logger_mutex:
                logger.info('dump param: {0}'.format(paramfile))
            fo.close()
    # write flag file
    with open(flag_file, 'w'):
        pass

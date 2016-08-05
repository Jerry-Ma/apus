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
import cPickle as pickle
from StringIO import StringIO
import subprocess
from copy import copy
from functools import wraps    # enable pickling of decorator
from datetime import timedelta
# from collections import Iterable
# from tempfile import NamedTemporaryFile

import ruffus
import ruffus.cmdline as cmdline
from ruffus.proxy_logger import make_shared_logger_and_proxy
# from ruffus import pipeline_printout_graph
# from ruffus import mkdir
from ruffus import Pipeline
from ruffus import formatter
# from ruffus import add_inputs
from ruffus import output_from
from ruffus.ruffus_exceptions import error_ambiguous_task
import env
import utils
import common


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


def unwrap_if_len_one(arg):
    return arg if len(arg) > 1 else arg[0]


class ApusConfig(object):
    """interfacing the input config module"""

    dirs = [
        ('jobdir', None), ('confdir', '{jobdir:s}'), ('diagdir', '{jobdir:s}'),
        ('task_io_default_dir', '{jobdir:s}'),
        ('logdir', ''),
        ]
    specs = [('jobkey', None), ('inputs', []), ('tlist', [])]
    runtime = [
        ('env_overrides', {}), ('slice_test', None),
        ('logger', None), ('logger_mutex', None),
        ('log_file', '{jobkey:s}.log'), ('history_file', '{jobkey:s}.ruffus')
        ]

    def __init__(self, config):
        # mark the begin time
        self.timestamp = time.time()
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
        return [d for d in set(getattr(self, k)
                for k in dict(self.dirs).keys()) if d != '']

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
            log_file=os.path.join(apusconf.logdir, apusconf.log_file),
            history_file=os.path.join(apusconf.logdir, apusconf.history_file)
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


def normalize_am_params(params):
    for k, v in params.items():
        if isinstance(v, list):
            v = ', '.join(map(str, v))
        else:
            v = str(v)
        params[k] = v
    return params


def normalize_task_name(name):
    return name.replace(' ', '_')


def create_ruffus_task(pipe, config, task, **kwargs):
    """create Ruffus task from the task dictionary and add to pipe

    keys: name, func, pipe, [in_, out]
    optional task keys: add_inputs, replace_inputs,
                        allow_slice, in_keys, out_keys,
                        kwargs
    optional context keys: verbose, dry_run
    """

    # validate the task dict first
    missingkeyerror = "missing key in task dict: {0}"
    for key in ['name', 'func', 'pipe', ['in_', 'out']]:
        if all(k not in task.keys() for k in ensure_list(key)):
            raise RuntimeError(missingkeyerror.format(key))
    config = copy(config)
    config.__dict__.update(**kwargs)

    # process task dict
    task_name_list = config.get_task_names()

    pipe_func = getattr(pipe, task['pipe'])
    task_name = task['name']
    task_func = get_task_func(task['func'])

    task_args = []
    task_kwargs = {'name': task_name, 'task_func': task_func}
    # handle follows
    task_follows = [t['name'] if isinstance(t, dict) else t
                    for t in ensure_list(task.get('follows', []))]
    # additional logic for astromatic tasks
    if task_func.__name__ == 'astromatic_task':
        task['params'] = normalize_am_params(task.get('params', {}))
        # generate configuration file if not supplied
        if 'conf' not in ensure_list(task.get('in_keys', None)):
            pre_task = {
                'name': 'auto config {0}'.format(task_name),
                'func': dump_config_files,
                'pipe': 'originate',
                'out': os.path.join(config.confdir, 'conf.{0}'.format(
                        normalize_task_name(task_name))),
                'extras': os.path.join(
                    config.confdir,
                    'conf.{0}.checker'.format(normalize_task_name(task_name))),
                'params': task.pop('params'),  # remove params from this task
                'outparams': task.get('outparams', []),
                'verbose': False,
                'check_if_uptodate': check_config_uptodate,
                'diagdir': config.diagdir,
                'prog': get_am_prog(task['func']),
                'jobs_limit': 1
                }
            create_ruffus_task(
                pipe, config, pre_task, task_io_default_dir='')
            task_follows.append(pre_task['name'])
            # connect pre_task to this
            conf_inputs = task.get('extras', [])
            conf_inputs.append(os.path.abspath(pre_task['out']))
            task_inkeys = task.get('in_keys', ['in', ])
            task_inkeys.append('conf')
            task['extras'] = conf_inputs
            task['in_keys'] = task_inkeys

    # handle input
    def handle_input(in_, in_key, formatter_key):
        formatter_inputs, simple_inputs, generator_inputs = \
            aggregate_task_inputs(
                ensure_list(task.get(in_, []), tuple_as_list=False))
        if len(generator_inputs) > 0:  # generator_inputs goes to unnamed arg
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
                task_kwargs[in_key] = unwrap_if_len_one(task_inputs)
            if task['pipe'] != 'merge':  # require formatter for non-merge pipe
                task_kwargs[formatter_key] = formatter(*task_formatters)
    handle_input('in_', 'input', 'filter')
    if 'in2' in task.keys():
        handle_input('in2', 'input2', 'filter2')

    def resolve_task_name(s):
        if isinstance(s, dict):
            s = s['name']
        if s in task_name_list:
            try:  # have to replace task name with task
                s, = pipe.lookup_task_from_name(s, "__main__")
            except (ValueError, error_ambiguous_task):
                pass
        return s
    # handle additional inputs and replace_inputs
    for inkey in ['add_inputs', 'replace_inputs']:
        task_inkey = []
        for in_ in ensure_list(task.get(inkey, None)):
            in_ = resolve_task_name(in_)
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
            normalize_task_name(task_name) + '.success')
    # handle context as extra
    task_extras = []
    for extra in ensure_list(task.get('extras', None)):
        if isinstance(extra, str) and not os.path.isabs(extra):
            task_extras.append(os.path.join(config.task_io_default_dir, extra))
        else:
            task_extras.append(extra)
    # context is parameters that are passed to the task function
    context_exclude_task_keys = [
            'in_', 'out', 'add_inputs', 'allow_slice']
    context_key_defaults = {
            'verbose': True,
            'dry_run': False,
            'kwargs': {},
            }
    context = {k: v for k, v in task.items() + context_key_defaults.items()
               if k not in context_exclude_task_keys}
    for key, defval in context_key_defaults.items():
        context[key] = task.get(key, defval)
    if config.dry_run:
        context['dry_run'] = True
    if 'follows' in context.keys():
        # for cleaner debug info
        context['follows'] = unwrap_if_len_one(task_follows)
    task_context = {
            'task': context,
            'logger': config.logger,
            'logger_mutex': config.logger_mutex,
            }
    task_extras.append(task_context)
    task_kwargs['extras'] = task_extras
    # create ruffus task
    ruffus_task = pipe_func(*task_args, **task_kwargs)
    if len(task_follows) > 0:
        ruffus_task.follows(*task_follows)
    # handle job_limit
    jobs_limit = task.get('jobs_limit', None)
    if jobs_limit is not None:
        ruffus_task.jobs_limit(jobs_limit)
    # add finish signal
    ruffus_task.posttask(task_finish_signal(task_name, config))
    # handle forced run
    if task.get('check_if_uptodate', None) is not None:
        ruffus_task.check_if_uptodate(task['check_if_uptodate'])
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
            'func': common.create_symbolic_link,
            'pipe': 'transform',
            'in_': (in_, reg),
            'out': out,
            'verbose': False,
            'allow_slice': True,
            'follows': t00,
            'kwargs': {'relpath': False},
            })
    if len(simple_inputs) > 0:
        tlist.append({
            'name': 'link simple inputs',
            'func': common.create_symbolic_link,
            'pipe': 'transform',
            'in_': simple_inputs,
            'out': os.path.join(config.jobdir, '{basename[0]}{ext[0]}'),
            'verbose': False,
            'follows': t00,
            'kwargs': {'relpath': False},
            })
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


def bootstrap(config=None, option=None):
    """entry point; parse command line argument, create pipeline object,
    and run it
    """
    print("+- APUS powered by Ruffus ver {0} -+".format(ruffus.__version__))
    if config is None:
        config = sys.modules['__main__']
    if option is None:
        option = sys.argv[1:]
    config, option = configure(config, option)
    # set up env with overrides
    env.am.set_overrides(config.env_overrides)

    # check existence of the jobdir
    def has_logdir():
        if not os.path.exists(config.logdir):
            utils.get_or_create_dir(config.logdir)
        return os.path.exists(config.logdir)
    if option.action == 'run':
        if not os.path.exists(config.jobdir):
            raise RuntimeError('job directory does not exist,'
                               ' run init to create one')
        elif not has_logdir():
            raise RuntimeError('unable to find/create log directory')
        else:
            build_pipeline(config, option)
    elif option.action == 'init':
        option.history_file = option.history_file + '.init'
        if not has_logdir():
            raise RuntimeError('unable to find/create log directory')
        else:
            build_init_pipeline(config, option)
    # handle redo-all
    if option.redo_all:
        task_list = ruffus.pipeline_get_task_names()
        option.forced_tasks.extend(task_list)
    if len(option.forced_tasks) > 0:
        for t in option.forced_tasks:
            config.logger.info("forced redo: {0}".format(utils.alert(t)))

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
        # refresh flowchart
        # subprocess.call(['python', sys.argv[0],
        #                  '-T', config.tlist[-1]['name'],
        #                  '--flowchart', 'test.png'])
        with config.logger_mutex:
            elapsed_time = timedelta(
                    seconds=time.time() - config.timestamp)
            config.logger.debug('task {0:^45s} finished @ {1:s}'
                                .format(utils.alert(task_name),
                                        utils.alert(elapsed_time))
                                )
    return ret_task_finish_signal


def get_flag_file(out_files):
    suffix = '.success'
    if len(out_files) == 1 and out_files[0][-len(suffix):] == suffix:
        flag = out_files[0]
        out_files = []
    else:
        flag = None
    return out_files, flag


def get_am_prog(func):
    program = [i for i in ['sex', 'scamp', 'swarp']
               if func.lower().startswith(i)][0]
    return program


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


def to_callable_task_args(conv_func):
    def wrapper(func):
        @wraps(func)
        def wrapped_func(in_files, out_files, *extras):
            if len(extras) == 0:  # from originate, rename the variables
                in_files, out_files, extras = [], in_files, out_files
            in_files += extras[:-1]
            out_files, flag_file = get_flag_file(out_files)
            overlap = list(set(in_files).intersection(out_files))
            if len(overlap) > 0:
                raise RuntimeError(
                        'danger: output {0} has same filename as inputs'
                        .format(unwrap_if_len_one(overlap)))
            context = copy(extras[-1])
            context['flag_file'] = flag_file
            if conv_func is not None:
                # copy task as well
                context['task'] = dict(context['task'], func=conv_func(
                        in_files, out_files, context))
            return func(in_files, out_files, context)
        return wrapped_func
    return wrapper


def _subprocess_callable(in_files, out_files, context):
    if any(isinstance(i, tuple) for i in in_files):
        if len(in_files) > 1:
            # raise RuntimeError(
            #     "subprocess task should not have nested inputs")
            in_files = [i for j in in_files for i in j]
        else:
            in_files = in_files[0]
    command = context['task']['func'].split()
    for key, value in zip(['{in}', '{out}'], [in_files, out_files]):
        if key in command:
            i = command.index(key)
            command[i:i + 1] = value
    return documented_subprocess_call(command, flag_file=context['flag_file'])


@ensure_args_as_list(0, 1, tuple_as_list=True)
@to_callable_task_args(_subprocess_callable)
def subprocess_task(in_files, out_files, context):
    """run command as subprocess by replacing func key in task context
    with callable"""
    return callable_task(in_files, out_files, context)


def _astromatic_callable(in_files, out_files, context):
    """return the command for executing astromatic task"""
    task = context['task']
    # get program
    prog = get_am_prog(task['func'])
    # split up inputs types
    rectified_inputs = get_astromatic_inputs(in_files, task['in_keys'])
    command = [env.am.get('{0}bin'.format(prog)), ]
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
    out_keys = task.get('out_keys', default_outkeys[prog])
    for i, key in enumerate(out_keys):
        command.extend(['-{0}'.format(key), out_files[i]])
    return documented_subprocess_call(command, flag_file=context['flag_file'])


@ensure_args_as_list(0, 1)
@to_callable_task_args(_astromatic_callable)
def astromatic_task(in_files, out_files, context):
    """create astromatic command to run by subprocess"""
    return callable_task(in_files, out_files, context)


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


@ensure_args_as_list(0, 1)
@to_callable_task_args(None)
def callable_task(in_files, out_files, context):
    task, logger, logger_mutex = [
            context[i] for i in ['task', 'logger', 'logger_mutex']]
    func = task['func']
    args = in_files + out_files
    dry_run = task.get('dry_run', False)
    verbose = task.get('verbose', True)
    if func.__doc__.startswith('subprocess: '):
        caller_string = func.__doc__[len('subprocess: '):]
    else:
        caller_string = "{0}({1})".format(func.__name__, ', '.join(args))
    if dry_run:
        caller_string = "~dry~run~1~sec: " + caller_string
    if verbose or dry_run:
        with logger_mutex:
            logger.debug(caller_string)
    if dry_run:
        time.sleep(1)
        # fix dry_run for originate
        if task['pipe'] == 'originate':
            touch_files = in_files + out_files
        else:
            touch_files = out_files
        map(common.touch_file, touch_files)
        output = '~dry~run~touched: {0}'.format(', '.join(touch_files))
    else:
        kwargs = dict(
                task['kwargs'],
                task=task,
                logger=logger,
                logger_mutex=logger_mutex)
        output = func(*args, **kwargs)
    if output or verbose:
        output = "finished silently" if not output else 'finished'
        with logger_mutex:
            logger.debug(output)


def dump_config_files(conf_file, checker_file, **kwargs):
    """create configuration file"""

    logger, logger_mutex = kwargs['logger'], kwargs['logger_mutex'],
    task = kwargs['task']
    prog = task['prog']
    am_bin = env.am.get('{0}bin'.format(prog))
    am_share = env.am.get('{0}share'.format(prog))
    # handle parameters
    conf_params = dict(env.am.get('{0}_default'.format(prog)),
                       **task.get('params', {}))
    if prog == 'sex':
        if 'PARAMETERS_NAME' not in task.get('in_keys', []):
            params_file = conf_params.get('PARAMETERS_NAME', None)
            if params_file is None or not os.path.isfile(params_file):
                params_file_keys = task.get('outparams', [])
                params_file = conf_file + '.sexparam'
                with logger_mutex:
                    logger.info('params file: {0}'.format(params_file))
                fo = StringIO(subprocess.check_output([am_bin, '-dp']))
                utils.dump_sex_param(
                        fo, params_file, env.am.get('sexparam_default'),
                        params_file_keys, clobber=True)
                # with logger_mutex:
                #     logger.info("keys: {0}".format(', '.join(keys)))
                fo.close()
                conf_params['PARAMETERS_NAME'] = params_file
    elif prog == 'scamp':
        conf_params['CHECKPLOT_NAME'] =\
            utils.get_scamp_checkplot_name(
                    os.path.abspath(task['diagdir']),
                    prefix=normalize_task_name(task['name']))
    elif prog == 'swarp':
        if conf_params.get('RESAMPLE_DIR', None) is None:
            conf_params['RESAMPLE_DIR'] = env.am.get('scratch_dir')
    # convert to strings
    # use absolute file paths whenever possible
    for k, v in conf_params.items():
        if isinstance(v, list):
            v = ', '.join(map(str, v))
        else:
            v = str(v)
        conf_params[k] = v
        if re.search('[/.]', v) is not None:  # only replace file-like vals
            if os.path.isfile(v):
                conf_params[k] = os.path.abspath(v)
            elif os.path.isfile(os.path.join(am_share, v)):
                conf_params[k] = os.path.abspath(os.path.join(am_share, v))
            else:
                conf_params[k] = v
    # create conf file
    fo = StringIO(subprocess.check_output([am_bin, '-dd']))
    utils.dump_astromatic_conf(fo, conf_file, clobber=True, **conf_params)
    with logger_mutex:
        logger.info('conf file: {0}'.format(conf_file))
        for k, v in conf_params.items():
            if os.path.isfile(v):
                v = os.path.relpath(v)
            logger.info('{0:>20s}: {1:s}'.format(k, v))
    fo.close()
    # write checker file
    with open(checker_file, 'w') as fo:
        pickle.dump(task.get('params', {}), fo)
        pickle.dump(task.get('outparams', {}), fo)


def check_config_uptodate(*args, **kwargs):
    conf_file, checker_file, context = args[-3:]
    for f in [conf_file, checker_file]:
        if not os.path.isfile(f):
            return True, "missing file {0}".format(f)
    task = context['task']
    if task['dry_run']:
        return False, "skipped checker due to dry run"
    with open(checker_file, 'r') as fo:
        try:
            old_params = pickle.load(fo)
            old_outparams = pickle.load(fo)
        except EOFError:
            return True, "corrupted checker"
    new_params = task.get('params', {})
    new_outparams = task.get('outparams', [])
    if len(new_params) != len(old_params) or \
            len(new_outparams) != len(old_outparams):
        return True, 'params/outparams changed its size'
    for key, val in new_params.items():
        if key in old_params.keys() and old_params[key] == val:
            continue
        else:
            return True, 'params dict changed its content'
    else:
        overlap = list(set(new_outparams).intersection(old_outparams))
        if len(overlap) == len(new_outparams):
            # print out params
            return False, "no change of params/outparams"
        else:
            return True, 'outparams list changed its content'


def documented_subprocess_call(command, flag_file=None):
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
        log = common.get_log_func(**kwargs)

        # with NamedTemporaryFile() as fo:
        #     subprocess.check_call(command, stdout=fo,
        #                           stderr=subprocess.STDOUT)
        #     fo.seek(0)
        #     output = fo.read()
        # output = subprocess.check_output(command)
        proc = subprocess.Popen(command,
                                stdout=subprocess.PIPE,
                                bufsize=1,
                                # stderr=subprocess.PIPE
                                )
        has_output = False
        for ln in iter(proc.stdout.readline, b''):
            if ln:
                has_output = True
            log('debug', ln.strip('\n'))
        if proc.poll() is not None and proc.returncode != 0:
            err_msg = "subprocess failed with code {0}".format(proc.returncode)
            # an error happened!
            # err_msg = "%s\nsubprocess failed with code: %s" % (
            #         proc.stderr.read(),
            #         proc.returncode)
            raise RuntimeError(err_msg)
        if flag_file is not None:
            common.touch_file(flag_file)
        return has_output
        # return output
    # print(' '.join(command))
    call.__doc__ = 'subprocess: ' + ' '.join(command)
    return call

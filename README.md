# Astronomy Pipeline Using ruffuS (APUS)

The `APUS` package is for construction of small to medium sized astronomy
related data reduction work flow in a scalable, flexible and repeatable
manner. It makes use the [Ruffus](https://github.com/bunbun/ruffus) package for
pipeline management, and ships with a number of useful task functions and
scripts, most notably, helpers for `AstrOmatic` stack (`sextractor`, `scamp`,
and `swarp`)

## Highlights

1. friendly to `AstrOmatic` software suite: centralized configuration file
   generation/management
2. flow-based job execution: parallelism, check-pointing, etc.

## Design notes

### Semantics

1. user defines pipeline in a single configuration file

    ```sh
    # PSEUDO CODE
    # define work directory layout
    jobkey, jobdir, [confdir, ...]

    # define input files
    inputs, [per_input_extra, extra, ...]

    # define symbolic linking rules for the original input files
    input_reg, [extra_reg, ...]     # regex for parsing the filename
    input_fmt, [extra_fmt, ...]     # fmt string for symlink target

    # define astromatic configuration files to dump
    [am_sharedir, am_diagdir, am_resampdir, ...]  # directories
    am_params                       # dict with keys like "prog[_suffix]"
                                    # each key will expand to a config file

    # define dictionaries that describe the tasks in pipeline
    t01 = {
        'name': 'do foo'
        'func': 'prog_suffix | foo.sh {in} {out}' | callable
                                    # prog_suffix: astromatic task using same
                                    #              key as in am_params
                                    # cmdline call: {in} {out} will be replaced
                                    #              by inputs and outputs
                                    # callable: called as func(inputs, outputs)
        'type_': 'transform | collate | ...'      # ruffus task decorators
       ['follows': 'task_name' or list]           # follows the given task
        'in_': 'glob_pattern| task_name' or list  # input files
        'reg': 'regex'                            # regex parses the inputs
       ['add_inputs': 'fmt' or list]              # additional inputs
        'params': 'extra param'                   # override the am_param config
        'out': 'fmt' or list                      # fmt string defines the output
       ['outkey': 'config key' or list]           # config key for each output
    }
    tlist = t01, t02, ...

    # entry point
    core.bootstrap()                     #  does all the magic
    ```

2. call signatures

    ```sh
    1. $<script> init -args   # prepare directories, create links, dump configuration files
    2. $<script> -args        # run the pipeline
    ```


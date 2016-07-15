# Astronomy Pipeline Using ruffuS (APUS)

The `APUS` package is for construction of small to medium sized astronomy
related data reduction work flow in a scalable, flexible and repeatable
manner. It makes use the [Ruffus](https://github.com/bunbun/ruffus) package for
pipeline management, and ships with a number of useful task functions and
scripts, most notably, helpers for `AstrOmatic` stack (`sextractor`, `scamp`,
and `swarp`)

## Highlights

1. Declarative pipeline task definition means shorter code than working with
   Ruffus directly

2. Friendly to `AstrOmatic` software suite: simplified task creation;
   auto generation/management of the configuration files

3. All goodies from Ruffus: flow-based job execution, parallelism, check-pointing, etc.

## Usage

1. Compose a python script that defines the pipeline tasks declaratively

    ```sh
    # PSEUDO CODE
    # file: apus_pipeline_example.py

    jobkey      # Mandatory; id of the job

    # work directory layout
    jobdir      # Mandatory; workdir of the job
    confdir     # Optional, default is jobdir; hold static/config files
    diagdir     # Optional, default is jobdir; scamp diagnostic plots

    # define input files; input files will be sym-linked to jobdir
    # no need to wrap in list if only one entry
    inputs = [                 # Mandatroy; each entry in the last corresponse to a set of inputs
        (orig1, reg1, fmt1),   # orig: globpattern or filename to the original files
        (orig2, reg2, fmt2),   # reg: regular expression to parse the file name to semantic parts,
        orig3,                 #      Internally it is used to construct a Ruffus `formatter`
        orig4,                 # fmt: formattter template string for sym-linked filename in jobdir;
        ]                      #      Internally it is processed by the Ruffus formatter substitution mechanism
                               # reg and fmt can be omitted, in which case the file will be linked using its original filename
    # for details of formatter object, see http://www.ruffus.org.uk/tutorials/new_tutorial/output_file_names.html#new-manual-formatter

    # APUS related config items
    env_overrides = {}         # Optional; a dict that overrides the attribute in `env.AmConf` class (see env.py)
                               # the most useful case is to specify the prefix directory of astromatic
                               # software installation `path_prefix`
    task_io_default_dir = jobdir  # Optional; default is jobdir
                                  # this will prepend any in/out filenames in the task
                                  # definition with the said directory,
                                  # result in a cleaner look of the task definitions
                                  # diable this feature by set it to empty string ""

    # Pipeline task definitions
    t_01 = {
        'name': 'foo bar',
        'func': './do_foobar.sh {in} {out}',  # function will be called as:
        'pipe': 'transform',             # do_foobar (in1 add1 add2) ex1 ex2 out1 out2
        'in_': ('in1', 'reg1'),
        'add_inputs': ['add1', 'add2'],  # Optional
        'extras': ['ex1', 'ex2'],        # Optional
        'out': ['fmt1', 'fmt2'],
        'dry_run': False,                # Optional
        'follows': 'taskname'            # Optional
    }
    t_02 = {
        'name': 'get catalog',
        'func': 'sex',                   # funcion will be called as:
        'pipe': 'transform',             # sex in1 -WEIGHT_IMAGE add1,add2 -c ex1 \
        'in_': ('in1', 'reg1'),          #    -CATALOG_NAME out1 -CHECKIMAGE_NAME out2 \
        'add_inputs': ['add1', 'add2'],  #    -WEIGHT_TYPE MAP_WEIGHT -CHECKIMAGE_TYPE SEGMENTATION
        'extras': 'ex1',
        'in_keys': [('in', 'WEIGHT_IMAGE'), 'conf']       # match to the inputs siganiture
        'out': ['fmt1', 'fmt2'],
        'out_keys': ['CATALOG_NAME', 'CHECKIMAGE_NAME'],  # match to the outputs siganiture
        'params': {'WEIGHT_TYPE': 'MAP_WEIGHT',
                   'CHECKIMAGE_TYPE': 'SEGMENTATION'},
        'dry_run': False,                # Optional
        'follows': 'taskname'            # Optional
    }

    tlist = [t_01, t_02]        # Mandatory; tasks included in the list will be processed

    if __name__ == "__main__":  # Mandatory; call APUS to bootstrap
        from apus import core
        core.bootstrap()
    ```

2. Call the script

    ```sh
    1. $ apus_pipeline_example.py init -args   # prepare directories, create links, dump configuration files
    2. $ apus_pipeline_example.py -args        # run the pipeline
    ```


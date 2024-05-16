import glob
import logging
import os
import random
from pathlib import Path
import platform

import numpy as np
import torch
import yaml
import re

VERBOSE = str(os.getenv('VERBOSE', True)).lower() == 'true'  # global verbose mode
__version__ = '0.1.0'

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict) # check input is a dictionary
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
               
    def __str__(self) -> str:
        return str(self.__dict__)
    
    def add(self, key, value):
        setattr(self, key, value)
        
    def add_args(self, args):
        for key, value in vars(args).items():
            self.add(key, value)
    
    def dump(self):
        data_dump = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictObj):
                value = value.dump()
            if isinstance(value, Path):
                value = str(value)
            data_dump[key] = value
        return data_dump

def yaml_load(file='config.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)
    
def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if not test:
        return os.access(dir, os.R_OK)  # possible issues on Windows
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False

def set_logging(name=None, verbose=VERBOSE):
    level = logging.INFO
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] - %(levelname)s - %(name)s - %(message)s'))
    handler.setLevel(level)
    log.addHandler(handler)

set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger(f"MAE 🚀 {__version__}")  # define globally

def set_logging(verbose=True):
    logging.basicConfig(
        format='[%(asctime)s] - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO)

def print_args(name, opt):
    # Print argparser arguments
    print(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]

def create_new_exp(path, exp_name='', sep=''):
    path = Path(path)
    
    if path.exists():
        if exp_name == '':
            dirs = glob.glob(f"{path}/exp{sep}*")  # similar paths
            matches = [re.search(rf"%s/exp{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 0
            p = path / f'exp{sep}{n}'
        else:
            p = path / exp_name
    else:
        path.mkdir(parents=True, exist_ok=True)
        n = 0
        
        if exp_name == '':
            p = path / f'exp{sep}{n}'
        else:
            p = path / exp_name

    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p

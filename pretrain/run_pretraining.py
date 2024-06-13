import os
import sys
import logging
from pathlib import Path
import argparse


from util.general import (
    yaml_load, 
    DictObj, 
    set_logging, 
    print_args, 
    create_new_exp
)
from util.callbacks import Callbacks
from trainer import Trainer


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOGGER = logging.getLogger(__name__)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='experiment configuration file name (e.g. pretrain)', required=True)
    parser.add_argument('--norm_pix_loss', action='store_true', help='normalize pixel loss')
    parser.add_argument('--exp_name', type=str, help='experiment name', required=True)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def setup(cfg):
    set_logging()
    
    print_args(FILE.stem, cfg)
    new_exp = create_new_exp(Path(cfg.project), exp_name=cfg.exp_name)
    cfg.add('save_dir', Path(new_exp))

def main(opt, callbacks=Callbacks()):
    cfg = yaml_load(ROOT / 'configs' / f'{opt.cfg}.yaml')
    cfg = DictObj(cfg)
    cfg.add_args(opt)
    
    setup(cfg)

    trainer = Trainer(cfg, callbacks)
    if cfg.resume_checkpoint != '':
        trainer.train(resume_checkpoint=cfg.resume_checkpoint)
    else:
        trainer.train()
        
    LOGGER.info('Training completed!')

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

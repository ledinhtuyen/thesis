import os
import sys
from pathlib import Path
import time
import math
from contextlib import redirect_stdout
import yaml
from datetime import timedelta
import logging
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import inf
from einops import rearrange
import timm.optim.optim_factory as optim_factory

import util.lr_sched as lr_sched
from util.logger import Logger
from util.general import init_seeds, colorstr, methods
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.utils import MetricMeter
import model
from dataset.Medical import *

torch.autograd.set_detect_anomaly(True)

LOGGER = logging.getLogger(__name__)

class Trainer:
  def __init__(self, cfg, callbacks=None):
    LOGGER.info(colorstr('Initializing Trainer...'))
    self.cfg = cfg
    self.callbacks = callbacks
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(colorstr(f"Device: {self.device}"))
    self.current_epoch = 0
    self.best_loss = inf

    self.setup(cfg)
    self.build_model(cfg)
    self.build_optimizer(cfg)
    self.build_dataloader(cfg)

    self.metric_logger = self.init_metric_logger()
  
  def init_metric_logger(self):
    metric_logger = MetricMeter(delimiter="   ")
    metric_logger.add_meter('train/loss')
    metric_logger.add_meter('train/lr')
    metric_logger.add_meter('val/loss')
    metric_logger.add_meter('val/best_loss')
    return metric_logger
    
  def setup(self, cfg):
    LOGGER.info(colorstr('Setup Environment...'))
    self.snapshot_dir, self.log_dir, self.epochs, self.batch_size, data  = \
      Path(cfg.save_dir) / cfg.snapshot_dir, Path(cfg.save_dir) / cfg.log_dir , cfg.epochs, cfg.Dataset.batch_size, cfg.Dataset.data_name
    self.save_period = cfg.save_period

    # Directories
    self.snapshot_dir.mkdir(parents=True, exist_ok=True)  # make dir
    self.last, self.best = self.snapshot_dir / 'last.pt', self.snapshot_dir / 'best.pt'

    with open(self.snapshot_dir / 'opt.yaml', 'w') as f:
      with redirect_stdout(f): # write to file
        yaml.dump(cfg.dump(), f, default_flow_style=False, indent=4)
    
    logger = Logger(self.log_dir, cfg, LOGGER)  # loggers instance

    # Register actions
    for k in methods(logger):
        self.callbacks.register_action(k, callback=getattr(logger, k))

    # Set random seed
    init_seeds(2024)

  def build_model(self, cfg):
    # Get model from cfg.Model.name in mae/model/mae.py
    LOGGER.info(colorstr('Building Model {}...'.format(cfg.Model.name)))
    self.model = model.__dict__[cfg.Model.name](
      img_size=cfg.Dataset.img_size,
      norm_pix_loss=cfg.norm_pix_loss
    )
    self.model = self.model.to(self.device)
    LOGGER.info(f"Model: {cfg.Model.name}")
    LOGGER.info(self.model)

  def build_optimizer(self, cfg):
    LOGGER.info(colorstr('Building Optimizer...'))
    
    lr = (cfg.hyp.base_lr * cfg.Dataset.batch_size * cfg.hyp.gradient_accumulation_steps) / 256
    weight_decay = cfg.hyp.weight_decay
      
    self.cfg.hyp.add('lr', lr)
    
    param_groups = optim_factory.param_groups_weight_decay(self.model, weight_decay=weight_decay)
    self.optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    LOGGER.info(f"Optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")

    self.loss_scaler = NativeScaler()
    LOGGER.info(self.optimizer)

  def build_dataloader(self, cfg):
    LOGGER.info(colorstr('Building DataLoader...'))
    meanstd = {'mean': cfg.Dataset.mean, 'std': cfg.Dataset.std}
    LOGGER.info(f"{colorstr('Mean')}: {meanstd['mean']}, {colorstr('Std')}: {meanstd['std']}")
    
    data = json.load(open(cfg.Dataset.annotation_file))
    train_data = data["train"]
    train_dataset = PretrainMedical(train_data, meanstd=meanstd, prefix_path=cfg.Dataset.prefix_path, train=True)
    self.train_dataloader = DataLoader(
      train_dataset, 
      batch_size=cfg.Dataset.batch_size, 
      shuffle=True, 
      num_workers=cfg.Dataset.num_workers, 
      pin_memory=cfg.Dataset.pin_memory,
      drop_last=cfg.Dataset.drop_last
    )
    LOGGER.info(f"Train Iterations: {len(self.train_dataloader)}")
    
    if data.get("test") is not None:
      test_data = data["test"]
      test_dataset = PretrainMedical(test_data, meanstd=meanstd, prefix_path=cfg.Dataset.prefix_path, train=False)
      self.test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.Dataset.batch_size,
        shuffle=True,
        num_workers=cfg.Dataset.num_workers,
        pin_memory=cfg.Dataset.pin_memory
      )
      LOGGER.info(f"Test Iterations: {len(self.test_dataloader)}")

  def train_one_epoch(self):
    self.model.train()
    self.optimizer.zero_grad()
    header = 'Epoch [{}]: '.format(self.current_epoch)
    LOGGER.info(header)

    accumulate_iter = self.cfg.hyp.gradient_accumulation_steps

    start_time_one_epoch = time.time()
    for data_iter_step, data in enumerate(tqdm(self.train_dataloader)):
      if data_iter_step % accumulate_iter == 0:
        lr_sched.adjust_learning_rate(self.optimizer, data_iter_step / len(self.train_dataloader) + self.current_epoch, self.cfg)
      
      data = data.to(self.device)
    
      with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        loss, _, _ = self.model(data)

      loss_value = loss.item()

      if not math.isfinite(loss_value):
        print("MAE Loss is {}, stopping training".format(loss_value))
        sys.exit(1)

      loss /= accumulate_iter
      self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                    update_grad=(data_iter_step + 1) % accumulate_iter == 0)
      if (data_iter_step + 1) % accumulate_iter == 0:
        self.optimizer.zero_grad()
      
      lr = self.optimizer.param_groups[0]["lr"]  
      self.metric_logger.update({'train/loss': loss_value, 'train/lr': lr})

      if (data_iter_step + 1) % accumulate_iter == 0:
        self.callbacks.run('on_train_accumulate_iter_end', metric_logger=self.metric_logger, global_step=int((data_iter_step / len(self.train_dataloader) + self.current_epoch) * 1000), epoch=self.current_epoch)

    self.callbacks.run('on_train_epoch_end', time=time.time() - start_time_one_epoch, epoch=self.current_epoch)

  def test_one_epoch(self):
    self.model.eval()

    visualize = True
    total_test_loss = 0
    with torch.no_grad():
      for data_iter_step, data in enumerate(tqdm(self.test_dataloader)):
        data = data.to(self.device)
        loss, pred, mask = self.model(data)
        data, pred, mask = data[:self.cfg.visual_imgs].cpu(), pred[:self.cfg.visual_imgs].cpu(), mask[:self.cfg.visual_imgs].cpu()
        
        if visualize and self.current_epoch % self.save_period == 0:
          patch_size = self.model.patch_size
          
          mask = mask.unsqueeze(-1).tile(1, 1, patch_size ** 2 * 3)
          mask = self.model.unpatchify(mask)
          pred = self.model.unpatchify(pred)
          pred = pred * mask + data * (1 - mask)

          img = torch.cat([data * (1 - mask), pred, data], dim=0)
          img = rearrange(img, '(v h1) c h w -> v c (h1 h) w', h1=self.cfg.visual_imgs)
          
          self.callbacks.run('on_val_batch_end', img=img, epoch=self.current_epoch)
          visualize = False

        total_test_loss += loss.item()

    test_loss = total_test_loss / len(self.test_dataloader)
    self.metric_logger.update({'val/loss': test_loss})

    if test_loss < self.best_loss:
      self.best_loss = test_loss
      self.metric_logger.update({'val/best_loss': test_loss})
      self.save_checkpoint('best.pth')
    self.callbacks.run('on_val_end', metric_logger=self.metric_logger, epoch=self.current_epoch)
    return test_loss

  def train(self, resume_checkpoint=None):
    self.model.train()
    self.current_epoch = 0
    if resume_checkpoint:
      self.current_epoch = self.load_checkpoint(resume_checkpoint)
    
    LOGGER.info(f'{colorstr("Start Training:")} from epoch {self.current_epoch} to {self.epochs}')
    start_time = time.time()

    # Start training
    for epoch in range(self.current_epoch, self.epochs):
      self.current_epoch = epoch
      self.train_one_epoch()

      if self.current_epoch % self.cfg.save_ckpt_freq == 0:
        self.save_checkpoint(f'epoch_{self.current_epoch}.pth')

      if self.current_epoch % self.save_period == 0:
        # Save last checkpoint
        self.save_checkpoint()

      if hasattr(self, 'test_dataloader'):
        self.test_one_epoch()
      LOGGER.info(f"Averaged stats: {self.metric_logger}")
      
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    LOGGER.info(f"Training time {total_time_str}")

  def save_checkpoint(self, filename='last.pth'):
      checkpoint_path = os.path.join(self.snapshot_dir, filename)
      checkpoint = {'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': self.loss_scaler.state_dict(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'loss': self.metric_logger.get_meter('train/loss').get_val(),
                    }
      torch.save(checkpoint, checkpoint_path)
      logging.info(f"Checkpoint saved at {checkpoint_path}")

  def load_checkpoint(self, checkpoint_path):
      checkpoint = torch.load(checkpoint_path)
      if 'epoch' not in checkpoint.keys():
        self.current_epoch = 0
        self.model.load_state_dict(checkpoint['model'], strict=False)
      else:
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(self.optimizer.param_groups[0]['lr'], checkpoint['lr'])
        self.current_epoch = checkpoint['epoch']
        self.loss_scaler.load_state_dict(checkpoint['scaler'])
        self.optimizer.param_groups[0]['lr'] = checkpoint['lr']
      LOGGER.info(f"Checkpoint loaded from {checkpoint_path} with epoch {self.current_epoch}, lr={self.optimizer.param_groups[0]['lr']}")
      return self.current_epoch + 1

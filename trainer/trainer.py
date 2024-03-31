import os
import sys
from pathlib import Path
import time
import math
from contextlib import redirect_stdout
import yaml

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import inf
from einops import rearrange

from datetime import datetime
import logging
from tqdm import tqdm
import util.lr_sched as lr_sched
from util.logger import Logger
from util.general import init_seeds, colorstr, methods
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.utils import MetricMeter
import model.mae as mae
from dataset.Medical import *

LOGGER = logging.getLogger(__name__)

class Trainer:
  def __init__(self, cfg, callbacks=None):
    LOGGER.info(colorstr('Initializing Trainer...'))
    self.cfg = cfg
    self.callbacks = callbacks
    self.gradient_accumulation_steps = cfg.hyp.gradient_accumulation_steps
    self.effect_batch_size = cfg.Dataset.batch_size * cfg.hyp.gradient_accumulation_steps
    self.cfg.hyp.add('lr', cfg.hyp.base_lr * self.effect_batch_size / 256)

    self.setup(cfg)
    self.build_model(cfg)
    self.build_optimizer(cfg)
    self.build_dataloader(cfg)
    self.current_epoch = 0

    self.metric_logger = self.init_metric_logger()
  
  def init_metric_logger(self):
    metric_logger = MetricMeter(delimiter="   ")
    metric_logger.add_meter('loss')
    metric_logger.add_meter('lr')
    metric_logger.add_meter('test_loss')
    metric_logger.add_meter('best_loss')
    metric_logger.update({'best_loss': inf})
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
    LOGGER.info(colorstr('Building Model...'))
    self.model = mae.__dict__[cfg.Model.name](
      img_size=cfg.Dataset.img_size,
      norm_pix_loss=cfg.norm_pix_loss
    )
    LOGGER.info(f"Model: {cfg.Model.name}")
    LOGGER.info(self.model)
    return self.model.cuda()

  def build_optimizer(self, cfg):
    LOGGER.info(colorstr('Building Optimizer...'))
    self.optimizer = AdamW(self.model.parameters(), lr=cfg.hyp.lr, weight_decay=cfg.hyp.weight_decay, betas=(0.9, 0.95))
    self.loss_scaler = NativeScaler()
    LOGGER.info(f"Optimizer: AdamW(lr={cfg.hyp.base_lr * self.effect_batch_size / 256}, weight_decay={cfg.hyp.weight_decay})")
    LOGGER.info(self.optimizer)

  def build_dataloader(self, cfg):
    LOGGER.info(colorstr('Building DataLoader...'))
    medical_data = Medical(Path(cfg.Dataset.prefix_path), Path(cfg.Dataset.annotation_file))
    train_dataset = PretrainMedical(medical_data, train=True, transform=train_transform(input_size=cfg.Dataset.img_size))
    test_dataset = PretrainMedical(medical_data, train=False, transform=test_transform(input_size=cfg.Dataset.img_size))

    self.train_dataloader = DataLoader(
      train_dataset, 
      batch_size=self.effect_batch_size, 
      shuffle=True, 
      num_workers=cfg.Dataset.num_workers, 
      pin_memory=cfg.Dataset.pin_memory,
      drop_last=cfg.Dataset.drop_last
    )
    self.test_dataloader = DataLoader(
      test_dataset,
      batch_size=self.effect_batch_size,
      shuffle=True,
      num_workers=cfg.Dataset.num_workers,
      pin_memory=cfg.Dataset.pin_memory
    )

    LOGGER.info(f"Train Iterations: {len(self.train_dataloader)}")
    LOGGER.info(f"Test Iterations: {len(self.test_dataloader)}")

  def train_one_epoch(self):
    self.model.train()
    self.optimizer.zero_grad()
    header = 'Epoch [{}]: '.format(self.current_epoch)
    LOGGER.info(header)


    for data_iter_step, data in enumerate(tqdm(self.train_dataloader)):
      if data_iter_step % self.gradient_accumulation_steps == 0:
        lr_sched.adjust_learning_rate(self.optimizer, data_iter_step / len(self.train_dataloader) + self.current_epoch, self.cfg)
      
      data = data.cuda()
    
      with torch.cuda.amp.autocast():
        loss, _, _ = self.model(data)

      loss_value = loss.item()
      if not math.isfinite(loss_value):
        print("Loss is {}, stopping training".format(loss_value))
        sys.exit(1)
      loss /= self.gradient_accumulation_steps
      self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                    update_grad=(data_iter_step + 1) % self.gradient_accumulation_steps == 0)
      if (data_iter_step + 1) % self.gradient_accumulation_steps == 0:
        self.optimizer.zero_grad()
      
      lr = self.optimizer.param_groups[0]["lr"]  
      self.metric_logger.update({'loss': loss_value, 'lr': lr})
      
      if (data_iter_step + 1) % self.gradient_accumulation_steps == 0:
        self.callbacks.run('on_train_accumulate_iter_end', loss=loss_value, lr=lr, global_step=int((data_iter_step / len(self.train_dataloader) + self.current_epoch) * 1000), epoch=self.current_epoch)


    print("Averaged stats:", self.metric_logger)

  def test_one_epoch(self):
    self.model.eval()

    visualize = True
    
    with torch.no_grad():
      for data_iter_step, data in enumerate(tqdm(self.test_dataloader)):
        data = data.cuda()
        loss, pred, mask = self.model(data)
        data, pred, mask = data.cpu(), pred.cpu(), mask.cpu()
        
        if visualize and self.current_epoch % self.save_period == 0:
          patch_size = self.model.patch_size
          grid_size = self.model.grid_size
          
          mask = mask.unsqueeze(-1).tile(1, 1, patch_size ** 2 * 3)
          mask = rearrange(mask, 'b (g g1) (c p p1) -> b c (g p) (g1 p1)', c=3, p=patch_size, g=grid_size)
          pred = rearrange(pred, 'b (g g1) (c p p1) -> b c (g p) (g1 p1)', c=3, p=patch_size, g=grid_size)
          pred = pred * mask + data * (1 - mask)

          img = torch.cat([data * (1 - mask), pred, data], dim=0)
          img = rearrange(img[:self.cfg.visual_imgs * 3], '(v h1) c h w -> v c (h1 h) w', h1=self.cfg.visual_imgs)
          
          self.callbacks.run('on_val_batch_end', img=img, epoch=self.current_epoch)
          visualize = False

        test_loss = loss.item()
        self.metric_logger.update({'test_loss': test_loss})

        if test_loss < self.metric_logger.best_loss.avg:
          self.metric_logger.best_loss.update(test_loss)
          self.save_checkpoint('best.pth')
    self.callbacks.run('on_val_end', metric_logger=self.metric_logger, epoch=self.current_epoch)

  def train(self, resume_checkpoint=None):
    self.model.train()
    if resume_checkpoint:
      self.current_epoch = self.load_checkpoint(resume_checkpoint)
    
    LOGGER.info(f'{colorstr("Start Training:")} from epoch {self.current_epoch} to {self.epochs}')
    start_time = time.time()
    for epoch in range(self.current_epoch, self.epochs):
      self.current_epoch = epoch
      self.train_one_epoch()
      if self.current_epoch % self.save_period == 0:  
        self.save_checkpoint()
      self.test_one_epoch()
      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    LOGGER.info(f"Training time {total_time_str}")

  def save_checkpoint(self, filename='last.pth'):
      checkpoint_path = os.path.join(self.snapshot_dir, filename)
      checkpoint = {'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': self.loss_scaler.state_dict(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'loss': self.metric_logger.loss,
                    }
      torch.save(checkpoint, checkpoint_path)
      logging.info(f"Checkpoint saved at {checkpoint_path}")

  def load_checkpoint(self, checkpoint_path):
      checkpoint = torch.load(checkpoint_path)
      self.model.load_state_dict(checkpoint['model'])
      self.optimizer.load_state_dict(checkpoint['optimizer'])
      print(self.optimizer.param_groups[0]['lr'], checkpoint['lr'])
      self.current_epoch = checkpoint['epoch']
      self.loss_scaler.load_state_dict(checkpoint['scaler'])
      self.optimizer.param_groups[0]['lr'] = checkpoint['lr']
      LOGGER.info(f"Checkpoint loaded from {checkpoint_path} with epoch {self.current_epoch}, lr={self.optimizer.param_groups[0]['lr']}")
      return self.current_epoch + 1

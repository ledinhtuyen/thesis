"""
Logging utils
"""

import os

from torch.utils.tensorboard import SummaryWriter

from util.general import colorstr, emojis

LOGGERS = ('tb',)  # TensorBoard, (Weights & Biases is comming soon)

class Logger():
    # Logger class
    def __init__(self, log_dir=None, cfg=None, logger=None, include=LOGGERS):
        self.log_dir = log_dir
        self.cfg = cfg
        # self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary

        # TensorBoard
        s = self.log_dir
        if 'tb' in self.include:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

    def on_train_accumulate_iter_end(self, loss, lr, global_step, epoch):
        # Callback runs on train accumulate iter end
        if self.tb:
            self.tb.add_scalar('train/epoch', epoch, global_step)
            self.tb.add_scalar('train/global_step', global_step, global_step)
            self.tb.add_scalar('train/loss', loss, global_step)
            self.tb.add_scalar('train/lr', lr, global_step)

    def on_train_epoch_end(self, time, epoch):
        if self.tb:
            self.tb.add_scalar('train/time', time, epoch)

    def on_val_batch_end(self, img, epoch):
        if self.tb:
            self.tb.add_images('val/images', img, epoch)
            
    def on_val_end(self, metric_logger, epoch):
        if self.tb:
            self.tb.add_scalar('val/loss', metric_logger.get_meter('test_loss').get_val(), epoch)
            self.tb.add_scalar('val/best_loss', metric_logger.get_meter('best_loss').get_val(), epoch)
            self.tb.add_scalar('val/epoch', epoch, epoch)

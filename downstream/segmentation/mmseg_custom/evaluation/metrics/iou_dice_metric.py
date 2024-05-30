# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS

@METRICS.register_module()
class IoU_Dice_Metric(BaseMetric):
    def __init__(self,
                ignore_index: int = 255,
                metrics: List[str] = ['mIoU', 'mDice'],
                nan_to_num: Optional[int] = None,
                beta: int = 1,
                collect_device: str = 'cpu',
                output_dir: Optional[str] = None,
                format_only: bool = False,
                prefix: Optional[str] = None,
                **kwargs) -> None:
      super().__init__(collect_device=collect_device, prefix=prefix)

      self.ignore_index = ignore_index
      self.metrics = metrics
      self.nan_to_num = nan_to_num
      self.beta = beta
      self.output_dir = output_dir
      if self.output_dir and is_main_process():
          mkdir_or_exist(self.output_dir)
      self.format_only = format_only
  
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                              self.ignore_index))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        

        # mask = (label != ignore_index)
        # pred_label = pred_label[mask]
        # label = label[mask]

        # intersect = pred_label[pred_label == label]
        # area_intersect = torch.histc(
        #     intersect.float(), bins=(num_classes), min=0,
        #     max=num_classes - 1).cpu()
        # area_pred_label = torch.histc(
        #     pred_label.float(), bins=(num_classes), min=0,
        #     max=num_classes - 1).cpu()
        # area_label = torch.histc(
        #     label.float(), bins=(num_classes), min=0,
        #     max=num_classes - 1).cpu()
        # area_union = area_pred_label + area_label - area_intersect
        # return area_intersect, area_union, area_pred_label, area_label

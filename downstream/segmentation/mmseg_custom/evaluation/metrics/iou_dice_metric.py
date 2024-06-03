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

from mmseg.registry import METRICS, EVALUATOR

@METRICS.register_module()
@EVALUATOR.register_module()
class IoUDiceMetricForBinarySegmentation(BaseMetric):
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
            pred_label = data_sample.pred_sem_seg.data.squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample.gt_sem_seg.data.squeeze().to(
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

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label
    
    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes mIoU, mDice, mPrecision, mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        mPrecision = 0
        mRecall = 0
        mDice = 0
        mIoU = 0
        for result in results:
            intersect_, union_, pred_label_, label_ = result
            intersect, union, pred_label, label = intersect_[1], union_[1], pred_label_[1], label_[1]
            mPrecision += intersect / (pred_label + 1e-7)
            mRecall += intersect / (label + 1e-7)
            mDice += 2 * intersect / (pred_label + label + 1e-7)
            mIoU += intersect / (union + 1e-7)

        num_samples = len(results)
        mPrecision = mPrecision / num_samples
        mRecall = mRecall / num_samples
        mDice = mDice / num_samples
        mIoU = mIoU / num_samples
        
        ret_metrics = OrderedDict()
        
        for metric in self.metrics:
            if metric == 'mDice':
                ret_metrics["mDice"] = mDice
            elif metric == 'mIoU':
                ret_metrics["mIoU"] = mIoU
            else:
                raise ValueError(f'Invalid metric name {metric}')
            
            ret_metrics["mPrecision"] = mPrecision
            ret_metrics["mRecall"] = mRecall
            
        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        
        if self.nan_to_num is not None:
            for metric, value in ret_metrics.items():
                ret_metrics[metric] = np.nan_to_num(value, nan=self.nan_to_num)
                
        class_names = self.dataset_meta['classes'][1]  # exclude background
        
        # summary table
        
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        ret_metrics_summary = OrderedDict({
                    ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                    for ret_metric, ret_metric_value in ret_metrics.items()
                })
        return ret_metrics_summary

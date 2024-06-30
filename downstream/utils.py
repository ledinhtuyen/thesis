import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return np.mean(np.stack(self.losses))
    
class MetricMeter(object):
    """A collection of metrics.

    Source: https://github.com/KaiyangZhou/Dassl.pytorch

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AvgMeter)
        self.delimiter = delimiter
        
    def add_meter(self, name):
        if isinstance(name, list):
            for n in name:
                self.meters[n] = AvgMeter()
        else:
            if name in self.meters:
                raise ValueError(
                    'The meter named {} already exists.'.format(name)
                )
            self.meters[name] = AvgMeter()
        
    def get_meter(self, name):
        return self.meters[name]

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{}: {:.4f}'.format(name, meter.show())
            )
        return self.delimiter.join(output_str)
    
    def get_avg(self):
        res = []
        for name, meter in self.meters.items():
            res.append(meter.avg)
        return res
    
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 1e-7)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 1e-7)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+1e-7))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + 1e-7)

def get_macro_scores(gts, prs): # Macro
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0

    for gt, pr in zip(gts, prs):
        mean_precision += precision_m(gt, pr)
        mean_recall += recall_m(gt, pr)
        mean_iou += iou_m(gt, pr)
        mean_dice += dice_m(gt, pr)

    if len(gts) != 0:
        mean_precision /= len(gts)
        mean_recall /= len(gts)
        mean_iou /= len(gts)
        mean_dice /= len(gts)
    
    # print("Macro scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))

    return (mean_iou, mean_dice, mean_precision, mean_recall)

def get_micro_scores(gts, prs): # Micro
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    
    total_area_intersect = 0
    total_area_union = 0
    total_pr = 0
    total_gt = 0
    
    for gt, pr in zip(gts, prs):
        total_area_intersect += torch.sum(torch.round(torch.clip(gt * pr, 0, 1)))
        total_area_union += torch.sum(torch.round(torch.clip(gt + pr, 0, 1)))
        total_pr += torch.sum(torch.round(torch.clip(pr, 0, 1)))
        total_gt += torch.sum(torch.round(torch.clip(gt, 0, 1)))
        
    mean_precision = total_area_intersect / (total_pr + 1e-7)
    mean_recall = total_area_intersect / (total_gt + 1e-7)
    mean_iou = total_area_intersect / (total_area_union + 1e-7)
    mean_dice = 2 * total_area_intersect / (total_pr + total_gt + 1e-7)
    
    # print("Micro scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))
    
    return (mean_iou, mean_dice, mean_precision, mean_recall)

class FocalLossV1(nn.Module):
    
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()

def plot_confusion_matrix(cm, class_names, normalize=False):
    """
    Plots a confusion matrix using matplotlib.

    Args:
    cm (array, shape = [n, n]): confusion matrix
    class_names (list): List of class names
    normalize (bool): Whether to normalize the values to percentages
    """
    cm = cm.cpu().numpy()
    if normalize:
        cm = np.nan_to_num(cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7))

    # Big size
    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(cm, cmap='Blues')

    ax.set_title('Confusion matrix')
    fig.colorbar(cax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks, class_names, rotation=45)
    ax.set_yticks(tick_marks, class_names)

    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    # To HWC
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

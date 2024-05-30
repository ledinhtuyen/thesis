import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.losses.utils import weighted_loss

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

@weighted_loss
def structure_loss(pred, mask, loss_type='bce', **kwargs):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    if loss_type == 'bce':
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    else:
        wfocal = FocalLossV1()(pred, mask)
        wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    if loss_type == 'bce':
        return (wbce + wiou).mean()
    else:
        return (wfocal + wiou).mean()

@MODELS.register_module()
class StructureLoss(nn.Module):
    def __init__(self,
                 loss_type='bce',
                 loss_weight=1.0, 
                 reduction='mean',
                 loss_name='loss_structure'):
        super(StructureLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.criterion = structure_loss
        self._loss_name = loss_name

    def forward(self, 
                pred, 
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if pred.size(1) == 1:
            assert target[target != ignore_index].max() <= 1, \
            'For pred with shape [N, 1, H, W], its label must have at ' \
            'most 2 classes'
            target = target.unsqueeze(1)

        loss = self.loss_weight * self.criterion(pred, target.float(), weight, reduction=reduction, avg_factor=avg_factor, loss_type=self.loss_type)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

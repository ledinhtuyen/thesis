import argparse
import logging
import os
import os.path as osp
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import albumentations as A

import hiera.hiera
from utils import clip_gradient, AvgMeter
from datetime import datetime
import json

import mmseg_custom
from mmengine.registry import init_default_scope
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmseg.registry import MODELS
init_default_scope('mmseg')

import hiera

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, prefix_path=None, img_size=384, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.prefix_path = prefix_path
        self.img_size = img_size
        self.t = A.Resize(img_size, img_size)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(osp.join(self.prefix_path, img_path))
        image = np.array(image)
        mask = cv2.imread(osp.join(self.prefix_path, mask_path), 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if image.shape != (self.img_size, self.img_size, 3):
            augmented = self.t(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)

epsilon = 1e-7

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)

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

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)        
    
    print("Macro scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))

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
    
  mean_precision = total_area_intersect / (total_pr + epsilon)
  mean_recall = total_area_intersect / (total_gt + epsilon)
  mean_iou = total_area_intersect / (total_area_union + epsilon)
  mean_dice = 2 * total_area_intersect / (total_pr + total_gt + epsilon)
  
  print("Micro scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))
  
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

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # print(inputs.size(), targets.size())
        #comment out if your model contains a sigmoid or equivalent activation layer
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets.float())
        
        inputs_ = torch.sigmoid(inputs)
        
        intersection = (inputs_ * targets).sum()      
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = torch.nansum(BCE) + torch.nansum(dice_loss)
        
        return Dice_BCE

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config', type=str,
                        default='', help='config file')
    parser.add_argument('--warmup_epochs', type=int,
                        default=2, help='epoch number')
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--accum_iter', type=int,
                        default=1, help='gradient accumulation steps')
    parser.add_argument('--test_batchsize', type=int,
                        default=64, help='test batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=384, help='training dataset size')
    parser.add_argument('--prefix_path', type=str,
                        default='/mnt/tuyenld/data/endoscopy/', help='prefix path')
    parser.add_argument('--num_workers', type=int,
                        default=16, help='test batch size')
    parser.add_argument('--seed', type=int,
                        default=2024, help='random seed')
    parser.add_argument('--freeze_backbone', action='store_true',
                        default=False, help='freeze the backbone params')
    parser.add_argument('--build_with_mmseg', action='store_true',
                        default=False, help='build with mmseg')
    parser.add_argument('--type_damage', type=str,
                        default='daday', help='type of damage')
    parser.add_argument('--clip', type=float,
                        default=1.0, help='gradient clipping margin')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        type=str,
        default="",
        help='whether to resume from the latest checkpoint')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    args = parser.parse_args()
    return args

def train(train_loader, 
          model, 
          optimizer, 
          epoch, 
          lr_scheduler, 
          save_path, 
          writer,
          args):
    print_log(f"Training on epoch {epoch}", logger=logging.getLogger())
    model.train()
    # ---- multi-scale training ----
    # size_rates = [0.75, 1, 1.25]
    # size_rates = [0.7, 1, 1.37]
    loss_record = AvgMeter()
    total_step = len(train_loader)
    criterion = structure_loss
    if args.amp:
        scaler = torch.cuda.amp.GradScaler(init_scale=2**13)
    with torch.autograd.set_detect_anomaly(True):
        start_time = datetime.now()
        optimizer.zero_grad()
        for i, pack in enumerate(tqdm(train_loader), start=1):
            if epoch <= args.warmup_epochs:
                optimizer.param_groups[0]["lr"] = args.init_lr * (i / total_step + epoch - 1) / args.warmup_epochs
            else:
                lr_scheduler.step()
            
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], (epoch-1) * total_step + i)

            # for rate in size_rates: 
                # optimizer.zero_grad()
                # ---- data prepare ----
            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()
            # ---- rescale ----
            # trainsize = int(round(args.init_trainsize*rate/32)*32)
            # images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            with torch.cuda.amp.autocast(enabled=args.amp):
                map4, map3, map2, map1 = model(images)
                # map1 = F.interpolate(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # map2 = F.interpolate(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # map3 = F.interpolate(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # map4 = F.interpolate(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                loss = criterion(map1, gts) + criterion(map2, gts) + criterion(map3, gts) + criterion(map4, gts)
            # ---- backward ----
            if args.amp:
                scaler.scale(loss).backward()
                if (i + 1) % args.accum_iter == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % args.accum_iter == 0:
                    clip_gradient(optimizer, args.clip)
                    optimizer.step()
                    optimizer.zero_grad()
            # ---- recording loss ----
            # if rate == 1:
            loss_record.update(loss.item(), args.batchsize)

        print('{} Training Epoch [{:03d}/{:03d}], '
                '[loss: {:0.4f}], time: {:4.2f}s'.
                format(datetime.now(), epoch, args.num_epochs,\
                        loss_record.show(), 
                        (datetime.now()-start_time).total_seconds()))

        writer.add_scalar('train/train_loss', loss_record.show(), epoch)
        writer.add_scalar('train/time', (datetime.now()-start_time).total_seconds(), epoch)

    # ckpt_path = save_path + f'/snapshots/{epoch}.pth'
    # print('[Saving Checkpoint:]', ckpt_path)
    # checkpoint = {
    #     'epoch': epoch + 1,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': lr_scheduler.state_dict()
    # }
    # torch.save(checkpoint, ckpt_path)
    
def test(test_dataloader, model, epoch, writer, args):
    print_log(f"Testing on epoch {epoch}", logger=logging.getLogger())
    model.eval()
    print_log(f"Testing on {args.type_damage}", logger=logging.getLogger())
    test_size = args.init_trainsize
    gt_, pr_ = [], []
    with torch.no_grad():
        for i, pack in enumerate(tqdm(test_dataloader), start=1):
            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()
            
            res, _, _, _ = model(images)
            res = F.interpolate(res, size=(test_size, test_size), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pr = res.round()
            gts = gts.data.cpu().squeeze()

            for gt, pr in zip(gts, pr):
                gt_.append(gt)
                pr_.append(pr)
        
        macro_iou_score, macro_dice_score, _, _ = get_macro_scores(gt_, pr_)
        micro_iou_score, micro_dice_score, _, _ = get_micro_scores(gt_, pr_)
        print('{} Testing Epoch [{:03d}/{:03d}], '
                '[name_dataset: {}, macro_dice: {:0.4f}, macro_iou: {:0.4f}, micro_dice: {:0.4f}, micro_iou: {:0.4f}]'.
                format(datetime.now(), epoch, args.num_epochs,\
                        args.type_damage, 
                        macro_dice_score,
                        macro_iou_score,
                        micro_dice_score,
                        micro_iou_score
                        ))

        writer.add_scalar(f'test_{args.type_damage}/mMacroDice', macro_dice_score, epoch)
        writer.add_scalar(f'test_{args.type_damage}/mMacroIoU', macro_iou_score, epoch)
        writer.add_scalar(f'test_{args.type_damage}/mMicroDice', micro_dice_score, epoch)
        writer.add_scalar(f'test_{args.type_damage}/mMicroIoU', micro_iou_score, epoch)

def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # load config
    if args.build_with_mmseg:
        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
            
        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])

        # resume training
        cfg.resume = args.resume
    
    # Create a new work_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.build_with_mmseg:
        save_path = osp.join(cfg.work_dir, timestamp)
    else:
        save_path = osp.join(args.work_dir, timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path + '/snapshots', exist_ok=True)
        os.makedirs(save_path + '/logs', exist_ok=True)
        
        # Init tensorboard
        writer = SummaryWriter(save_path + '/logs')
        
        # Save config file to work_dir
        if args.build_with_mmseg:
            cfg.dump(save_path + '/config.py')
    else:
        print("Save path existed")

    if args.type_damage != "polyp":
        data = json.load(open("/home/s/tuyenld/endoscopy/ft_ton_thuong.json"))
        data_damage = data[args.type_damage]
    else:
        data_damage = json.load(open("/home/s/tuyenld/endoscopy/polyp.json"))

    # Transform
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(),
        A.GaussianBlur(),
        # A.OneOf([
        #     A.RandomCrop(224, 224, p=1),
        #     A.CenterCrop(224, 224, p=1)
        # ], p=0.2),
        # A.Resize(args.init_trainsize, args.init_trainsize)
    ], p=0.5)

    # Build the dataloader
    train_img_paths = data_damage["train"]["images"]
    train_mask_paths = data_damage["train"]["masks"]
    
    train_dataset = Dataset(train_img_paths, train_mask_paths, img_size=args.init_trainsize, prefix_path=args.prefix_path, transform=train_transform)
    
    # Train with 10% of the dataset with no random choice and separate
    # train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, len(train_dataset), 10)))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    # Build validation dataloader
    val_img_paths = data_damage["test"]["images"]
    val_mask_paths = data_damage["test"]["masks"]
    val_dataset = Dataset(val_img_paths, val_mask_paths, img_size=args.init_trainsize, prefix_path=args.prefix_path)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batchsize,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    # Build the model
    if args.build_with_mmseg:
        model = MODELS.build(cfg.model).cuda()
    else:
        model = mmseg_custom.models.EncoderDecoderColonFormer(
            backbone=hiera.hiera_base_224(
                pretrained=True, checkpoint="mae_in1k_ft_in1k"
            ),
            decode_head=dict(
                type="UPerHead",
                in_channels=[96, 192, 384, 768],
                in_index=[0, 1, 2, 3],
                channels=128,
                dropout_ratio=0.1,
                num_classes=2,
                out_channels=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
                )
            ),
            build_with_mmseg=args.build_with_mmseg,
            in_channels=(192, 384, 768),
        ).cuda()
    
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                    T_max=len(train_loader)*args.num_epochs,
                                    eta_min=args.init_lr/1000)
        
    start_epoch = 1
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    print_log('Start running, epoch: %d' % start_epoch, logger=logging.getLogger())
    for epoch in range(start_epoch, args.num_epochs+1):
        train(train_loader, 
              model, 
              optimizer, 
              epoch, 
              lr_scheduler, 
              save_path, 
              writer, 
              args)
        test(val_loader, model, epoch, writer, args)

if __name__ == '__main__':
    main()

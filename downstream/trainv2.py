import argparse
import logging
import os
import os.path as osp

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

from utils import clip_gradient, AvgMeter
from datetime import datetime

import mmseg_custom
from mmengine.registry import init_default_scope
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmseg.registry import MODELS
init_default_scope('mmseg')

IMG_SIZE = (384, 384)

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform
        self.t = A.Compose([
            A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        if image.shape != (IMG_SIZE[0], IMG_SIZE[1], 3) or mask.shape != (IMG_SIZE[0], IMG_SIZE[1]):
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

def get_scores(gts, prs):
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
    
    print("scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))

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

def bce_dice_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = torch.sum(pred * mask)
    union = torch.sum(pred) + torch.sum(mask)
    dice = 1 - 2 * inter / union
    return bce + dice

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config', type=str,
                        default='', help='config file path')
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
    parser.add_argument('--build_with_mmseg', action='store_true',
                        default=False, help='whether the model is built with mmseg')
    parser.add_argument('--clip', type=float,
                        default=1.0, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
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
    size_rates = [0.7, 1, 1.37]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    total_step = len(train_loader)
    criterion = structure_loss
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    with torch.autograd.set_detect_anomaly(True):
        start_time = datetime.now()
        optimizer.zero_grad()
        for i, pack in enumerate(tqdm(train_loader), start=1):
            if epoch <= args.warmup_epochs:
                optimizer.param_groups[0]["lr"] = args.init_lr * (i / total_step + epoch - 1) / args.warmup_epochs
            else:
                lr_scheduler.step()
            
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], (epoch-1) * total_step + i)

            for rate in size_rates: 
                # optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = images.cuda()
                gts = gts.cuda()
                # ---- rescale ----
                trainsize = int(round(args.init_trainsize*rate/32)*32)
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16):
                    map4, map3, map2, map1 = model(images)
                    map1 = F.interpolate(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    map2 = F.interpolate(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    map3 = F.interpolate(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    map4 = F.interpolate(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    loss = criterion(map1, gts) + criterion(map2, gts) + criterion(map3, gts) + criterion(map4, gts)
                # ---- metrics ----
                dice_score = dice_m(map4, gts)
                iou_score = iou_m(map4, gts)
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
                if rate == 1:
                    loss_record.update(loss.item(), args.batchsize)
                    dice.update(dice_score.item(), args.batchsize)
                    iou.update(iou_score.item(), args.batchsize)

            # ---- train visualization ----
            if i == total_step:
                print('{} Training Epoch [{:03d}/{:03d}], '
                        '[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}], time: {:4.2f}s'.
                        format(datetime.now(), epoch, args.num_epochs,\
                                loss_record.show(), 
                                dice.show(), 
                                iou.show(),
                                (datetime.now()-start_time).total_seconds()))
                writer.add_scalar('train/train_loss', loss_record.show(), epoch)
                writer.add_scalar('train/train_mDice', dice.show(), epoch)
                writer.add_scalar('train/train_mIoU', iou.show(), epoch)
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
    
def test(test_dataloader_dict, model, epoch, writer, args):
    print_log(f"Testing on epoch {epoch}", logger=logging.getLogger())
    model.eval()
    for k in test_dataloader_dict:
        print_log(f"Testing on {k}", logger=logging.getLogger())
        test_size = args.init_trainsize
        gt_, pr_ = [], []
        with torch.no_grad():
            for i, pack in enumerate(tqdm(test_dataloader_dict[k]), start=1):
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
            
            iou_score, dice_score, precision, recall = get_scores(gt_, pr_)
            print('{} Testing Epoch [{:03d}/{:03d}], '
                    '[name_dataset: {}, dice: {:0.4f}, iou: {:0.4f}]'.
                    format(datetime.now(), epoch, args.num_epochs,\
                            k, 
                            dice_score,
                            iou_score))

            writer.add_scalar(f'test_{k}/mDice', dice_score, epoch)
            writer.add_scalar(f'test_{k}/mIoU', iou_score, epoch)

def main():
    args = parse_args()
    
    # load config
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
    save_path = osp.join(cfg.work_dir, timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path + '/snapshots', exist_ok=True)
        os.makedirs(save_path + '/logs', exist_ok=True)
        
        # Init tensorboard
        writer = SummaryWriter(save_path + '/logs')
        
        # Save config file to work_dir
        cfg.dump(save_path + '/config.py')
    else:
        print("Save path existed")

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        # A.D4(),
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(),
        A.GridDistortion(),
        A.MotionBlur(),
        A.GaussianBlur(),
        A.OneOf([
            A.RandomCrop(224, 224, p=1),
            A.CenterCrop(224, 224, p=1)
        ], p=0.2),
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    ], p=1.0)

    # Build the dataloader
    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob('{}/image/*'.format(args.train_path))
    train_mask_paths = glob('{}/masks/*'.format(args.train_path))
    train_img_paths.sort()
    train_mask_paths.sort()

    train_dataset = Dataset(train_img_paths, train_mask_paths, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    test_path = dict(
        Kvasir="/workspace/DATA2/public_dataset/TestDataset/Kvasir",
        CVC_ClinicDB="/workspace/DATA2/public_dataset/TestDataset/CVC-ClinicDB",
        CVC_ColonDB="/workspace/DATA2/public_dataset/TestDataset/CVC-ColonDB",
        CVC_T="/workspace/DATA2/public_dataset/TestDataset/CVC-300",
        ETIS_Larib="/workspace/DATA2/public_dataset/TestDataset/ETIS-LaribPolypDB",
    )
    
    # Build validation dataloader
    test_dataloader_dict = {}
    for k, v in test_path.items():
        test_img_paths = glob('{}/images/*'.format(v))
        test_mask_paths = glob('{}/masks/*'.format(v))
        test_img_paths.sort()
        test_mask_paths.sort()
        test_dataset = Dataset(test_img_paths, test_mask_paths)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batchsize,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        test_dataloader_dict[k] = test_loader

    # Build the model
    if args.build_with_mmseg:
        model = MODELS.build(cfg.model).cuda()
    else:
        import hiera
        backbone = hiera.hiera_base_224()
        backbone.load_state_dict(torch.load('/workspace/hiera/examples/runs/state_dict.pth'))
        # backbone = hiera.hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
        model = mmseg_custom.models.EncoderDecoderRaBiT(
            backbone=backbone,
            # decode_head=dict(
            #     type="UPerHead",
            #     in_channels=[96, 192, 384, 768],
            #     in_index=[0, 1, 2, 3],
            #     channels=128,
            #     dropout_ratio=0.1,
            #     num_classes=2,
            #     out_channels=1,
            #     norm_cfg=dict(type='BN', requires_grad=True),
            #     align_corners=False,
            #     loss_decode=dict(
            #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
            #     )
            # ),
            build_with_mmseg=False,
            in_channels=(192, 384, 768),
        ).cuda()
    
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                    T_max=len(train_loader)*args.num_epochs,
                                    eta_min=args.init_lr/1000)
        
    start_epoch = 1
    if args.resume != "":
        print_log('Resuming from checkpoint: %s' % args.resume, logger=logging.getLogger())
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
        test(test_dataloader_dict, model, epoch, writer, args)

if __name__ == '__main__':
    main()

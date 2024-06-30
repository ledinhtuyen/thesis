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

from utils import clip_gradient, AvgMeter, structure_loss, dice_m, iou_m, get_macro_scores
from datetime import datetime

import mmseg_custom
from mmengine.registry import init_default_scope
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmseg.registry import MODELS
init_default_scope('mmseg')

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, img_size = 384, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = transform
        self.t = A.Compose([
            A.Resize(img_size, img_size),
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config', type=str,
                        default='', help='config file path')
    parser.add_argument('--warmup_epochs', type=int,
                        default=1, help='epoch number')
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
    parser.add_argument('--num_workers', type=int,
                        default=16, help='test batch size')
    parser.add_argument('--seed', type=int,
                        default=2024, help='random seed')
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

    ckpt_path = save_path + f'/snapshots/{epoch}.pth'
    print('[Saving Checkpoint:]', ckpt_path)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)
    
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
                pr = res.round()
                gts = gts.data.cpu().squeeze()

                for gt, pr in zip(gts, pr):
                    gt_.append(gt)
                    pr_.append(pr)
            
            iou_score, dice_score, precision, recall = get_macro_scores(gt_, pr_)
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
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
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
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(),
        A.GaussianBlur(),
        A.OneOf([
            A.RandomCrop(224, 224, p=1),
            A.CenterCrop(224, 224, p=1)
        ], p=0.2),
        A.Resize(args.init_trainsize, args.init_trainsize),
    ], p=0.5)

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
    
    benchmark_dir = '/'.join(args.train_path.split('/')[0:-1])
    
    test_path = dict(
        Kvasir=f"{benchmark_dir}/TestDataset/Kvasir",
        CVC_ClinicDB=f"{benchmark_dir}/TestDataset/CVC-ClinicDB",
        CVC_ColonDB=f"{benchmark_dir}/TestDataset/CVC-ColonDB",
        CVC_T=f"{benchmark_dir}/TestDataset/CVC-300",
        ETIS_Larib=f"{benchmark_dir}/TestDataset/ETIS-LaribPolypDB",
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
    model = MODELS.build(cfg.model).cuda()
    
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
        train(train_loader, model, optimizer, epoch, lr_scheduler, save_path, writer, args)
        test(test_dataloader_dict, model, epoch, writer, args)

if __name__ == '__main__':
    main()

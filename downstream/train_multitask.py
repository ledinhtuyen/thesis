import argparse
import logging
import os
import os.path as osp
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy, ConfusionMatrix

import albumentations as A

from utils import clip_gradient, MetricMeter, plot_confusion_matrix
from datetime import datetime

import mmseg_custom
import multitask
from multitask.multidataset import Data, MultiDataset
from multitask.structure_loss import structure_loss
from utils import dice_m, iou_m, get_macro_scores, get_micro_scores

from mmengine.registry import init_default_scope
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmseg.registry import MODELS
init_default_scope('mmseg')

def segment_loss(pred, mask, type):
    pred_ = pred[(type >= 1) & (type <= 6)]
    mask_ = mask[(type >= 1) & (type <= 6)]
    return torch.nansum(structure_loss(pred_, mask_))
  
def cls_loss(pred, label, type=None):
    if type == None: # type classification
      pred_ = pred
      label_ = label
    else: # position classification
      pred_ = pred[type == 0]
      label_ = label[type == 0]
    return torch.nansum(nn.CrossEntropyLoss()(pred_, label_))

def bi_cls_loss(pred, label, type):
    # hp classification
    pred_ = torch.flatten(pred[type == 5])
    label_ = label[type == 5].float()
    return torch.nansum(nn.BCEWithLogitsLoss()(pred_, label_))

class CalcMetric:
    def __init__(self, writer) -> None:
        self.pred = {}
        self.gts = {}
        self.writer = writer
        
    def add(self, output, masks, cls_label, type):
      for i in range(8):
        if i == 0:
          pred = output["pos"][type == 0].cpu()
          label = cls_label[type == 0]
        elif i >= 1 and i <= 6 and i != 5:
          pred = output["map"][0][type == i].cpu().sigmoid().round()
          label = masks[type == i]
        elif i == 5:
            pred = output["map"][0][type == 5].cpu().sigmoid().round()
            label = masks[type == 5]

            pred_cls = torch.flatten(output["hp"][type == 5]).cpu().sigmoid()
            label_cls = cls_label[type == 5]
            if "hp" in self.pred:
                self.pred["hp"] = torch.cat((self.pred["hp"], pred_cls)) # hp
                self.gts["hp"] = torch.cat((self.gts["hp"], label_cls)) # hp
            else:
                self.pred["hp"] = pred_cls
                self.gts["hp"] = label_cls
        elif i == 7:
          pred = output["type"].cpu()
          label = type
        
        if i in self.pred:
          self.pred[i] = torch.cat((self.pred[i], pred))
          self.gts[i] = torch.cat((self.gts[i], label))
        else:
          self.pred[i] = pred
          self.gts[i] = label
    
    def calc_cls_metric(self, type, epoch, save_path):
        if type == 0:
            class_names = ["Hầu họng", "Thực quản", "Tam vị", "Thân vị", "Phình vị", "Hang vị", "Bờ cong lớn", "Bờ cong nhỏ", "Hành tá tràng", "Tá tràng"]
            macro_acc = Accuracy(task="multiclass", num_classes=10, average="macro")
            micro_acc = Accuracy(task="multiclass", num_classes=10, average="micro")
            confmat = ConfusionMatrix(task="multiclass", num_classes=10)
            pred = self.pred[type]
            gts = self.gts[type]
        elif type == 5:
            class_names = ["Lành tính", "Ác tính"]
            macro_acc = Accuracy(task="binary", num_classes=2, average="macro")
            micro_acc = Accuracy(task="binary", num_classes=2, average="micro")
            confmat = ConfusionMatrix(task="binary", num_classes=2)
            pred = self.pred["hp"]
            gts = self.gts["hp"]
        elif type == 7:
            class_names = ["VTGP", "UTTQ", "VTQ", "VLHTT", "UTDD", "VDD/HP", "POLYP"]
            macro_acc = Accuracy(task="multiclass", num_classes=7, average="macro")
            micro_acc = Accuracy(task="multiclass", num_classes=7, average="micro")
            confmat = ConfusionMatrix(task="multiclass", num_classes=7)
            pred = self.pred[type]
            gts = self.gts[type]

        macro_acc_ = macro_acc(pred, gts)
        micro_acc_ = micro_acc(pred, gts)
        confusion_matrix = confmat(pred, gts)
        conf_img = plot_confusion_matrix(confusion_matrix, class_names, normalize=True)
        
        # Save image
        cv2.imwrite(f'{save_path}/imgs/cls_{type}_confusion_matrix_{epoch}.png', conf_img)

        self.writer.add_scalar(f'cls_{type}/macro_acc', macro_acc_, epoch)
        self.writer.add_scalar(f'cls_{type}/micro_acc', micro_acc_, epoch)
        self.writer.add_image(f'cls_{type}/confusion_matrix', conf_img, epoch, dataformats='HWC')

    def calc_seg_metric(self, type, epoch):
        self.pred[type] = self.pred[type].data.squeeze()
        self.gts[type] = self.gts[type].data.squeeze()

        macro_iou, macro_dice, _, _ = get_macro_scores(self.pred[type], self.gts[type])
        micro_io, micro_dice, _, _ = get_micro_scores(self.pred[type], self.gts[type])
        
        self.writer.add_scalar(f'seg_{type}/macro_dice', macro_dice, epoch)
        self.writer.add_scalar(f'seg_{type}/macro_iou', macro_iou, epoch)
        self.writer.add_scalar(f'seg_{type}/micro_dice', micro_dice, epoch)
        self.writer.add_scalar(f'seg_{type}/micro_iou', micro_io, epoch)

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', help='train config file path')
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
    parser.add_argument('--num_workers', type=int,
                        default=16, help='test batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=384, help='training dataset size')
    parser.add_argument('--metadata_file', type=str,
                        default='metadata.json', help='metadata file')
    parser.add_argument('--prefix_path', type=str,
                        default='/mnt/tuyenld/data/endoscopy/', help='prefix path')
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
    met = MetricMeter()
    met.add_meter(['all_loss', 'seg_loss', 'hp_loss', 'pos_loss', 'type_loss'])
    total_step = len(train_loader)
    if args.amp:
        scaler = torch.cuda.amp.GradScaler(init_scale=2**14, enabled=args.amp)
    with torch.autograd.set_detect_anomaly(True):
        start_time = datetime.now()
        optimizer.zero_grad()
        for i, pack in enumerate(tqdm(train_loader), start=1):
            if epoch <= args.warmup_epochs:
                optimizer.param_groups[0]["lr"] = args.init_lr * (i / total_step + epoch - 1) / args.warmup_epochs
            else:
                lr_scheduler.step()
            
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], (epoch-1) * total_step + i)

            # ---- data prepare ----
            type, images, masks, cls_labels = pack
            type = type.cuda()
            images = images.cuda()
            masks = masks.cuda()
            cls_labels = cls_labels.cuda()
            # ---- forward ----
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)
                seg_loss = segment_loss(output["map"][0], masks, type)
                hp_loss = bi_cls_loss(output["hp"], cls_labels, type)
                pos_loss = cls_loss(output["pos"], cls_labels, type)
                type_loss = cls_loss(output["type"], type)
                loss = seg_loss + hp_loss + pos_loss + type_loss
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
            met.update({
                'all_loss': loss.data,
                'seg_loss': seg_loss.data,
                'hp_loss': hp_loss.data,
                'pos_loss': pos_loss.data,
                'type_loss': type_loss.data
            })

        print('{} Training Epoch [{:03d}/{:03d}], '
                '[{}], time: {:4.2f}s'.
                format(datetime.now(), epoch, args.num_epochs,
                        str(met),
                        (datetime.now()-start_time).total_seconds()))
        writer.add_scalar('train/all_loss', met.get_meter('all_loss').show(), epoch)
        writer.add_scalar('train/seg_loss', met.get_meter('seg_loss').show(), epoch)
        writer.add_scalar('train/hp_loss', met.get_meter('hp_loss').show(), epoch)
        writer.add_scalar('train/pos_loss', met.get_meter('pos_loss').show(), epoch)
        writer.add_scalar('train/type_loss', met.get_meter('type_loss').show(), epoch)
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

def test(test_dataloader, model, epoch, save_path, writer, args):
    print_log(f"Testing on epoch {epoch}", logger=logging.getLogger())
    model.eval()
    met = CalcMetric(writer)
    with torch.no_grad():
        for i, pack in enumerate(tqdm(test_dataloader), start=1):
            type, images, masks, cls_label = pack
            images = images.cuda()

            output = model(images)
            
            met.add(output, masks, cls_label, type)
        
        for i in range(8):
          if i == 0 or i == 7:
            met.calc_cls_metric(i, epoch, save_path)
          elif i == 5:
            met.calc_cls_metric(i, epoch, save_path)
            met.calc_seg_metric(i, epoch)
          else:
            met.calc_seg_metric(i, epoch)

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
        os.makedirs(save_path + '/imgs', exist_ok=True)
        
        # Init tensorboard
        writer = SummaryWriter(save_path + '/logs')
        
        # Save config file to work_dir
        cfg.dump(save_path + '/config.py')
    else:
        print("Save path existed")
        
    # Train data augmentation
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        # A.D4(),
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(),
        A.GaussianBlur(),
        A.OneOf([
            A.RandomCrop(224, 224, p=1),
            A.CenterCrop(224, 224, p=1)
        ], p=0.2),
        A.Resize(args.init_trainsize, args.init_trainsize)
    ], p=1.0)

    # Build the data
    data = Data(args.metadata_file)
    
    train_dataset = MultiDataset(data.train_samples, args.prefix_path, 
                                 img_size=args.init_trainsize, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    # Build validation dataloader
    val_dataset = MultiDataset(data.val_samples, args.prefix_path, img_size=args.init_trainsize)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batchsize,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    # Build the model
    model = MODELS.build(cfg.model).cuda()
    
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
        test(val_loader, model, epoch, save_path, writer, args)

if __name__ == '__main__':
    main()

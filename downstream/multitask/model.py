import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg_custom.models.segmentors.include.conv_layer import Conv
from mmseg_custom.models.segmentors.utils import BiRAFPN

from mmseg.registry import MODELS

@MODELS.register_module()
class MultiTask(nn.Module):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 compound_coef=4,
                 numrepeat = 4,
                 ):
        super(MultiTask, self).__init__()
        
        self.backbone = MODELS.build(backbone)
        self.backbone.init_weights(pretrained=pretrained)
        
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.numrepeat = numrepeat + 1
        self.compound_coef = compound_coef
        self.conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],#448
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }
        
        self.head_seg = self.build_head_segment()
        self.head_cls_hp = self.build_head_cls(1)
        self.head_cls_positon = self.build_head_cls(10)
        self.head_type = self.build_head_cls(8)

    def build_head_segment(self):
        head_segment = nn.ModuleDict()
        
        head_segment["bifpn"] = nn.Sequential(
                                  *[BiRAFPN(self.fpn_num_filters[self.compound_coef],
                                          self.conv_channel_coef[self.compound_coef],
                                          True if _ == 0 else False,
                                          attention=True ,
                                          use_p8=self.compound_coef > 7)
                                    for _ in range(self.numrepeat)]
                                )

        head_segment["conv1"] = Conv(768,self.conv_channel_coef[self.compound_coef][0],1,1,padding=0,bn_acti=True)
        head_segment["conv2"] = Conv(768,self.conv_channel_coef[self.compound_coef][1],1,1,padding=0,bn_acti=True)
        head_segment["conv3"] = Conv(768,self.conv_channel_coef[self.compound_coef][2],1,1,padding=0,bn_acti=True)
        head_segment["head1"] = Conv(self.fpn_num_filters[self.compound_coef],1,1,1,padding=0,bn_acti=False)
        head_segment["head2"] = Conv(self.fpn_num_filters[self.compound_coef],1,1,1,padding=0,bn_acti=False)
        head_segment["head3"] = Conv(self.fpn_num_filters[self.compound_coef],1,1,1,padding=0,bn_acti=False)

        return head_segment
    
    def build_head_cls(self, num_classes):
        head_cls = nn.Sequential(
                          nn.Linear(768, 512),
                          nn.ReLU(),
                          nn.Linear(512, num_classes)
                        )
        return head_cls

    def forward_segment(self, head, x):
        x1 = x[0] # 1/4
        x2 = x[1] # 1/8
        x3 = x[2] # 1/16
        x4 = x[3] # 1/32
        
        x2 = head["conv1"](x2)
        x3 = head["conv2"](x3)
        x4 = head["conv3"](x4)
        p3,p4,p5,p6,p7 = head["bifpn"]([x2,x3,x4])
        p3 = head["head3"](p3)
        p4 = head["head2"](p4)
        p5 = head["head1"](p5)
        
        lateral_map_2 = F.interpolate(p5,scale_factor=32,mode='bilinear')
        lateral_map_5 = F.interpolate(p3,scale_factor=8,mode='bilinear') 
        lateral_map_3 = F.interpolate(p4,scale_factor=16,mode='bilinear') 
        lateral_map_1 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1
      
    def forward_cls(self, head, x):
        x = x[0]
        x = x.mean([2,3])
        return head(x)

    def forward(self, inputs):
        x = self.backbone(inputs)
        
        if hasattr(self, 'neck'):
            x = self.neck(x)
        
        # Forward segment
        map = self.forward_segment(self.head_seg, x)
        
        # Forward cls hp
        hp = self.forward_cls(self.head_cls_hp, x)
        
        # Forward cls position
        position = self.forward_cls(self.head_cls_positon, x)
        
        # # Forward cls type
        type = self.forward_cls(self.head_type, x)
        
        return {"type" : type, "map" : map, "hp" : hp, "pos" : position}

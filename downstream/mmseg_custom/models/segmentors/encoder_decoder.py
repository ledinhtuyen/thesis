import torch
import torch.nn as nn
import torch.nn.functional as F

from .include.conv_layer import Conv
from .include.axial_atten import AA_kernel
from .include.context_module import CFPModule
from .utils import BiRAFPN

from mmseg.registry import MODELS
        
@MODELS.register_module()
class EncoderDecoderRaBiT(nn.Module):
    def __init__(self,
                 backbone,
                 decode_head=None,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 compound_coef=4,
                 numrepeat = 4,
                 bottleneck=True,
                 in_channels=[768, 768, 768]
                 ):
        super(EncoderDecoderRaBiT, self).__init__()
        
        self.backbone = MODELS.build(backbone)
        self.backbone.init_weights(pretrained=pretrained)
        
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.numrepeat = numrepeat + 1
        self.compound_coef = compound_coef
        conv_channel_coef = {
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

        self.bifpn = nn.Sequential(
            *[BiRAFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True ,
                    use_p8=compound_coef > 7,bottleneck=bottleneck)
              for _ in range(self.numrepeat)])

        self.conv1 = Conv(in_channels[0],conv_channel_coef[compound_coef][0],1,1,padding=0,bn_acti=True)#128
        self.conv2 = Conv(in_channels[1],conv_channel_coef[compound_coef][1],1,1,padding=0,bn_acti=True)#320
        self.conv3 = Conv(in_channels[2],conv_channel_coef[compound_coef][2],1,1,padding=0,bn_acti=True)#512
        self.head1 = Conv(self.fpn_num_filters[compound_coef],1,1,1,padding=0,bn_acti=False)
        self.head2 = Conv(self.fpn_num_filters[compound_coef],1,1,1,padding=0,bn_acti=False)
        self.head3 = Conv(self.fpn_num_filters[compound_coef],1,1,1,padding=0,bn_acti=False)
    
    def forward(self, inputs):
        x = self.backbone(inputs)
        
        if hasattr(self, 'neck'):
            x = self.neck(x)
        
        x1 = x[0] # 1/4
        x2 = x[1] # 1/8
        x3 = x[2] # 1/16
        x4 = x[3] # 1/32

        x2 = self.conv1(x2)
        x3 = self.conv2(x3)
        x4 = self.conv3(x4)
        p3, p4, p5, _, _ = self.bifpn([x2,x3,x4])
        p3 = self.head3(p3)
        p4 = self.head2(p4)
        p5 = self.head1(p5)
        lateral_map_2 = F.interpolate(p5,scale_factor=32,mode='bilinear')
        lateral_map_5 = F.interpolate(p3,scale_factor=8,mode='bilinear') 
        lateral_map_3 = F.interpolate(p4,scale_factor=16,mode='bilinear') 
        lateral_map_1 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1

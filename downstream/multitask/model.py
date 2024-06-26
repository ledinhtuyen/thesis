import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg_custom.models.segmentors.include.conv_layer import Conv
from mmseg_custom.models.segmentors.utils import BiRAFPN
from mmseg_custom.models.segmentors.include.axial_atten import AA_kernel
from mmseg_custom.models.segmentors.include.context_module import CFPModule

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
                 in_channels=(192, 384, 768),
                 build_with_mmseg=True
                 ):
        super(MultiTask, self).__init__()
        self.in_channels = in_channels
        self.build_with_mmseg = build_with_mmseg
        
        if self.build_with_mmseg:
            self.backbone = MODELS.build(backbone)
            self.backbone.init_weights(pretrained=pretrained)
        else:
            self.backbone = backbone
        
        if neck is not None:
            self.neck = MODELS.build(neck)
            
        if decode_head is not None:
            self.decode_head = MODELS.build(decode_head)
            self.decode_head.init_weights()
        
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
        
        self.head_seg = self.build_head_segment_colonformer()
        self.head_cls_hp = self.build_head_cls(1)
        self.head_cls_positon = self.build_head_cls(10)
        self.head_type = self.build_head_cls(7)
        
    def build_head_segment_colonformer(self):
        head_segment = nn.ModuleDict()
        
        head_segment["CFP_1"] = CFPModule(self.in_channels[0], d = 8)
        head_segment["CFP_2"] = CFPModule(self.in_channels[1], d = 8)
        head_segment["CFP_3"] = CFPModule(self.in_channels[2], d = 8)
        ###### dilation rate 4, 62.8

        head_segment["ra1_conv1"] = Conv(self.in_channels[0],32,3,1,padding=1,bn_acti=True)
        head_segment["ra1_conv2"] = Conv(32,32,3,1,padding=1,bn_acti=True)
        head_segment["ra1_conv3"] = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        head_segment["ra2_conv1"] = Conv(self.in_channels[1],32,3,1,padding=1,bn_acti=True)
        head_segment["ra2_conv2"] = Conv(32,32,3,1,padding=1,bn_acti=True)
        head_segment["ra2_conv3"] = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        head_segment["ra3_conv1"] = Conv(self.in_channels[2],32,3,1,padding=1,bn_acti=True)
        head_segment["ra3_conv2"] = Conv(32,32,3,1,padding=1,bn_acti=True)
        head_segment["ra3_conv3"] = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        head_segment["aa_kernel_1"] = AA_kernel(self.in_channels[0],self.in_channels[0])
        head_segment["aa_kernel_2"] = AA_kernel(self.in_channels[1],self.in_channels[1])
        head_segment["aa_kernel_3"] = AA_kernel(self.in_channels[2],self.in_channels[2])
        
        return head_segment

    def build_head_segment_rabit(self):
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
    
    def forward_segment_colonformer(self, head, x):
        x1 = x[0] # 1/4
        x2 = x[1] # 1/8
        x3 = x[2] # 1/16
        x4 = x[3] # 1/32
        
        decoder_out = self.decode_head(x)
        
        decoder_1 = decoder_out
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = head["CFP_3"](x4)
        # cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = head["aa_kernel_3"](cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, self.in_channels[2], -1, -1).mul(aa_atten_3)
        
        ra_3 = head["ra3_conv1"](aa_atten_3_o) 
        ra_3 = head["ra3_conv2"](ra_3) 
        ra_3 = head["ra3_conv3"](ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = head["CFP_2"](x3)
        # cfp_out_2 += x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = head["aa_kernel_2"](cfp_out_2)
        aa_atten_2 += cfp_out_2
        aa_atten_2_o = decoder_3_ra.expand(-1, self.in_channels[1], -1, -1).mul(aa_atten_2)
        
        ra_2 = head["ra2_conv1"](aa_atten_2_o) 
        ra_2 = head["ra2_conv2"](ra_2) 
        ra_2 = head["ra2_conv3"](ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = head["CFP_1"](x2)
        # cfp_out_3 += x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = head["aa_kernel_1"](cfp_out_3)
        aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_4_ra.expand(-1, self.in_channels[0], -1, -1).mul(aa_atten_1)
        
        ra_1 = head["ra1_conv1"](aa_atten_1_o) 
        ra_1 = head["ra1_conv2"](ra_1) 
        ra_1 = head["ra1_conv3"](ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1

    def forward_segment_rabit(self, head, x):
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
        x = x[3]
        x = x.mean([2,3])
        return head(x)

    def forward(self, inputs):
        if self.build_with_mmseg:
            x = self.backbone(inputs)
        else:
            x = self.backbone(inputs, return_intermediates=True)[1]
        
        if hasattr(self, 'neck'):
            x = self.neck(x)
        
        if not self.build_with_mmseg:
            x1 = x[0].permute(0, 3, 1, 2).contiguous() # 1/4
            x2 = x[1].permute(0, 3, 1, 2).contiguous() # 1/8
            x3 = x[2].permute(0, 3, 1, 2).contiguous() # 1/16
            x4 = x[3].permute(0, 3, 1, 2).contiguous() # 1/32
            x = [x1, x2, x3, x4]
        
        # Forward segment
        map = self.forward_segment_colonformer(self.head_seg, x)
        
        # Forward cls hp
        hp = self.forward_cls(self.head_cls_hp, x)
        
        # Forward cls position
        position = self.forward_cls(self.head_cls_positon, x)
        
        # # Forward cls type
        type = self.forward_cls(self.head_type, x)
        
        return {"type" : type, "map" : map, "hp" : hp, "pos" : position}

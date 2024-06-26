import torch
import torch.nn as nn
import torch.nn.functional as F

from .include.conv_layer import Conv
from .include.axial_atten import AA_kernel
from .include.context_module import CFPModule
from .utils import BiRAFPN

from mmseg.registry import MODELS

@MODELS.register_module()
class EncoderDecoderColonFormer(nn.Module):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 build_with_mmseg=True,
                 in_channels=[768, 768, 768]):
        super(EncoderDecoderColonFormer, self).__init__()
        self.build_with_mmseg = build_with_mmseg
        self.in_channels = in_channels
        
        if self.build_with_mmseg:
            self.backbone = MODELS.build(backbone)
            self.backbone.init_weights(pretrained=pretrained)
        else:
            self.backbone = backbone

        if neck is not None:
            self.neck = MODELS.build(neck)

        self.decode_head = MODELS.build(decode_head)
        self.decode_head.init_weights()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.CFP_1 = CFPModule(in_channels[0], d = 8)
        self.CFP_2 = CFPModule(in_channels[1], d = 8)
        self.CFP_3 = CFPModule(in_channels[2], d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(in_channels[0],32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(in_channels[1],32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(in_channels[2],32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(in_channels[0],in_channels[0])
        self.aa_kernel_2 = AA_kernel(in_channels[1],in_channels[1])
        self.aa_kernel_3 = AA_kernel(in_channels[2],in_channels[2])

    def forward(self, inputs):
        if self.build_with_mmseg:
            x = self.backbone(inputs)
        else:
            x = self.backbone(inputs, return_intermediates=True)[1]
        
        if hasattr(self, 'neck'):
            x = self.neck(x)

        if self.build_with_mmseg:
            x1 = x[0] # 1/4
            x2 = x[1] # 1/8
            x3 = x[2] # 1/16
            x4 = x[3] # 1/32
        else:
            x1 = x[0].permute(0, 3, 1, 2).contiguous() # 1/4
            x2 = x[1].permute(0, 3, 1, 2).contiguous() # 1/8
            x3 = x[2].permute(0, 3, 1, 2).contiguous() # 1/16
            x4 = x[3].permute(0, 3, 1, 2).contiguous() # 1/32
            x = [x1, x2, x3, x4]

        decoder_out = self.decode_head(x)
        
        decoder_1 = decoder_out
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4)
        # cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, self.in_channels[2], -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3)
        # cfp_out_2 += x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2 += cfp_out_2
        aa_atten_2_o = decoder_3_ra.expand(-1, self.in_channels[1], -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2)
        # cfp_out_3 += x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_4_ra.expand(-1, self.in_channels[0], -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1

@MODELS.register_module()
class EncoderDecoderCustom(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoderCustom, self).__init__()
        self.backbone = MODELS.build(backbone)
        self.decode_head = MODELS.build(decode_head)
        self.backbone.init_weights(pretrained=pretrained)
        
    def forward(self, inputs):
        x = self.backbone(inputs)
        decoder_out = self.decode_head(x)
        return decoder_out
        
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
                 build_with_mmseg=True,
                 in_channels=[768, 768, 768]
                 ):
        super(EncoderDecoderRaBiT, self).__init__()
        self.build_with_mmseg = build_with_mmseg
        
        if self.build_with_mmseg:
            self.backbone = MODELS.build(backbone)
            self.backbone.init_weights(pretrained=pretrained)
        else:
            self.backbone = backbone
        
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
        if self.build_with_mmseg:
            x = self.backbone(inputs)
        else:
            x = self.backbone(inputs, return_intermediates=True)[1]
        
        if hasattr(self, 'neck'):
            x = self.neck(x)
        
        if self.build_with_mmseg:
            x1 = x[0] # 1/4
            x2 = x[1] # 1/8
            x3 = x[2] # 1/16
            x4 = x[3] # 1/32
        else:
            x1 = x[0].permute(0, 3, 1, 2).contiguous() # 1/4
            x2 = x[1].permute(0, 3, 1, 2).contiguous() # 1/8
            x3 = x[2].permute(0, 3, 1, 2).contiguous() # 1/16
            x4 = x[3].permute(0, 3, 1, 2).contiguous() # 1/32

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

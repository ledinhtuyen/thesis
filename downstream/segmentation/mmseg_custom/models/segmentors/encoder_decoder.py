import torch
import torch.nn as nn
import torch.nn.functional as F

from .include.conv_layer import Conv
from .include.axial_atten import AA_kernel
from .include.context_module import CFPModule

from mmseg.registry import MODELS

@MODELS.register_module()
class EncoderDecoderV2(nn.Module):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoderV2, self).__init__()

        self.backbone = MODELS.build(backbone)
        self.backbone.init_weights(pretrained=pretrained)
        
        if neck is not None:
            self.neck = MODELS.build(neck)

        self.decode_head = MODELS.build(decode_head)
        self.decode_head.init_weights()
        
        if auxiliary_head is not None:
            self.auxiliary_head = MODELS.build(auxiliary_head)
            self.auxiliary_head.init_weights()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.CFP_1 = CFPModule(768, d = 8)
        self.CFP_2 = CFPModule(768, d = 8)
        self.CFP_3 = CFPModule(768, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(768,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(768,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(768,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(768,768)
        self.aa_kernel_2 = AA_kernel(768,768)
        self.aa_kernel_3 = AA_kernel(768,768)

    def forward(self, inputs):
        x = self.backbone(inputs)
        
        if hasattr(self, 'neck'):
            x = self.neck(x)

        x1 = x[0] # 1/4
        x2 = x[1] # 1/8
        x3 = x[2] # 1/16
        x4 = x[3] # 1/32

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
        aa_atten_3_o = decoder_2_ra.expand(-1, 768, -1, -1).mul(aa_atten_3)
        
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
        aa_atten_2_o = decoder_3_ra.expand(-1, 768, -1, -1).mul(aa_atten_2)
        
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
        aa_atten_1_o = decoder_4_ra.expand(-1, 768, -1, -1).mul(aa_atten_1)
        
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

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = MODELS.build(decode_head)

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()

        self.CFP_1 = CFPModule(192, d = 8)
        self.CFP_2 = CFPModule(384, d = 8)
        self.CFP_3 = CFPModule(768, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(192,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(384,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(768,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(192,192)
        self.aa_kernel_2 = AA_kernel(384,384)
        self.aa_kernel_3 = AA_kernel(768,768)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  96x88x88
        x2 = segout[1]  # 192x44x44
        x3 = segout[2]  # 384x22x22
        x4 = segout[3]  # 768x11x11

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4)
        # cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, 768, -1, -1).mul(aa_atten_3)
        
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
        aa_atten_2_o = decoder_3_ra.expand(-1, 384, -1, -1).mul(aa_atten_2)
        
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
        aa_atten_1_o = decoder_4_ra.expand(-1, 192, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1

import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from . import losses as L
import torchvision.ops as ops
import numpy as np
from . import pwc_net


# SelfDZSR++ Model
class DZSRModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(DZSRModel, self).__init__(opt)

        self.opt = opt

        self.visual_names = ['data_lr', 'data_hr', 'data_sr']  
        self.loss_names = ['RefSR_L1', 'RefSR_SWD', 'RefSR_Total', 'KernelGen_L1', 'KernelGen_Filter', 'KernelGen_Total'] 

        self.model_names = ['RefSR', 'KernelGen']  
        self.optimizer_names = ['RefSR_optimizer_%s' % opt.optimizer, 'KernelGen_optimizer_%s' % opt.optimizer] #

        RefSR = SelfRefSR(opt)
        self.netRefSR = N.init_net(RefSR, opt.init_type, opt.init_gain, opt.gpu_ids)

        kernelgen = N.KernelGen(opt)
        self.netKernelGen= N.init_net(kernelgen, opt.init_type, opt.init_gain, opt.gpu_ids)

        student = N.ContrasExtractorSep()
        self.netStudent = N.init_net(student, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.set_requires_grad(self.netStudent, requires_grad=False)

        if opt.camera == 'nikon':
            pwcnet = pwc_net.PWCNET()
            self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.set_requires_grad(self.netPWCNET, requires_grad=False)
            self.load_network_path(self.netStudent, './ckpt/nikon_pretrain_models/Student_model_400.pth')
        elif opt.camera == 'iphone':
            self.load_network_path(self.netStudent, './ckpt/iphone_pretrain_models/Student_model_400.pth')

        if self.isTrain:
            if opt.camera == 'nikon':
                self.load_network_path(self.netKernelGen, './ckpt/nikon_pretrain_models/KernelGen_model_400.pth')
            elif opt.camera == 'iphone':
                self.load_network_path(self.netKernelGen, './ckpt/iphone_pretrain_models/KernelGen_model_400.pth')

            self.optimizer_RefSR = optim.Adam(self.netRefSR.parameters(),
                                          lr=opt.lr,
                                          betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.weight_decay)

            self.optimizer_KernelGen = optim.Adam(self.netKernelGen.parameters(),
                                lr=opt.lr/2,
                                betas=(opt.beta1, opt.beta2),
                                weight_decay=opt.weight_decay)

            self.optimizers = [self.optimizer_RefSR, self.optimizer_KernelGen] #

            self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
            self.criterionSWD = N.init_net(L.SWDLoss(), gpu_ids=opt.gpu_ids)
            self.criterionFilter = N.init_net(L.FilterLoss(), gpu_ids=opt.gpu_ids)

    def set_input(self, input):
        self.data_lr = input['lr'].to(self.device)
        self.data_hr = input['hr'].to(self.device)
        self.data_ref2x_lr = input['ref2x_lr'].to(self.device)
        self.data_ref4x = input['ref4x'].to(self.device)
        self.data_noise = input['noise'].to(self.device)
        self.coord_ref4x = input['coord_ref4x'].to(self.device)
        self.image_name = input['fname']

    def forward(self):
        if not self.opt.isTrain:
            self.stu_out_ref4x = self.netStudent(self.data_lr, self.data_ref4x)

            self.data_sr = self.netRefSR(
                self.data_lr, self.data_lr, self.image_name, self.stu_out_ref4x, self.image_name, 
                self.data_ref4x, self.data_ref2x_lr, self.image_name, self.coord_ref4x)		
        else:
            # Using patch-based optical flow
            if self.opt.camera == 'nikon':  
                down_hr = F.interpolate(input=self.data_hr, scale_factor=1/self.scale, \
                                        mode='bilinear',align_corners=True)
                flow = self.get_flow(down_hr, self.data_lr, self.netPWCNET)  # [16, 2, 48, 48]
                self.lr_warp, self.lr_mask = self.get_backwarp('/', self.data_lr, self.netPWCNET, flow)
                self.hr_mask = F.interpolate(input=self.lr_mask, scale_factor=self.scale, \
                                            mode='bilinear', align_corners=True)

                self.data_down_hr, self.weight = self.netKernelGen(self.data_hr*self.hr_mask, self.lr_warp)
                self.data_down_hr_de = self.data_down_hr.detach()
        
                self.data_hr_bic = F.interpolate(self.data_hr, scale_factor=1/self.opt.scale, \
                                                mode='bicubic', align_corners=True)
                self.data_down_hr_de = self.data_down_hr_de + (self.data_noise - self.data_hr_bic) * 1.0
                self.data_down_hr_de = torch.clamp(self.data_down_hr_de, 0, 1) * self.lr_mask
                
                self.stu_out_ref4x = self.netStudent(self.lr_warp, self.data_ref4x)

                self.data_sr = self.netRefSR(
                    self.lr_warp, self.data_down_hr_de, self.image_name, self.stu_out_ref4x, self.image_name, 
                    self.data_ref4x, self.data_ref2x_lr, self.image_name, self.coord_ref4x)
            
            # Not using patch-based optical flow
            # The size of full image is small.
            # The full image can be regarded as a small patch during pre-alignment.
            elif self.opt.camera == 'iphone':  
                self.data_down_hr, self.weight = self.netKernelGen(self.data_hr, self.data_lr)
                self.data_down_hr_de = self.data_down_hr.detach()
        
                self.data_hr_bic = F.interpolate(self.data_hr, scale_factor=1/self.opt.scale, \
                                                mode='bicubic', align_corners=True)
                self.data_down_hr_de = self.data_down_hr_de + (self.data_noise - self.data_hr_bic) * 1.0
                self.data_down_hr_de = torch.clamp(self.data_down_hr_de, 0, 1)
                
                self.stu_out_ref4x = self.netStudent(self.data_lr, self.data_ref4x)

                self.data_sr = self.netRefSR(
                    self.data_lr, self.data_down_hr_de, self.image_name, self.stu_out_ref4x, self.image_name, 
                    self.data_ref4x, self.data_ref2x_lr, self.image_name, self.coord_ref4x)

    def backward(self):
        if self.opt.camera == 'nikon':
            self.loss_KernelGen_L1 = self.criterionL1(self.lr_warp, self.data_down_hr*self.lr_mask).mean()
            self.loss_KernelGen_Filter = self.criterionFilter(self.weight[0]).mean() * 100
            for conv_w in self.weight[1:]:
                self.loss_KernelGen_Filter = self.loss_KernelGen_Filter + self.criterionFilter(conv_w).mean() * 100
            self.loss_KernelGen_Total = self.loss_KernelGen_L1 + self.loss_KernelGen_Filter

            self.loss_RefSR_L1 = self.criterionL1(self.data_hr*self.hr_mask, self.data_sr*self.hr_mask).mean()
            self.loss_RefSR_SWD = self.criterionSWD(self.data_hr*self.hr_mask, self.data_sr*self.hr_mask).mean()
            self.loss_RefSR_Total = self.loss_RefSR_L1 + self.loss_RefSR_SWD

            self.loss_Total = self.loss_RefSR_Total + self.loss_KernelGen_Total
            self.loss_Total.backward()

        elif self.opt.camera == 'iphone':
            self.loss_KernelGen_L1 = self.criterionL1(self.data_lr, self.data_down_hr).mean()
            self.loss_KernelGen_Filter = self.criterionFilter(self.weight[0]).mean() * 100
            for conv_w in self.weight[1:]:
                self.loss_KernelGen_Filter = self.loss_KernelGen_Filter + self.criterionFilter(conv_w).mean() * 100
            self.loss_KernelGen_Total = self.loss_KernelGen_L1 + self.loss_KernelGen_Filter

            self.loss_RefSR_L1 = self.criterionL1(self.data_hr, self.data_sr).mean()
            self.loss_RefSR_SWD = self.criterionSWD(self.data_hr, self.data_sr).mean()
            self.loss_RefSR_Total = self.loss_RefSR_L1 + self.loss_RefSR_SWD

            self.loss_Total = self.loss_RefSR_Total + self.loss_KernelGen_Total
            self.loss_Total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_KernelGen.zero_grad()
        self.optimizer_RefSR.zero_grad()
        self.backward()
        self.optimizer_KernelGen.step()
        self.optimizer_RefSR.step()


class SelfRefSR(nn.Module):
    def __init__(self, opt):
        super(SelfRefSR, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.n_resblock = 16
        n_feats = 64
        self.paste = opt.paste
        self.predict = opt.predict
        n_upscale = int(math.log(opt.scale, 2))
        
        self.corr = N.CorrespondenceGeneration()

        ref_extractor_2 = [N.MeanShift(),
                           N.conv(3, 64, mode='CR'),
                           N.conv(64, 64, mode='CRCRCRC')]
        self.ref_extractor_2 = N.seq(ref_extractor_2)

        if self.paste:
            ref_head_2 = [N.MeanShift(),
                          N.DownBlock(4),
                          N.conv(3*4**2, n_feats, mode='C')]
            self.ref_head_2 = N.seq(ref_head_2)

        m_head = [N.MeanShift(),
                  N.conv(3, n_feats, mode='C')]
        self.head = N.seq(m_head)

        self.ada1 = N.AdaptBlock(opt, n_feats, n_feats)
        self.ada2 = N.AdaptBlock(opt, n_feats, n_feats)
        self.ada3 = N.AdaptBlock(opt, n_feats, n_feats)

        self.deform_conv_2 = ops.DeformConv2d(64, 64, kernel_size=3, stride=1, \
            padding=1, dilation=1, bias=True, groups=64)

        if self.predict:
            self.predictor = N.Predictor(opt)

        self.concat_fea = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
            
        for i in range(self.n_resblock):
            setattr(self, 'block%d'%i, N.ResBlock(n_feats, n_feats, mode='CRC', predict=self.predict))
        self.body_lastconv = N.conv(n_feats, n_feats, mode='C')

        if opt.scale == 3:
            m_up = N.upsample_pixelshuffle(n_feats, n_feats, mode='3')
        else:
            m_up = [N.upsample_pixelshuffle(n_feats, n_feats, mode='2') \
                      for _ in range(n_upscale)]
        self.up = N.seq(m_up)

        m_tail = [N.conv(n_feats, 3, mode='C'),
                  N.MeanShift(sign=1)]
        self.tail = N.seq(m_tail)
        
    def forward(self, lr, hr_bic, stu_out_ref2x, stu_out_ref4x, data_ref2x, data_ref4x, data_ref2x_lr, coord_ref2x, coord_ref4x, paste=True):
        N, C, H, W = lr.size()

        pre_offset_ref4x = self.corr(stu_out_ref4x)
        pre_offset_ref4x = F.interpolate(pre_offset_ref4x, data_ref4x.shape[-2:], mode='bilinear', align_corners=True) * 4
        ref4x_feats = self.ref_extractor_2(data_ref4x)
        ref4x_deform = self.deform_conv_2(ref4x_feats, pre_offset_ref4x)    
        ref4x_deform = F.interpolate(ref4x_deform, lr.shape[-2:], mode='bilinear', align_corners=True)

        del pre_offset_ref4x, ref4x_feats, stu_out_ref4x

        h = self.head(lr) 
        if hr_bic is None:
            h_hr = h.clone()
        else:
            h_hr = self.head(hr_bic) 

        h = self.ada1(h, h_hr)
        h = self.ada2(h, h_hr)
        h = self.ada3(h, h_hr)

        if paste and self.opt.isTrain:
            head_ref4x = self.ref_head_2(data_ref4x)
            for i in range(N):
                rand_num = np.random.rand() 
                if rand_num < 0.3:
                    ref4x_deform[i, :, coord_ref4x[i,0]:coord_ref4x[i,1], 
                            coord_ref4x[i,2]:coord_ref4x[i,3]] = head_ref4x[i] 
        elif paste and not self.opt.isTrain:
            head_ref4x = self.ref_head_2(data_ref4x)
            for i in range(N):
                ref4x_deform[i, :, coord_ref4x[i,0]:coord_ref4x[i,1], 
                        coord_ref4x[i,2]:coord_ref4x[i,3]] = head_ref4x[i] 
        
        cat_fea = self.concat_fea(torch.cat([h, ref4x_deform], 1))
        res = cat_fea.clone()
        
        if self.predict:
            _, _, H, W = data_ref2x_lr.size()
            pre_lr = data_ref2x_lr[:,:, H//4:3*H//4, W//4:3*W//4]
            pre = self.predictor(pre_lr, data_ref4x, cat_fea)
        else:
            pre = None

        for i in range(self.n_resblock):
            res = getattr(self, 'block%d'%i)(res, pre)

        del pre, cat_fea, ref4x_deform
        
        res = self.body_lastconv(res)
        res += h
        res = self.up(res)
        out = self.tail(res)	
        return out

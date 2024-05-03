# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp, sqrt
from torch.nn import L1Loss, MSELoss
from torchvision import models
from util.util import grid_positions, warp
import random


# -------------------------------------------------------
# SSIM Loss
# -------------------------------------------------------

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(
            -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) \
        for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and \
                self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size,
                     channel, self.size_average)


# -------------------------------------------------------
# VGG Loss
# -------------------------------------------------------

def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std
    

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def forward(self, img1, img2, p=6):
        x = normalize_batch(img1)
        y = normalize_batch(img2)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        # content_loss += self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2']) * 0.1
        # content_loss += self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2']) * 0.2
        content_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2']) * 1
        content_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2']) * 1
        content_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2']) * 2

        return content_loss / 4.


# -------------------------------------------------------
# LOSW (Local Overlapped Sliced Wasserstein) Loss
# -------------------------------------------------------

class SWDLoss(nn.Module):
    def __init__(self):
        super(SWDLoss, self).__init__()
        self.add_module('vgg', VGG19())
        # self.criterion = SWD()
        self.criterion = SWDLocal()

    def forward(self, img1, img2, p=6):
        x = normalize_batch(img1)
        y = normalize_batch(img2)
        N, C, H, W = x.shape  # 192*192
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        swd_loss = 0.0
        swd_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2'], k=H//4//p) * 1  # H//4=48
        swd_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2'], k=H//8//p) * 1  # H//4=24
        swd_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2'], k=H//16//p) * 2  # H//4=12

        return swd_loss * 8 / 100.0


class SWD(nn.Module):
    def __init__(self):
        super(SWD, self).__init__()
        self.l1loss = torch.nn.L1Loss() 

    def forward(self, fake_samples, true_samples, k=0):
        N, C, H, W = true_samples.shape

        num_projections = C//2

        true_samples = true_samples.view(N, C, -1)
        fake_samples = fake_samples.view(N, C, -1)

        projections = torch.from_numpy(np.random.normal(size=(num_projections, C)).astype(np.float32))
        projections = torch.FloatTensor(projections).to(true_samples.device)
        projections = F.normalize(projections, p=2, dim=1)

        projected_true = projections @ true_samples
        projected_fake = projections @ fake_samples

        sorted_true, true_index = torch.sort(projected_true, dim=2)
        sorted_fake, fake_index = torch.sort(projected_fake, dim=2)
        return self.l1loss(sorted_true, sorted_fake).mean() 


class SWDLocal(torch.nn.Module):
    def __init__(self):
        super(SWDLocal, self).__init__()
        self.l1loss = torch.nn.L1Loss()

    def forward(self, true_samples, fake_samples, k):
        N, C, H, W = true_samples.shape
        num_projections = C//2 

        true_samples = F.unfold(true_samples, kernel_size=(k, k), padding=0, stride=2) # [N, C*k*k, H*W]
        fake_samples = F.unfold(fake_samples, kernel_size=(k, k), padding=0, stride=2)

        p = true_samples.shape[-1]
        # [N, 3, 3, 3, 4096] -> [N, 3, 4096, 3, 3]
        true_samples = true_samples.view(N, C, k, k, p).permute(0, 4, 1, 2, 3).contiguous()
        true_samples = true_samples.view(N, p, C, k*k)
        fake_samples = fake_samples.view(N, C, k, k, p).permute(0, 4, 1, 2, 3).contiguous()
        fake_samples = fake_samples.view(N, p, C, k*k)

        projections = torch.from_numpy(np.random.normal(size=(num_projections, C)).astype(np.float32))
        projections = torch.FloatTensor(projections).to(true_samples.device)
        projections = F.normalize(projections, p=2, dim=1)

        projected_true = projections @ true_samples   
        projected_fake = projections @ fake_samples 

        sorted_true, true_index = torch.sort(projected_true, dim=3)
        sorted_fake, fake_index = torch.sort(projected_fake, dim=3)

        return self.l1loss(sorted_true, sorted_fake).mean()


# -------------------------------------------------------
# Position Preserving Loss for Auxiliary-LR Generator
# -------------------------------------------------------

class FilterLoss(nn.Module): # kernel_size%2 = 1
    def __init__(self):
        super(FilterLoss, self).__init__()

    def forward(self, filter_weight):  # [out, in, kernel_size, kernel_size]
        weight = filter_weight
        out_c, in_c, k, k = weight.shape 
        index = torch.arange(-(k//2), k//2+1, 1)

        index = index.to(filter_weight.device)
        index = index.unsqueeze(dim=0).unsqueeze(dim=0)  # [1, 1, kernel_size] 
        index_i = index.unsqueeze(dim=3)  # [1, 1, kernel_size, 1]  
        index_j = index.unsqueeze(dim=0)  # [1, 1, 1, kernel_size]  

        diff = torch.mean(weight*index_i, dim=2).abs() + torch.mean(weight*index_j, dim=3).abs()
        return diff.mean()
        

# -------------------------------------------------------
# CoBi Loss in https://arxiv.org/abs/1905.05169
# The code was modified from https://github.com/roimehrez/contextualLoss
# -------------------------------------------------------

class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3


class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        self.cs_NHWC = self.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)

    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.1), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        
        r_Ts = torch.sum(Tvecs * Tvecs, 2)
        r_Is = torch.sum(Ivecs * Ivecs, 2)
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
            A = Tvec @ torch.transpose(Ivec, 0, 1)  # (matrix multiplication)
            cs_flow.A = A
            r_T = torch.reshape(r_T, [-1, 1])  # turn to column vector
            dist = r_T - 2 * A + r_I
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_L1(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec = Ivecs[i], Tvecs[i]
            dist = torch.abs(torch.sum(Ivec.unsqueeze(1) - Tvec.unsqueeze(0), dim=2))
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(1), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # prepare feature before calculating cosine distance
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        I_features = CSFlow.l2_normalize_channelwise(I_features)

        # work seperatly for each example in dim 1
        cosine_dist_l = []
        N = T_features.size()[0]
        for i in range(N):
            T_features_i = T_features[i, :, :, :].unsqueeze_(0)  # 1HWC --> 1CHW
            I_features_i = I_features[i, :, :, :].unsqueeze_(0).permute((0, 3, 1, 2))
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1HWC --> PC11, with P=H*W
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
            cosine_dist_1HWC = cosine_dist_i.permute((0, 2, 3, 1))
            cosine_dist_l.append(cosine_dist_i.permute((0, 2, 3, 1)))  # back to 1HWC

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)

        cs_flow.raw_distances = - (cs_flow.cosine_dist - 1) / 2  ### why -

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size
        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        self.varT = T_features.var(0, keepdim=True).var(1, keepdim=True).var(2, keepdim=True)
        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, T_features):
        # 1HWC --> 11PC --> PC11, with P=H*W
        (N, H, W, C) = T_features.shape
        P = H * W
        patches_PC11 = T_features.reshape(shape=(1, 1, P, C)).permute(dims=(2, 3, 0, 1))
        return patches_PC11

    @staticmethod
    def pdist2(x, keepdim=False):
        sx = x.shape
        x = x.reshape(shape=(sx[0], sx[1] * sx[2], sx[3]))
        differences = x.unsqueeze(2) - x.unsqueeze(1)
        distances = torch.sum(differences**2, -1)
        if keepdim:
            distances = distances.reshape(shape=(sx[0], sx[1], sx[2], sx[3]))
        return distances

    @staticmethod
    def calcR_static(sT, order='C', deformation_sigma=0.05):
        # oreder can be C or F (matlab order)
        pixel_count = sT[0] * sT[1]

        rangeRows = range(0, sT[1])
        rangeCols = range(0, sT[0])
        Js, Is = np.meshgrid(rangeRows, rangeCols)
        row_diff_from_first_row = Is
        col_diff_from_first_col = Js

        row_diff_from_first_row_3d_repeat = np.repeat(row_diff_from_first_row[:, :, np.newaxis], pixel_count, axis=2)
        col_diff_from_first_col_3d_repeat = np.repeat(col_diff_from_first_col[:, :, np.newaxis], pixel_count, axis=2)

        rowDiffs = -row_diff_from_first_row_3d_repeat + row_diff_from_first_row.flatten(order).reshape(1, 1, -1)
        colDiffs = -col_diff_from_first_col_3d_repeat + col_diff_from_first_col.flatten(order).reshape(1, 1, -1)
        R = rowDiffs ** 2 + colDiffs ** 2
        R = R.astype(np.float32)
        R = np.exp(-(R) / (2 * deformation_sigma ** 2))
        return R


def random_sampling(tensor_NHWC, n, indices=None):
    N, H, W, C = tensor_NHWC.size()
    S = H * W
    tensor_NSC = torch.reshape(tensor_NHWC, [N, S, C])

    if indices is None:
        all_indices = list(range(S))
        random.shuffle(all_indices)
        shuffled_indices = torch.from_numpy(np.array(all_indices)).type(torch.int64).to(tensor_NHWC.device)

        no_shuffled_indices = torch.from_numpy(np.array(list(range(n)))).type(torch.int64).to(tensor_NHWC.device)
        indices = shuffled_indices[no_shuffled_indices] if indices is None else indices
    res = tensor_NSC[:, indices, :]

    return res, indices


def random_pooling(feats, output_1d_size=100):
    N, H, W, C = feats[0].size()
    feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2)
    res = [feats_sampled_0]

    for i in range(1, len(feats)):
        feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
        res.append(feats_sampled_i)
    res = [torch.reshape(feats_sampled_i, [N, output_1d_size, output_1d_size, C]) for feats_sampled_i in res]
    return res


# CoBi loss
def CoBi_loss(T_features, I_features, nnsigma=1, b=0.5, w_spatial=0.2, maxsize=101, deformation=False, dis=False):
    device = T_features.device
    
    # since this originally Tensorflow implemntation
    # we modify all tensors to be as TF convention and not as the convention of pytorch.
    def from_pt2tf(Tpt):
        Ttf = Tpt.permute(0, 2, 3, 1)
        return Ttf
    # N x C x H x W --> N x H x W x C

    _,_,fh,fw = T_features.size()
    if fh*fw > maxsize**2:
        T_features_tf, I_features_tf = random_pooling([from_pt2tf(T_features),from_pt2tf(I_features)])
    else:
        T_features_tf = from_pt2tf(T_features)
        I_features_tf = from_pt2tf(I_features)

    rows = torch.arange(0,T_features_tf.shape[1]).to(device)
    cols = torch.arange(0,T_features_tf.shape[2]).to(device)
    rows = rows.type(torch.float32)/(T_features_tf.shape[1]) # * 255.
    cols = rows.type(torch.float32)/(T_features_tf.shape[0]) # * 255.

    features_grid = torch.meshgrid(rows, cols)
    features_grid = torch.cat([torch.unsqueeze(features_grid_i, 2) for features_grid_i in features_grid], axis=2)
    features_grid = torch.unsqueeze(features_grid, axis=0)
    features_grid = features_grid.repeat([T_features_tf.shape[0], 1, 1, 1])

    cs_flow_sp = CSFlow.create_using_L2(features_grid, features_grid, nnsigma, b)
    cs_flow = CSFlow.create_using_L2(I_features_tf, T_features_tf, nnsigma, b) # [N,H,W,C]->[N,H,W,H*W]

    # To:
    cs = cs_flow.cs_NHWC
    cs_sp = cs_flow_sp.cs_NHWC
    cs_comb = cs * (1.-w_spatial) + cs_sp * w_spatial
    
    k_max_NC = torch.max(torch.max(cs_comb, 1)[0],1)[0]

    CS = torch.mean(k_max_NC, 1)
    loss = -torch.log(CS + 1e-5)
    loss = torch.mean(loss)
    return loss


def symetric_CoBi_loss(T_features, I_features):
    score = (CoBi_loss(T_features, I_features) + CoBi_loss(I_features, T_features)) / 2
    return score


class CobiLoss(nn.Module):
    def __init__(self):
        super(CobiLoss,self).__init__()

    def forward(self, T_features, I_features):
        N, C, _, _ = T_features.size()
        kernel = 16

        T_features = F.unfold(T_features, kernel_size=(kernel, kernel), padding=0, stride=1) # [N, 27, 4096] [N, C*k*k, H*W]
        I_features = F.unfold(I_features, kernel_size=(kernel, kernel), padding=0, stride=1) # [N, 27, 4096] [N, C*k*k, H*W]
        
        p = I_features.shape[2]
        
        T_features = T_features.view(N, C, kernel, kernel, p).permute(0, 4, 1, 2, 3).contiguous()
        T_features = T_features.view(N, p*C, kernel, kernel)

        I_features = I_features.view(N, C, kernel, kernel, p).permute(0, 4, 1, 2, 3).contiguous()
        I_features = I_features.view(N, p*C, kernel, kernel)

        return CoBi_loss(T_features, I_features) 


# -------------------------------------------------------
# Margin Loss
# -------------------------------------------------------

class MarginLoss(nn.Module):
    def __init__(self, opt, kl=False):
        super(MarginLoss, self).__init__()
        self.margin = 1.0  
        self.safe_radius = 4  # tea:3; stu:4
        self.scaling_steps = 2 
        self.temperature = 0.15 
        self.distill_weight = 15 
        self.perturb = opt.perturb
        self.kl = kl

    def forward(self, img1_1, img1_2, img2_1=None, img2_2=None, transformed_coordinates=None):
        device = img1_1.device
        loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
        pos_dist = 0.
        neg_dist = 0.
        distill_loss_all = 0.
        has_grad = False

        n_valid_samples = 0
        batch_size = img1_1.size(0)

        for idx_in_batch in range(batch_size):
            # Network output
            # shape: [c, h1, w1]
            dense_features1 = img1_1[idx_in_batch]
            c, h1, w1 = dense_features1.size()  # [256, 48, 48]

            # shape: [c, h2, w2]
            dense_features2 = img1_2[idx_in_batch]
            _, h2, w2 = dense_features2.size()  # [256, 48, 48]

            # shape: [c, h1 * w1]
            all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
            descriptors1 = all_descriptors1

            # Warp the positions from image 1 to image 2\
            # shape: [2, h1 * w1], coordinate in [h1, w1] dim,
            # dim 0: y, dim 1: x, positions in feature map
            fmap_pos1 = grid_positions(h1, w1, device)
            # shape: [2, h1 * w1], coordinate in image level (4 * h1, 4 * w1)
            # pos1 = upscale_positions(fmap_pos1, scaling_steps=self.scaling_steps)
            pos1 = fmap_pos1
            pos1, pos2, ids = warp(pos1, h1, w1, 
                transformed_coordinates[idx_in_batch], self.perturb)

            # shape: [2, num_ids]
            fmap_pos1 = fmap_pos1[:, ids]
            # shape: [c, num_ids]
            descriptors1 = descriptors1[:, ids]

            # Skip the pair if not enough GT correspondences are available
            if ids.size(0) < 128:
                continue

            # Descriptors at the corresponding positions
            # fmap_pos2 = torch.round(downscale_positions(pos2, \
            # 	scaling_steps=self.scaling_steps)).long()  # [2, hw]
            fmap_pos2 = torch.round(pos2).long()  # [2, hw]

            # [256, 48, 48] -> [256, hw]
            descriptors2 = F.normalize(
                dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0)
            
            # [hw, 1, 256] @ [hw, 256, 1] -> [hw, hw]
            positive_distance = 2 - 2 * (descriptors1.t().unsqueeze(1) @ \
                descriptors2.t().unsqueeze(2)).squeeze()  
                
            position_distance = torch.max(torch.abs(fmap_pos2.unsqueeze(2).float() - 
                fmap_pos2.unsqueeze(1)), dim=0)[0]  # [hw, hw]
            is_out_of_safe_radius = position_distance > self.safe_radius
            distance_matrix = 2 - 2 * (descriptors1.t() @ descriptors2)  # [hw, hw]
            negative_distance2 = torch.min(distance_matrix + (1 - 
                is_out_of_safe_radius.float()) * 10., dim=1)[0]  # [hw]

            all_fmap_pos1 = grid_positions(h1, w1, device)
            position_distance = torch.max(torch.abs(fmap_pos1.unsqueeze(2).float() - 
                all_fmap_pos1.unsqueeze(1)), dim=0)[0]
            is_out_of_safe_radius = position_distance > self.safe_radius
            distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
            negative_distance1 = torch.min(distance_matrix + (1 - 
                is_out_of_safe_radius.float()) * 10., dim=1)[0]

            diff = positive_distance - torch.min(negative_distance1, negative_distance2)

            if not self.kl:
                loss = loss + torch.mean(F.relu(self.margin + diff))
            else:
                # distillation loss
                # student model correlation
                student_distance = torch.matmul(descriptors1.transpose(0, 1), descriptors2)
                student_distance = student_distance / self.temperature
                student_distance = F.log_softmax(student_distance, dim=1)

                # teacher model correlation
                teacher_dense_features1 = img2_1[idx_in_batch]
                c, h1, w1 = dense_features1.size()
                teacher_descriptors1 = F.normalize(teacher_dense_features1.view(c, -1), dim=0)
                teacher_descriptors1 = teacher_descriptors1[:, ids]

                teacher_dense_features2 = img2_2[idx_in_batch]
                teacher_descriptors2 = F.normalize(
                    teacher_dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0)
                
                teacher_distance = torch.matmul(
                    teacher_descriptors1.transpose(0, 1), teacher_descriptors2)
                teacher_distance = teacher_distance / self.temperature
                teacher_distance = F.softmax(teacher_distance, dim=1)

                distill_loss = F.kl_div(student_distance, teacher_distance, \
                    reduction='batchmean') * self.distill_weight
                distill_loss_all += distill_loss

                loss = loss + torch.mean(F.relu(self.margin + diff)) + distill_loss

            pos_dist = pos_dist + torch.mean(positive_distance)
            neg_dist = neg_dist + torch.mean(torch.min(negative_distance1, negative_distance2))

            has_grad = True
            n_valid_samples += 1
        
        if not has_grad:
            raise NotImplementedError

        loss = loss / n_valid_samples
        pos_dist = pos_dist / n_valid_samples
        neg_dist = neg_dist / n_valid_samples

        if not self.kl:
            return loss, pos_dist, neg_dist
        else:
            distill_loss_all = distill_loss_all / n_valid_samples
            return loss, pos_dist, neg_dist, distill_loss_all


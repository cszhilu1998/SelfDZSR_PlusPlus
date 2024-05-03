import random
import numpy as np
import os
from data.base_dataset import BaseDataset
from .imlib import imlib
from util.util import *
from .degrade.degrade_kernel import degrade_kernel


class NIKONDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='NIKON'):
        super(NIKONDataset, self).__init__(opt, split, dataset_name)
        if self.root == '':
            rootlist = ['']
            for root in rootlist:
                if os.path.isdir(root):
                    self.root = root
                    break
    
        self.batch_size = opt.batch_size
        self.mode = opt.mode  # RGB, Y or L
        self.imio = imlib(self.mode, lib=opt.imlib)
        self.patch_size = opt.patch_size  # 48

        self.scale = 4
        self.camera = {'nikon':['DSC']}

        self.x_scale = 'x' + str(self.scale)
        if split == 'train':
            self.train_root = os.path.join(self.root, 'train_')
            self.lr_dirs, self.ref2x_dirs, self.hr_dirs, self.names = self._get_image_dir(self.train_root, self.camera['nikon'])
            self.len_data = 1000 * self.batch_size  
            self._getitem = self._getitem_train

        elif split == 'val':
            self.val_root = os.path.join(self.root, 'test_')
            self.lr_dirs, self.ref2x_dirs, self.hr_dirs, self.names = self._get_image_dir(self.val_root, self.camera['nikon']) 
            self._getitem = self._getitem_val
            self.len_data = len(self.names)

        elif split == 'test':
            self.test_root = os.path.join(self.root, 'test_')
            self.lr_dirs, self.ref2x_dirs, self.hr_dirs, self.names = self._get_image_dir(self.test_root, self.camera['nikon']) 
            self._getitem = self._getitem_test
            self.len_data = len(self.names)

        else:
            raise ValueError

        self.lr_images = [0] * len(self.names)
        self.ref2x_images = [0] * len(self.names)
        self.hr_images = [0] * len(self.names)
        read_images(self)

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data

    def _getitem_train(self, idx):
        idx = idx % len(self.names)

        lr_img = self.lr_images[idx]
        hr_img = self.hr_images[idx]
        ref2x_img = self.ref2x_images[idx]

        lr_img, ref2x_img, hr_img, _ = self._crop_patch(lr_img, ref2x_img, hr_img, p=self.patch_size)
        lr_img, ref2x_img, hr_img = augment(lr_img, ref2x_img, hr_img)

        ref2x_lr, ref2x, ref4x, coord_ref2x, coord_ref4x = self._crop_ref(lr_img, ref2x_img, hr_img)

        hr_trans = np.transpose(hr_img, (1, 2, 0))
        hr_bili_noise, degradation_list = degrade_kernel(hr_trans, self.scale)
        noise = hr_bili_noise
        noise = np.transpose(noise, (2, 0, 1))
        
        noise = np.float32(noise) / 255
        lr_img = np.float32(lr_img) / 255 
        hr_img = np.float32(hr_img) / 255
        ref2x = np.float32(ref2x) / 255
        ref2x_lr = np.float32(ref2x_lr) / 255
        ref4x = np.float32(ref4x) / 255

        return {'lr': lr_img,
                'hr': hr_img,
                'noise': noise,
                'ref2x_lr': ref2x_lr,
                'ref2x': ref2x,
                'ref4x': ref4x,
                'coord_ref2x': coord_ref2x,
                'coord_ref4x': coord_ref4x,
                'fname': self.names[idx]}

    def _getitem_val(self, idx):
        lr_img = self.lr_images[idx]
        hr_img = self.hr_images[idx]
        ref2x_img = self.ref2x_images[idx]

        lr_img, ref2x_img, hr_img = self._crop_center(lr_img, ref2x_img, hr_img, p=256)
        ref2x_lr, ref2x, ref4x, coord_ref2x, coord_ref4x = self._crop_ref_center(lr_img, ref2x_img, hr_img)

        hr_trans = np.transpose(hr_img, (1, 2, 0))
        hr_bili_noise, degradation_list = degrade_kernel(hr_trans, self.scale)
        noise = hr_bili_noise
        noise = np.transpose(noise, (2, 0, 1))
        
        noise = np.float32(noise) / 255
        lr_img = np.float32(lr_img) / 255 
        hr_img = np.float32(hr_img) / 255
        ref2x = np.float32(ref2x) / 255
        ref2x_lr = np.float32(ref2x_lr) / 255
        ref4x = np.float32(ref4x) / 255

        return {'lr': lr_img,
                'hr': hr_img,
                'noise': noise,
                'ref2x_lr': ref2x_lr,
                'ref2x': ref2x,
                'ref4x': ref4x,
                'coord_ref2x': coord_ref2x,
                'coord_ref4x': coord_ref4x,
                'fname': self.names[idx]}

    def _getitem_test(self, idx):
        lr_img = self.lr_images[idx]
        hr_img = self.hr_images[idx]
        ref2x_img = self.ref2x_images[idx]

        lr_img, ref2x_img, hr_img, paddings = self.img_pad(lr_img, ref2x_img, hr_img)

        if not self.opt.full_res:
            lr_img, ref2x_img, hr_img = self._crop_center(lr_img, ref2x_img, hr_img, p=400)
        ref2x_lr, ref2x, ref4x, coord_ref2x, coord_ref4x = self._crop_ref_center(lr_img, ref2x_img, hr_img)

        hr_trans = np.transpose(hr_img, (1, 2, 0))
        hr_bili_noise, degradation_list = degrade_kernel(hr_trans, self.scale)
        noise = hr_bili_noise
        noise = np.transpose(noise, (2, 0, 1))
        
        noise = np.float32(noise) / 255
        lr_img = np.float32(lr_img) / 255 
        hr_img = np.float32(hr_img) / 255
        ref2x = np.float32(ref2x) / 255
        ref2x_lr = np.float32(ref2x_lr) / 255
        ref4x = np.float32(ref4x) / 255

        return {'lr': lr_img,
                'hr': hr_img,
                'noise': noise,
                'ref2x_lr': ref2x_lr,
                'ref2x': ref2x,
                'ref4x': ref4x,
                'coord_ref2x': coord_ref2x,
                'coord_ref4x': coord_ref4x,
                'fname': self.names[idx],
                'paddings': paddings}
   
    def img_pad(self, lr_img, ref2x_img, hr_img):
        ih, iw = lr_img.shape[-2:]
        new_ih = (ih // 8 + 1) * 8
        new_iw = (iw // 8 + 1) * 8

        pad_h = new_ih - ih
        pad_w = new_iw - iw

        p_t = int(pad_h / 2.)
        p_b = pad_h - p_t
        p_l = int(pad_w / 2.)
        p_r = pad_w - p_l

        paddings = np.array([4*p_t, 4*p_t + 4*ih, 4*p_l, 4*p_l + 4*iw])

        lr_img = np.pad(lr_img,((0,0),(p_t,p_b),(p_l,p_r)),'edge') 
        ref2x_img = np.pad(ref2x_img,((0,0),(2*p_t,2*p_b),(2*p_l,2*p_r)),'edge') 
        hr_img = np.pad(hr_img,((0,0),(4*p_t,4*p_b),(4*p_l,4*p_r)),'edge') 

        return lr_img, ref2x_img, hr_img, paddings

    def _get_image_dir(self, dataroot, cameras=['']):
        lr_dirs = []
        ref2x_dirs = []
        hr_dirs = []  # input_x4_raw target_x4_rgb
        image_names = []

        for file_name in os.listdir(dataroot + 'LR/'):  
            if cameras != ['']:
                for camera in cameras:
                    if file_name.startswith(camera):
                        lr_dirs.append(dataroot + 'LR/' + file_name)
                        hr_file_name = file_name.replace('x1', self.x_scale)
                        image_names.append(hr_file_name)
                        hr_dirs.append(dataroot + 'HR/' + hr_file_name)

                        ref2x_file_name = file_name.replace('x1', 'x2')
                        ref2x_dirs.append(dataroot + 'Ref/' + ref2x_file_name)		
            else:
                lr_dirs.append(dataroot + 'LR/' + file_name)
                hr_file_name = file_name.replace('x1', self.x_scale)
                image_names.append(hr_file_name)
                hr_dirs.append(dataroot + 'HR/' + hr_file_name)

                ref2x_file_name = file_name.replace('x1', 'x2')
                ref2x_dirs.append(dataroot + 'Ref/' + ref2x_file_name)
        
        image_names = sorted(image_names) 
        lr_dirs = sorted(lr_dirs)
        hr_dirs = sorted(hr_dirs)
        ref2x_dirs = sorted(ref2x_dirs)

        return lr_dirs, ref2x_dirs, hr_dirs, image_names
        # return lr_dirs[19:], ref2x_dirs[19:], hr_dirs[19:], image_names[19:]

    def _crop_patch(self, lr, ref2x, hr, p):
        ih, iw = lr.shape[-2:]
        pw = random.randrange(0, iw - p + 1)
        ph = random.randrange(0, ih - p + 1)
        hpw, hph = self.scale * pw, self.scale * ph
        hr_patch_size = self.scale * p
        crop_coord = [ph, ph+p, pw, pw+p]
        crop_coord = np.array(crop_coord, dtype=np.int32)
        return lr[..., ph:ph+p, pw:pw+p], \
               ref2x[..., 2*ph:2*ph+2*p, 2*pw:2*pw+2*p], \
               hr[..., hph:hph+hr_patch_size, hpw:hpw+hr_patch_size], \
               crop_coord

    def _crop_ref(self, lr, ref, hr, p=None):
        p = self.patch_size // 2 # random.randrange(4, self.patch_size-16)
        ih, iw = lr.shape[-2:]
        ph = random.randrange(0, ih - p + 1)
        pw = random.randrange(0, iw - p + 1)
        coord_ref = [ph, ph+p, pw, pw+p]
        coord_ref = np.array(coord_ref, dtype=np.int32)
        coord_hr = [ph+p//4, ph+3*p//4, pw+p//4, pw+3*p//4]
        coord_hr = np.array(coord_hr, dtype=np.int32)

        return lr[..., ph:ph+p, pw:pw+p], \
               ref[..., 2*ph:2*ph+2*p, 2*pw:2*pw+2*p], \
               hr[..., 4*ph+p:4*ph+3*p, 4*pw+p:4*pw+3*p], \
               coord_ref, coord_hr

    def _crop_center(self, lr, ref, hr, fw=0.25, fh=0.25, p=0):
        ih, iw = lr.shape[-2:]
        if p != 0:
            fw = p / iw
            fh = p / ih
        lr_patch_h, lr_patch_w = round(ih * fh), round(iw * fw)
        ph = ih // 2 - lr_patch_h // 2
        pw = iw // 2 - lr_patch_w // 2

        return lr[..., ph:ph+lr_patch_h, pw:pw+lr_patch_w], \
               ref[..., 2*ph:2*ph+2*lr_patch_h, 2*pw:2*pw+2*lr_patch_w], \
               hr[..., 4*ph:4*ph+4*lr_patch_h, 4*pw:4*pw+4*lr_patch_w]

    def _crop_ref_center(self, lr, ref, hr):
        ih, iw = lr.shape[-2:]
        lr_patch_h, lr_patch_w = ih // 2, iw // 2
        ph = ih // 4
        pw = iw // 4
        
        coord_ref = [ph, ph+lr_patch_h, pw, pw+lr_patch_w]
        coord_ref = np.array(coord_ref, dtype=np.int32)
        coord_hr = [3*ih//8, 3*ih//8+ih//4, 3*iw//8, 3*iw//8+iw//4]
        coord_hr = np.array(coord_hr, dtype=np.int32)

        return lr[..., ph:ph+lr_patch_h, pw:pw+lr_patch_w], \
               ref[..., 2*(ph):2*(ph+lr_patch_h), 2*(pw):2*(pw+lr_patch_w)], \
               hr[..., 4*(3*ih//8):4*(3*ih//8+ih//4), 4*(3*iw//8):4*(3*iw//8+iw//4)], \
               coord_ref, coord_hr


def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    for _ in range(3):
        try:
            obj.lr_images[i] = obj.imio.read(obj.lr_dirs[i])
            obj.hr_images[i] = obj.imio.read(obj.hr_dirs[i])
            obj.ref2x_images[i] = obj.imio.read(obj.ref2x_dirs[i])
            failed = False
            break
        except:
            failed = True
    if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
    # may use `from multiprocessing import Pool` instead, but less efficient and
    # NOTE: `multiprocessing.Pool` will duplicate given object for each process.
    from multiprocessing.dummy import Pool
    from tqdm import tqdm
    print('Starting to load images via multiple imreaders')
    pool = Pool() # use all threads by default
    for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
        pass
    pool.close()
    pool.join()

if __name__ == '__main__':
    pass
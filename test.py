import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy


if __name__ == '__main__':
    opt = TestOptions().parse()

    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, 'test', opt)
        datasets[dataset_name] = tqdm(dataset)

    visualizer = Visualizer(opt)
    model = create_model(opt)

    for load_iter in load_iters:
        opt.load_iter = load_iter
        model.setup(opt)
        model.eval()

        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name)
            tqdm_val.reset()

            time_val = 0
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                model.set_input(data)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()

                if opt.save_imgs:
                    folder_dir = './ckpt/%s/sr_output_%s' % (opt.name, opt.load_iter)  
                    os.makedirs(folder_dir, exist_ok=True)

                    if opt.camera == 'nikon':
                        save_dir = '%s/%s' % (folder_dir, data['fname'][0])
                    elif opt.camera == 'iphone':
                        img_path = data['fname'][0].split('_')
                        os.makedirs(folder_dir + '/' + img_path[0], exist_ok=True)
                        save_dir = '%s/%s/%s' % (folder_dir, img_path[0], img_path[1])
                    
                    pad = data['paddings'][0]
                    img = np.array(res['data_sr'][0].cpu()).astype(np.uint8)[:, pad[0]:pad[1], pad[2]:pad[3]]
                    dataset_test.imio.write(img, save_dir)
                        
            print('Time: %.3f s AVG Time: %.3f ms Epoch: %s\n' % (time_val, time_val/dataset_size_test*1000, load_iter))

    for dataset in datasets:
        datasets[dataset].close()

import numpy as np
from os.path import join
from tensorboardX import SummaryWriter
from functools import partial
from functools import wraps
import time

def write_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(30):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                print('%s OSError' % str(args))
                time.sleep(1)
        return ret
    return wrapper

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        if opt.isTrain:
            self.name = opt.name
            self.writer = SummaryWriter(logdir=join(self.save_dir, 'log'))
        else:
            self.name = opt.name
            self.save_dir = join(opt.checkpoints_dir, opt.name, 'test_log', opt.camera)
            self.writer = SummaryWriter(logdir=join(self.save_dir))

    @write_until_success
    def display_current_results(self, phase, visuals, iters):
        for k, v in visuals.items():
            v = v.cpu()
            # if k == 'pred':
            #     self.process_preds(self.writer, phase, k, v, iters)
            # else:
            self.writer.add_image('%s/%s'%(phase, k), v[0]/255, iters)
        self.writer.flush()

    # def process_pred(self, pred):
    #     buffer = BytesIO()
    #     plt.figure(1)
    #     plt.clf()
    #     plt.axis('off')
    #     img = plt.imshow(pred, cmap=plt.cm.hot)
    #     plt.colorbar()
    #     plt.savefig(buffer)
    #     im = np.array(Image.open(buffer).convert('RGB')).transpose(2, 0, 1)
    #     buffer.close()
    #     return im / 255

    # def process_preds(self, writer, phase, k, v, iters):
    #     preds = v[0]
    #     if len(preds) == 1:
    #         writer.add_image('%s/%s'%(phase, k),
    #                          self.process_pred(preds[0]),
    #                          iters)
    #     else:
    #         writer.add_images('%s/%s'%(phase, k),
    #                           np.stack([self.process_pred(pred)\
    #                                        for pred in preds]),
    #                           iters)

    @write_until_success
    def print_current_losses(self, epoch, iters, losses,
                             t_comp, t_data, total_iters):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' \
                  % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.4e ' % (k, v)
            self.writer.add_scalar('loss/%s'%k, v, total_iters)
        print(message)
    
    @write_until_success
    def print_psnr(self, epoch, total_epoch, time_val, mean_psnr, print_psnr=True):
        self.writer.add_scalar('val/psnr', mean_psnr, epoch)
        if print_psnr:
            print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t PSNR: %f'
                    % (epoch, total_epoch, time_val, mean_psnr))



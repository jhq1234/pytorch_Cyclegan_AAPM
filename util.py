import os
import numpy as np
from scipy.stats import poisson
import torch
import torch.nn as nn
from skimage.transform import rescale, resize

##네트워크 grad 설정하기
def set_required_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

## 대표적인 initialization method normal distribution, xavier, kaiming, orthogonal
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname=m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

##

## 네트워크 저장하기
def save(ckpt_dir, Unet, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'Unet': Unet.state_dict(), 'optim': optim.state_dict(),},
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, Unet, optim):
    if not os.path.exists(ckpt_dir):
        epoch =0
        return Unet, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int('100'.join(filter(str.isdigit, f))))

    # dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    # dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[i]))
    # i에 원하는 epoch을 입력한다.
    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    Unet.load_state_dict(dict_model['Unet'])
    optim.load_state_dict(dict_model)['optim']
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return Unet, optim, epoch

## Sampling 추가하기
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(sz)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk
    elif type == "random":
        prob = opts[0]
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd > prob).astype(np.float)

        # rnd = np.random.rand(sz[0], sz[1], 1)
        # prob = 0.5
        # msk = (rnd > prob).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        dst = img * msk
    elif type == "gaussian":
        # x0, y0은 center이다.
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]
        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)  # meshgrid로 x,y 생성

        gaus = a * np.exp(-((x-x0)**2/(2*sgmx**2)+(y-y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)

        # gaus = a * np.exp(-((x - x0) ** 2 / (2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
        # gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < gaus).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        dst = img * msk

    return dst

## Noise 추가하기
def add_noise(img, type="random", opts=None):
    sz = img.shape

    if type == "random":
        sgm = opts[0].astype(np.int)
        noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[2])

        dst = img + noise

    elif type == "poisson":
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    return dst

## Blurring 추가하기
def add_blur(img, type="bilinear", opts=None):
    if type == "nearest":
        order = 0
    elif type == "bilinear":
        order = 1
    elif type == "biquadratic":
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "biquintic":
        order = 5

    ## resolution scale
    sz = img.shape

    dw = opts[0]

    if len(opts) == 1:
        keepdim = True
    else:
        keepdim = opts[1]

    # dst = rescale(img, scale=(dw, dw, 1), order=order)
    dst = resize(img, output_shape=(sz[0] // dw, sz[1] // dw, sz[2]), order=order)

    if keepdim:
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst
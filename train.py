## 라이브러리 추가하기
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import itertools

from torchvision import transforms, datasets
from model import *
from dataset import *
from util import *

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def train(args):
    ## 트레이닝에 필요한 하이퍼파라미터 설정
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    info_dir = args.info_dir
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir
    norm = args.norm

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    if args.mode == 'client':
        mode = 'train'
    else:
        mode = args.mode
    train_continue = args.train_continue

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker
    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)
    print("task: %s" % task)
    print("opts: %s" % opts)
    print("network: %s" % network)
    print("learning type: %s" % learning_type)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)
    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir_train, 'png'))
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.join('./','checkpoint'))

    ## 네트워크 Transformer, Dataset, DataLoader 적용
    if mode == 'train':
        ## Jittering technique 적용
        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'),
                                transform=None,
                                data_type='both')
        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)

    ## 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    ## 네트워크 생성하기
    if network =="Unet":
        Unet = UNet(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        # init_weights(Unet, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    fn_loss = nn.MSELoss().to(device)
    
    ## Optimizer 설정하기
    optim = torch.optim.Adam(Unet.parameters(), lr=lr)
    
    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    ## Tenasorboard를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    ## 신경망 병렬처리 구성하기
    Unet = nn.DataParallel(Unet, device_ids=[0, 1])
    Unet.cuda()

    start = time.time()

    # 네트워크 학습시키기
    st_epoch = 0
    if mode == 'train':
        if train_continue == "on":       
            Unet, optim, st_epoch = load(ckpt_dir=ckpt_dir, Unet=Unet, optim=optim)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            Unet.train()

            loss_Unet = []
            loss_Unet_train = []

            start = time.time()

            for batch, data in enumerate(loader_train, 1):
                # forward pass
                input_a = data['data_a'].to(device)
                input_b = data['data_b'].to(device)

                # forward Unet
                output = Unet(input_a)

                # Backward Unet
                loss_Unet = fn_loss(output, input_b)
                loss_Unet.backward()
                optim.step()

                # Backward netD_a
                input_a_p1 = input_a[0].permute(1,2,0).cpu().numpy()
                input_b_p1 = input_b[0].permute(1,2,0).cpu().numpy()
                output_p1 = output[0].permute(1,2,0).cpu().detach().numpy()

                PSNR_input_GT = peak_signal_noise_ratio(input_a_p1, input_b_p1)
                PSNR_input_output = peak_signal_noise_ratio(output_p1, input_b_p1)
                SSIM_input_GT = structural_similarity(input_a_p1, input_b_p1, multichannel=True)
                SSIM_input_output = structural_similarity(output_p1, input_b_p1, multichannel=True)

                # 손실함수 계산
                # 1. Generator의 손실함수
                # 2. Discriminator에 대해 real part의 손실함수
                # 3. Discriminator에 대해 fake part의 손실함수

                loss_Unet_train += [loss_Unet.item()]

                end = time.time()
                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d |"
                      "Loss : %.4f " %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_Unet_train)))

                if batch % 10 == 0:
                    writer_train.add_scalar('PSNR_GT', PSNR_input_GT, batch + (epoch-1*batch))
                    writer_train.add_scalar('PSNR_output', PSNR_input_output, batch + (epoch-1*batch))
                    writer_train.add_scalar('SSIM_GT', SSIM_input_GT, batch + (epoch-1*batch))
                    writer_train.add_scalar('SSIM_output', SSIM_input_output, batch + (epoch-1*batch))
                    print("PSNR score : GT %.4f Output %.4f |"
                          "SSIM score : GT %.4f Output %.4f " %
                          (PSNR_input_GT, PSNR_input_output,
                          SSIM_input_GT, SSIM_input_output))
                    

                if batch % 50 == 0:
                    print("TRAIN TIME for 50 Batch is : %d" %(end-start))
                    start = time.time()
                    # Tensorboard 저장하기
                    # input_a = fn_tonumpy(fn_denorm(input_a, mean=0.5, std=0.5))
                    # input_b = fn_tonumpy(fn_denorm(input_b, mean=0.5, std=0.5))
                    # output_a = fn_tonumpy(fn_denorm(output_a, mean=0.5, std=0.5))
                    # output_b = fn_tonumpy(fn_denorm(output_b, mean=0.5, std=0.5))

                    input_a = fn_tonumpy(input_a)
                    input_b = fn_tonumpy(input_b)
                    output = fn_tonumpy(output)

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input_a.png' % id), input_a[0].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input_b.png' % id), input_b[0].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output_a.png' % id), output[0].squeeze(), cmap='gray')

                    writer_train.add_image('input_a', input_a, id, dataformats='NHWC')
                    writer_train.add_image('input_b', input_b, id, dataformats='NHWC')
                    writer_train.add_image('output_a', output, id, dataformats='NHWC')

            print('1 Epoch cost time : %f s', start - time.time())   
            writer_train.add_scalar('loss_Unet', np.mean(loss_Unet_train), epoch)

            # writer_train.add_scalar('loss_G_a2b', np.mean(loss_G_a2b_train), epoch)
            # writer_train.add_scalar('loss_G_b2a', np.mean(loss_G_b2a_train), epoch)
            # writer_train.add_scalar('loss_D_a', np.mean(loss_D_a_train), epoch)
            # writer_train.add_scalar('loss_D_b', np.mean(loss_D_b_train), epoch)
            # writer_train.add_scalar('loss_cycle_a', np.mean(loss_cycle_a_train), epoch)
            # writer_train.add_scalar('loss_cycle_b', np.mean(loss_cycle_b_train), epoch)
            # writer_train.add_scalar('loss_ident_a', np.mean(loss_ident_a_train), epoch)
            # writer_train.add_scalar('loss_ident_b', np.mean(loss_ident_b_train), epoch)
            # if epoch % 1 == 0:
            save(ckpt_dir=ckpt_dir, Unet=Unet, optim=optim, epoch=epoch)

        writer_train.close()

def test(args):
    ## 트레이닝에 필요한 하이퍼파라미터 설정
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    info_dir = args.info_dir
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir
    task = args.task
    mode = args.mode
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    train_continue = args.train_continue

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)
    print("task: %s" % task)
    print("opts: %s" % opts)
    print("network: %s" % network)
    print("learning type: %s" % learning_type)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)
    print("device: %s" % device)


    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))
        os.makedirs(os.path.join(result_dir_test, 'numpy'))


    ## 네트워크 Test
    if mode == 'test':
        transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, data_type='both')
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

        # num_data_test_a = len(dataset_test_a)
        # num_batch_test_a = np.ceil(num_data_test_a / batch_size)

        # # dataset_test_b = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, data_type='b')
        # # loader_test_b = DataLoader(dataset_test_b, batch_size=batch_size, shuffle=False, num_workers=8)

        # num_data_test_b = len(dataset_test_b)
        # num_batch_test_b = np.ceil(num_data_test_b / batch_size)

    ## 네트워크 생성하기
    if network =="cyclegan":
        Unet = UNet(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)

        init_weights(Unet, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    fn_loss = nn.L1Loss().to(device)

    ## Optimizer 설정하기
    optim = torch.optim.Adam(Unet.parameters(), lr=lr)

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    # fn_class = lambda x: 1.0 * (x > 0.5)

    ## 네트워크 학습시키기
    st_epoch = 0
    start = time.time()
    Unet = nn.DataParallel(Unet, device_ids=[0,1])
    Unet.cuda()

    st_epoch = 0

    score_writer = open(result_dir+info_dir, 'w')
    score_writer.write("PSNR, SSIM score Document \n")

    total_PSNR_GT = 0
    total_PSNR_output = 0
    total_SSIM_GT = 0
    total_SSIM_output = 0

    counter = 0

    #TEST MODE
    if mode == "test":
        Unet, optim, st_epoch = load(ckpt_dir=ckpt_dir,
                                        Unet=Unet, optim=optim)

        with torch.no_grad():
            Unet.eval()

            start = time.time()
            for batch, data in enumerate(loader_test, 1):
                # forward pass
                input_a = data['data_a'].to(device)
                input_b = data['data_b'].to(device)
                output = Unet(input_a)

                for i in range(0,batch_size):
                    
                    id = batch_size * (batch - 1) + i

                    if id > 420:
                        break
    
                    input_a_p = input_a[i].permute(1,2,0).cpu().numpy()
                    input_b_p = input_b[i].permute(1,2,0).cpu().numpy()
                    output_p = output[i].permute(1,2,0).cpu().detach().numpy()

                    PSNR_input_GT = peak_signal_noise_ratio(input_a_p, input_b_p)
                    PSNR_input_output = peak_signal_noise_ratio(output_p, input_b_p)
                    SSIM_input_GT = structural_similarity(input_a_p, input_b_p, multichannel=True)
                    SSIM_input_output = structural_similarity(output_p, input_b_p, multichannel=True)

                    total_PSNR_GT += PSNR_input_GT
                    total_PSNR_output += PSNR_input_output
                    total_SSIM_GT += SSIM_input_GT
                    total_SSIM_output += SSIM_input_output
                    counter += 1

                    print("PSNR score : GT %.4f Output :  %.4f |"
                          "SSIM score : GT %.4f Output :  %.4f "
                          "ID : %d " %
                          (PSNR_input_GT, PSNR_input_output,
                          SSIM_input_GT, SSIM_input_output, id))

                    score_writer.write("PSNR score : GT %.4f Output : %.4f |"
                                        "SSIM score : GT %.4f Output : %.4f "
                                        "ID : %d \n" %
                                        (PSNR_input_GT, PSNR_input_output,
                                        SSIM_input_GT, SSIM_input_output, id))

                # Tensorboard 저장하기
                input_a = fn_tonumpy(input_a)
                input_b = fn_tonumpy(input_b)
                output = fn_tonumpy(output)
                
                for j in range(input_a.shape[0]):
                    id = batch_size * (batch - 1) + j

                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input_a.png' % id), input_a[j].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input_b.png' % id), input_b[j].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output_a.png' % id), output[j].squeeze(), cmap='gray')

                    print('TEST: BATCH %04d / %04d | ' % (id + 1, num_data_test))

            print("Average PSNR GT score : %.4f Output : %.4f |"
                    "Average SSIM GT score: %.4f Output : %.4f" %
                    (total_PSNR_GT/counter, total_PSNR_output/counter,
                    total_SSIM_GT/counter, total_SSIM_output/counter))

            score_writer.write("Average PSNR GT score : %.4f Output : %.4f |"
                                "Average SSIM GT score: %.4f Output : %.4f" %
                                (total_PSNR_GT/counter, total_PSNR_output/counter,
                                total_SSIM_GT/counter, total_SSIM_output/counter))

            score_writer.close()
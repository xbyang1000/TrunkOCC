from __future__ import print_function
import os
import scipy.io as scio
import time
# from openpyxl import Workbook, load_workbook
# import pandas as pd
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
import xlrd
import xlwt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import transforms
from xlutils import copy

from edgeml_pytorch.trainer.drocc_trainer import DROCCTrainer
# from img_process import img_merge, cut_image, border_img, drawPic

from resnet import MyResNet





def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
    """Adjust learning rate during training.

    Parameters
    ----------
    epoch: Current training epoch.
    total_epochs: Total number of epochs for training.
    only_ce_epochs: Number of epochs for initial pretraining.
    learning_rate: Initial learning rate for training.
    """
    # We dont want to consider the only ce
    # based epochs for the lr scheduler
    epoch = epoch - only_ce_epochs
    drocc_epochs = total_epochs - only_ce_epochs
    # lr = learning_rate
    if epoch <= drocc_epochs:
        lr = learning_rate * 0.01
    if epoch <= 0.80 * drocc_epochs:
        lr = learning_rate * 0.1
    if epoch <= 0.40 * drocc_epochs:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def main():
    train_datadir = '/mnt/flameimg/DROCCtrain'
    test_datadir = '/mnt/flameimg/DROCCvideo'

    # train_datadir = r'F:\videofireimg\fire\train'
    # test_datadir = r'F:\videofireimg\fire\valid'

    # train_datadir = r'F:\videofireimg\cropvideo2-onecls\ten_ten\train'
    # test_datadir = r'F:\videofireimg\cropvideo2-onecls\ten_ten\test'
    # normMean = [0.61707443, 0.41926995, 0.34889632]
    # normStd = [0.214335, 0.17251974, 0.14073351]
    # 准备工作
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # 将图片转换成Tensor，归一化至[0,1]
    #     # 标准化至[-1,1],规定均值和标准差
    #     transforms.Normalize(mean=normMean, std=normStd)
    # ])

    train_data = torchvision.datasets.ImageFolder(train_datadir, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.ImageFolder(test_datadir, transform=torchvision.transforms.ToTensor())

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集长度为{}".format(train_data_size))  # fire为0,nofire为1
    print("测试集长度为{}".format(test_data_size))
    size=4

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)


    model3 = MyResNet().to(device)
    model3 = nn.DataParallel(model3)  # main中超参数存入模型

    if args.optim == 1:
        optimizer = optim.SGD(model3.parameters(), lr=args.lr, momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model3.parameters(), lr=args.lr)
        print("using Adam")

    # trainer = DROCCTrainer(model, optimizer, args.lamda, args.radius, args.gamma, device)
    trainer3 = DROCCTrainer(model3, optimizer, args.lamda, args.radius, args.gamma, device)

    # if os.path.exists(os.path.join(args.model_dir, 'model-AUC.pt')):
    if os.path.exists(os.path.join(args.model_dir, 'model.pt')):
        # trainer.load(args.model_dir,'model-AUC.pt')
        print("model-AUC.pt Saved Model Loaded***********************************")
    else:
        print("args.lr", args.lr)
        print("args.radius", args.radius)
        print("args.gamma", args.gamma)
        print("model architecture: ", model3)
        trainer3.train(size, train_loader, test_loader, args.lr, adjust_learning_rate, args.epochs,
                       metric=args.metric, ascent_step_size=args.ascent_step_size, only_ce_epochs=0,
                       model_dir=args.model_dir)
        trainer3.load(args.model_dir, 'cropvideo4-modelPlus-AUC300.pty')

        # trainer3.save(args.model_dir,metric=args.metric)

    # if os.path.exists(os.path.join(args.model_dir, '4px-modelPlus-AUC0.pt')):  # 19,20，21，22
    #     trainer3.load(args.model_dir, '4px-modelPlus-AUC0.pt')
    #     print("4px-modelPlus-AUC1.pt Saved Model Loaded***************************************")
    #     res = trainer3.test(test_loader, 'AUC')
    #     # # thresh=res[1]+0.0035
    #     thresh = res[1]+0.5
        # print("AUC", res[0])



    # drawPic(TPbox,FPbox,index)







if __name__ == '__main__':
    torch.set_printoptions(precision=5)

    parser = argparse.ArgumentParser(description='PyTorch Simple Training')
    parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                        help='CIFAR10 normal class index')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('-oce,', '--only_ce_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train with only CE loss')
    parser.add_argument('--ascent_num_steps', type=int, default=100, metavar='N',
                        help='Number of gradient ascent steps')
    # parser.add_argument('--hd', type=int, default=128, metavar='N',
    #                     help='Num hidden nodes for LSTM model')
    parser.add_argument('--lr', type=float, default=0.0008, metavar='LR',
                        help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.0001, metavar='LR',
                        help='step size of gradient ascent')
    parser.add_argument('--mom', type=float, default=0.0, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='log',
                        help='path where to save checkpoint')
    # parser.add_argument('--model_dir', default='temp',
    #                     help='path where to save checkpoint')
    parser.add_argument('--one_class_adv', type=int, default=1, metavar='N',
                        help='adv loss to be used or not, 1:use 0:not use(only CE)')
    parser.add_argument('--radius', type=float, default=18, metavar='N',
                        help='radius corresponding to the definition of set N_i(r)')
    parser.add_argument('--lamda', type=float, default=1, metavar='N',
                        help='Weight to the adversarial loss')
    parser.add_argument('--reg', type=float, default=0, metavar='N',
                        help='weight reg')
    parser.add_argument('--eval', type=int, default=0, metavar='N',
                        help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('--optim', type=int, default=1, metavar='N',
                        help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=1, metavar='N',
                        help='r to gamma * r projection for the set N_i(r)')
    parser.add_argument('-d', '--data_path', type=str, default='.')
    # parser.add_argument('--metric', type=str, default='F1')
    parser.add_argument('--metric', type=str, default='AUC')
    args = parser.parse_args()

    # settings
    # Checkpoint store path
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    main()

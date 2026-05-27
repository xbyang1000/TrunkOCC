from __future__ import print_function
import os

import joblib
import scipy.io as scio
import time
# from openpyxl import Workbook, load_workbook
import pandas as pd
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
from img_process import img_merge, cut_image, border_img, drawPic

from resnet import MyResNet


class TREE_LeNet(nn.Module):

    def __init__(self):
        super(TREE_LeNet, self).__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(6272, self.rep_dim, bias=False)#输入为2048
        self.fc2 = nn.Linear(self.rep_dim, int(self.rep_dim / 2), bias=False)
        self.fc3 = nn.Linear(int(self.rep_dim / 2), 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TREE_LeNetPlus1D(nn.Module):

    def __init__(self,size):
        super(TREE_LeNetPlus1D, self).__init__()
        # self.pool = nn.MaxPool1d(2, 2)
        # self.conv1 = nn.Conv1d(1, 16, 3, bias=False, padding=1)
        # self.bn2d1 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv1d(16, 32, 3, bias=False, padding=1)
        # self.bn2d2 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        # self.conv3 = nn.Conv1d(32, 64, 3, bias=False, padding=1)
        # self.bn2d3 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        # self.conv4 = nn.Conv1d(64, 128, 3, bias=False, padding=1)
        # self.bn2d4 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(2, 70, bias=False)
        self.lstm = nn.LSTM(input_size=70, hidden_size=20, batch_first=True)
        self.fc3 = nn.Linear(20, 1, bias=False)


    def forward(self, x):
        # x = self.conv1(x)
        # x = self.pool(F.leaky_relu(self.bn2d1(x)))
        # x = self.conv2(x)
        # x = self.pool(F.leaky_relu(self.bn2d2(x)))
        # x = self.conv3(x)
        # x = self.pool(F.leaky_relu(self.bn2d3(x)))
        # x = self.conv4(x)
        x=self.fc1(x)

        # 检查尺寸
        batch_size = x.size(0)
        if batch_size == 0:
            return torch.empty(0, 1)
        x = x.view(x.size(0), -1)
        x, _ = self.lstm(x)
        nn.Sigmoid()
        x = self.fc3(x)
        return x


class TREE_LeNetPlus(nn.Module):

    def __init__(self):
        super(TREE_LeNetPlus, self).__init__()

        # self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        # self.conv1 = nn.Conv2d(3, 32, 3, bias=False, padding=2)
        self.conv1 = nn.Conv2d(3, 32, 3, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 256, 3, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)


        # self.fc1 = nn.Linear(6272, int(self.rep_dim / 2), bias=False)#输入为2048  #56px时删除
        # self.fc2 = nn.Linear(self.rep_dim, int(self.rep_dim / 2), bias=False)  #56px时删除
        # if size==4:
        # self.fc3 = nn.Linear(12544, 1, bias=False)
        self.fc3 = nn.LazyLinear(1, bias=False)
            # self.fc3 = nn.Linear(2048, 1, bias=False)
        # print("size=4")
        # elif size==10:
        #     self.fc3 = nn.Linear(9216, 1, bias=False)
        #     print("size=10")

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        # x = self.pool(F.leaky_relu(self.bn2d3(x)))  #10px时删除

        x = self.conv4(x)
        # x = self.pool(F.leaky_relu(self.bn2d4(x)))  #10px时删除

        '''展成x.size(0)行，自适应列。x.size(0)指batchsize，比如：二分类中batchsize=128,展成128行9216列，nn.Linear参数为9216行2列
        最后输出128行2列，即128个batch，每个batch包含两个类别的概率(如果激活函数是SIGMOD，softmax等)，在本实验中，real time val
        一个块图一个batch,即batchsize=1，x=x.view(x.size(0), -1)后即为行向量，fc3(x)变为一个值，最后的output为leaky-relu后的结果'''
        x = x.view(x.size(0), -1)

        # x = F.leaky_relu(self.fc1(x))  #10px时删除
        # x = F.leaky_relu(self.fc2(x))  #10px时删除
        x = F.leaky_relu(self.fc3(x))
        return x

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
    if epoch <= 0.40 * drocc_epochs:
        lr = learning_rate * 0.1
    if epoch <= 0.10 * drocc_epochs:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def main():
    train_datadir = r'E:\oneclass\train'
    test_datadir = r'E:\oneclass\test'

    # train_datadir = r'F:\videofireimg\cropvideo2-onecls\ten_ten\train'
    # test_datadir = r'F:\videofireimg\cropvideo2-onecls\ten_ten\test'
    # normMean = [0.61707443, 0.41926995, 0.34889632]
    # normStd = [0.214335, 0.17251974, 0.14073351]
    # # 准备工作
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # 将图片转换成Tensor，归一化至[0,1]
    #     # 标准化至[-1,1],规定均值和标准差
    #     transforms.Normalize(mean=normMean, std=normStd)
    # ])

    train_data = torchvision.datasets.ImageFolder(train_datadir, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.ImageFolder(test_datadir, transform=torchvision.transforms.ToTensor())

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集长度为{}".format(train_data_size))#fire为0,nofire为1
    print("测试集长度为{}".format(test_data_size))
    size=8


    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,shuffle=True)


    # model= FIRE_LeNet().to(device)
    # model = nn.DataParallel(model)
    # model=MyResNet().to(device)
    # model = nn.DataParallel(model)
    model3 = TREE_LeNetPlus().to(device)
    model3 = nn.DataParallel(model3)#main中超参数存入模型


    if args.optim == 1:
        optimizer = optim.SGD(model3.parameters(),lr=args.lr,momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model3.parameters(),lr=args.lr)
        print("using Adam")

    # trainer = DROCCTrainer(model, optimizer, args.lamda, args.radius, args.gamma, device)
    trainer3=DROCCTrainer(model3, optimizer, args.lamda, args.radius, args.gamma, device)

    if os.path.exists(os.path.join(args.model_dir, 'model-AUC.pt')):
    # if os.path.exists(os.path.join(args.model_dir, 'model.pt')):
        # trainer.load(args.model_dir,'model-AUC.pt')
        print("No Train***********************************")
    else:
        print("args.lr", args.lr)
        print("args.radius", args.radius)
        print("args.gamma", args.gamma)
        trainer3.train(size,train_loader, test_loader, args.lr, adjust_learning_rate, args.epochs,
                      metric=args.metric, ascent_step_size=args.ascent_step_size, only_ce_epochs=0,model_dir=args.model_dir)
        # trainer3.load(args.model_dir, 'cropvideo4-modelPlus-AUC300.pty')
        # trainer3.save(args.model_dir,metric=args.metric)


    if os.path.exists(os.path.join(args.model_dir, '4px-STMDA_2.pt')):
        trainer3.load(args.model_dir,'4px-STMDA_2.pt')
        print("Model Loaded***************************************")
        res= trainer3.test(test_loader, 'AUC')
        thresh = res[1]
        print("AUC:", res[0])



# 图片边缘预处理
#     s = 'first'
    # path = r'/mnt/videofireimg/cropvideo2-onecls/threepic/fire{}_video3.png'.format(s)
    # border_img(path,s)

# 分割图片
#     path=r'/mnt/videofireimg/cropvideo2-onecls/threepic/tenten_{}_video3.png'.format(s)
#     cutsize=10
#     storagePath=r'/mnt/videofireimg/cropvideo2-onecls/ten_ten/Video3val/{}'.format(s)
#     cut_image(path,cutsize,storagePath,s)

# 验证
#     path_src=r'/mnt/videofireimg/cropvideo2-onecls/four_four/val/{}/'.format(s)
#     fire_index=trainer3.val(path_src,s,thresh)
#     print(fire_index)
# 合并图片
#     path=r'/mnt/videofireimg/cropvideo2-onecls/four_four/val/{}'.format(s)
#     img_merge(path,fire_index,s)


# 实时验证
    cutsize=4

    path = r'C:\Users\Lenovo\Desktop\OneClassDemo-main\flamechose328label.mat'
    # matdata = scio.loadmat(path)
    import h5py
    matdata = h5py.File(path)
    # Label= np.transpose(np.array(matdata.get('B')))
    Label = np.transpose(np.array(matdata.get('flamechose328label')))
    # data=pd.read_csv(r'/mnt/videofireimg/flamevideo-onecls/four_four/val/label25s_3min22.csv')
    # wb = load_workbook(r'/mnt/videofireimg/flamevideo-onecls/four_four/val/label25s_3min22.xlsx', read_only=True)
    # read = xlrd.open_workbook(r'/mnt/videofireimg/flamevideo-onecls/four_four/val/label25s_3min22.xlsx')
    # sheet1 = read.sheet_by_index(0)
    # sheet1 = wb.worksheets[0]
    # Label=data.values

    row = Label.shape[0]
    col = Label.shape[1]
    print(row, col)
    # Label = []
    # timebox=[]
    TPbox=[]
    FPbox=[]
    aucbox=[]
    # for y in range(col):
    #     Label.append([])

    # for y in range(col):
    #     for x in range(row):
            # if sheet1.cell_value(x, y) != '':
            # if sheet1.cell(x+1, y+1) != '':
                # t = sheet1.cell_value(x, y)
                # t = sheet1.cell(x+1, y+1)
                # Label[y].append(sheet1.cell(x+1, y+1).value)
            # print("pp")
        # Label[y]=np.array(Label[y])

    # cap = cv2.VideoCapture(r'E:\OneClassDemo\fire8.avi')
    index = np.array([i for i in range(1,5219,16)])
    # condition = [i for i in range(581,716,5)]

    index = np.hstack((index, 5219))
    # index = [1,786,1446]
    # print('总帧数：',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    framenum=5219
    # # for k in range(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))+1,3):#原先第三个参数为3,第二个参数不用加一
    # for k in range(1, framenum + 1, 1):
    #     # if k+2<=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
    #     #     index.append(k+1)
    #     #     index.append(k + 2)
    #     # elif k+1<=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
    #     #     index.append(k + 1)
    #     index.append(k)

    print('index len:',len(index))

    f =1
    i=0
    num = 1
    t = 0
    while True:
        # 逐帧读取视频
        # ret, frame = cap.read()
        # if f in condition:
        #     f = f + 1
        #     print("pp")
        #     continue
        frame=cv2.imread(r'E:\oneclass\1.png')
        # if not ret:
        #     break
        if f>framenum:
            break
        if f in index:
            print(f,i)

            res = trainer3.real_time_val(f, thresh, cutsize, frame)
            # res = trainer3.real_time_val2(f, test_loader, cutsize, frame)

            # tempval_dir = '/mnt/videofireimg/flamevideo-onecls/four_four/tempval'
            # val_data = torchvision.datasets.ImageFolder(tempval_dir, transform=torchvision.transforms.ToTensor())
            # val_data_size = len(val_data)
            # print("val长度为{}".format(val_data_size))
            # val_loader = DataLoader(val_data, batch_size=128)
            # res = trainer3.test2(val_loader, 'AUC',thresh)
            # # timebox.append(res[3])
            print("运行时间{}".format(res[0]))

            y_pred = np.array(res[1])
            # Label[i].append(sheet1.cell(x + 1, y + 1))
            # print("误警率:", np.sum(y_pred[np.where(Label[i] == 1)] == 0) / np.sum(Label[i] == 1))
            # FPbox.append(np.sum(y_pred[np.where(Label[i] == 1)] == 0) / np.sum(Label[i] == 1)*100)
            # print("火焰检测率:", np.sum(y_pred[np.where(Label[i] == 0)] == 0) / np.sum(Label[i] == 0))
            # TPbox.append(np.sum(y_pred[np.where(Label[i] == 0)] == 0) / np.sum(Label[i] == 0)*100)

            auc_value = roc_auc_score(Label[:,i:i+1], y_pred)
            aucbox.append(auc_value)
            print("误警率:", np.sum(y_pred[np.where(Label[:,i:i+1] == 1)[0]] == 0) / np.sum(Label[:,i:i+1] == 1))
            FPbox.append(np.sum(y_pred[np.where(Label[:,i:i+1] == 1)[0]] == 0) / np.sum(Label[:,i:i+1] == 1) * 100)
            print("树干检测率:", np.sum(y_pred[np.where(Label[:,i:i+1] == 0)[0]] == 0) / np.sum(Label[:,i:i+1] == 0))
            TPbox.append(np.sum(y_pred[np.where(Label[:,i:i+1] == 0)[0]] == 0) / np.sum(Label[:,i:i+1] == 0) * 100)
            print("***********************")
            print("AUC", aucbox)
            print("FP", FPbox)
            print("TP",TPbox)
            path = r'E:\oneclass\temp.xls'
            xls = xlrd.open_workbook(path)  #得到文件
            workbook = copy.copy(xls)  # 复制文件并保留格式
            worksheet = workbook.get_sheet(0)  # 打开表单
            # workbook = xlwt.Workbook(encoding='utf-8')
            # worksheet = workbook.add_sheet('sheet1')
            if num == 2:
                for x in range(2):
                    worksheet.write(t, 0, TPbox[x])
                    worksheet.write(t, 1, FPbox[x])
                    worksheet.write(t, 2, aucbox[x])
                    t = t + 1
                workbook.save(path)
                num = 1
                TPbox = []
                FPbox = []
                aucbox=[]
            else:
                num = num + 1
            print("-------------------------------------")
            i=i+1
        # i = i + 191#flame
        f = f + 1
    # 释放资源并关闭窗口
    # cap.release()
    cv2.destroyAllWindows()

    # drawPic(TPbox,FPbox,index)

    # path=r'F:\videofireimg\flamevideo-twocls\val\Ghostres.xls'
    # # 创建可写的workbook对象
    # workbook = xlwt.Workbook(encoding='utf-8')
    # # 创建工作表sheet
    # worksheet = workbook.add_sheet('sheet1')
    #
    # for x in range(len(TPbox)):
    #     worksheet.write(x, 0, TPbox[x])
    #     worksheet.write(x, 1, FPbox[x])
    # # 保存表
    # workbook.save(path)







    


if __name__ == '__main__':
    torch.set_printoptions(precision=5)

    parser = argparse.ArgumentParser(description='PyTorch Simple Training')
    parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                        help='CIFAR10 normal class index')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('-oce,', '--only_ce_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train with only CE loss')
    parser.add_argument('--ascent_num_steps', type=int, default=100, metavar='N',
                        help='Number of gradient ascent steps')
    # parser.add_argument('--hd', type=int, default=128, metavar='N',
    #                     help='Num hidden nodes for LSTM model')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.03, metavar='LR',
                        help='step size of gradient ascent')
    parser.add_argument('--mom', type=float, default=0.0, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='log',
                        help='path where to save checkpoint')
    # parser.add_argument('--model_dir', default='temp',
    #                     help='path where to save checkpoint')
    parser.add_argument('--one_class_adv', type=int, default=1, metavar='N',
                        help='adv loss to be used or not, 1:use 0:not use(only CE)')
    parser.add_argument('--radius', type=float, default=0.1, metavar='N',
                        help='radius corresponding to the definition of set N_i(r)')
    parser.add_argument('--lamda', type=float, default=1, metavar='N',
                        help='Weight to the adversarial loss')
    parser.add_argument('--reg', type=float, default=0, metavar='N',
                        help='weight reg')
    parser.add_argument('--eval', type=int, default=0, metavar='N',
                        help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('--optim', type=int, default=1, metavar='N',
                        help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=2, metavar='N',
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

import os
import copy
import time
import cv2
import joblib
import matplotlib
from sklearn.cluster import KMeans
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact

matplotlib.use('TkAgg')
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import xlrd
import xlwt
from sklearn import svm
from torch.utils.data import DataLoader
from xlutils.copy import copy as xlcopy
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support,accuracy_score
from edgeml_pytorch.trainer.drocclf_trainer import DROCCLFTrainer
from edgeml_pytorch.trainer.drocclf_trainer import get_gradients


def saveToExcel(celoss,advloss,epoch,trtime):
    path = r'F:\videofireimg\ST-MDA\temp.xls'
    if os.path.exists(path):
        read = xlrd.open_workbook(path)
        write_workbook=xlcopy(read)
        worksheet=write_workbook.get_sheet(0)
        # worksheet = read.sheet_by_index(0)
    else:
        # 创建可写的workbook对象
        write_workbook = xlwt.Workbook(encoding='utf-8')
        # 创建工作表sheet
        worksheet = write_workbook.add_sheet('sheet1')

    worksheet.write(epoch, 0, celoss)
    worksheet.write(epoch, 1, advloss)
    worksheet.write(epoch, 2, trtime)
    # worksheet.write(epoch, 2, auc)
    # worksheet.write(epoch, 1, advloss)
    # 保存表
    write_workbook.save(path)


#trainer class for DROCC
class DROCCTrainer:
    """
    Trainer class that implements the DROCC algorithm proposed in
    https://arxiv.org/abs/2002.12718
    """

    def __init__(self, model, optimizer, lamda, radius, gamma, device):
        """Initialize the DROCC Trainer class

        Parameters
        ----------
        model: Torch neural network object
        optimizer: Total number of epochs for training.
        lamda: Weight given to the adversarial loss，对抗性损失的权重
        radius: Radius of hypersphere to sample points from.
        gamma: Parameter to vary projection.
        device: torch.device object for device to use.
        """     
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.device = device

    def train(self, size,train_loader, val_loader, learning_rate, lr_scheduler, total_epochs,
                only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50,model_dir='temp',
                metric='AUC'):
        """Trains the model on the given training dataset with periodic 
        evaluation on the validation dataset.

        Parameters
        ----------
        train_loader: Dataloader object for the training dataset.
        val_loader: Dataloader object for the validation dataset.
        learning_rate: Initial learning rate for training.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        ascent_step_size: Step size for gradient ascent for adversarial 
                          generation of negative points.
        ascent_num_steps: Number of gradient ascent steps for adversarial 
                          generation of negative points.
        metric: Metric used for evaluation (AUC / F1).
        """
        best_score = -np.inf
        best_model = None
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        c = self.init_center_c(train_loader, self.model)
        # adv_sample_box=[]
        # data_sample_box=[]
        data=[]
        temp=torch.tensor(data)
        for epoch in range(total_epochs):
            #Make the weights trainable
            self.model.train()
            lr_scheduler(epoch, total_epochs, only_ce_epochs, learning_rate, self.optimizer)
            
            #Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  #AdvLoss
            epoch_ce_loss = 0  #Cross entropy Loss

            # 每轮训练结束后，模型迭代，分类器能力增强，生成伪火焰样本的能力也需要增强
            s_time=time.time()
            batch_idx = -1
            for data, target in train_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                nofiredata = data[target == 1]
                if epoch==0:
                    temp=torch.cat((temp,data),0)
                # data = data[:,1:3,:,:].to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)
                # target=torch.reshape(target,[1])

                self.optimizer.zero_grad()
                
                # Extract the logits for cross entropy loss

                logits = self.model(data)
                logits = torch.squeeze(logits, dim = 1)
                # dist =(logits-c) ** 2
                # if nofiredata.shape[0]==0:
                #     dist =(logits-c) ** 2
                #     ce_loss = torch.mean(dist)
                # else:
                #     ce_loss = F.binary_cross_entropy_with_logits(logits, target)

                # binary_cross_entropy_with_logits的输入是网络输出的logits（未经sigmoid函数激活的），
                # 并且该函数会自动进行sigmoid函数激活处理，该损失函数已经内部自带了计算logit的操作，
                # 无需在传入给这个loss函数之前手动使用sigmoid/softmax将之前网络的输入映射到[0,1]之间
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                # 使用费雪正则化
                # lambda_fim = 0.01
                # fisher_diag = self.compute_fisher_diagonal(self.model, logits)
                # fisher_loss = sum(lambda_fim * fisher_diag[name] * torch.sum(param ** 2) for name, param in
                #                   self.model.named_parameters())
                # ce_loss=fisher_loss
                epoch_ce_loss += ce_loss

                '''
                Adversarial Loss is calculated only for the positive data points (label==0).
                '''
                if  epoch >= only_ce_epochs:
                    data = data[target == 0]
                    # adv_lossbox = self.one_class_adv_loss(logits, data, epoch, batch_idx)
                    # data=data[0:data.shape[0]-nofiredata.shape[0],:,:,:]
                    # img = data[0]
                    # img = img.cpu().numpy()
                    # img = np.transpose(img, (1, 2, 0))  # C*H*W -> H*W*C
                    # plt.imshow(img)
                    # plt.show()
                    # AdvLoss
                    if nofiredata.shape[0] == 0:
                        adv_lossbox = self.one_class_adv_loss(logits, data, epoch, batch_idx)
                    else:
                        gradients = get_gradients(self.model, self.device, data, target)
                        adv_lossbox = DROCCLFTrainer.one_class_adv_loss(data, gradients)
                    adv_loss=adv_lossbox[0]
                    # adv_sample_box.append(adv_lossbox[1])
                    # print("adv_loss",adv_loss)
                    epoch_adv_loss += adv_loss
                    loss = ce_loss + adv_loss * self.lamda
                else: 
                    # If only CE based training has to be done
                    loss = ce_loss
                # Backprop
                # with backpack(DiagGGNExact()):  #传入的是实例
                #     loss.backward()
                loss.backward()#求导，计算损失函数梯度，以便确定参数更新的方向和大小，实现反向传播
                # 根据计算得到的梯度，更新网络器参数，更新后的参数将被用于下一次的前向传递计算和反向传播计算
                self.optimizer.step()
            e_time=time.time()
            trtime=e_time-s_time
            print(trtime)
            epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
            epoch_adv_loss = epoch_adv_loss/(batch_idx + 1) #Average AdvLoss

            # ****************************
            test_score= self.test(val_loader, metric)#进入test函数，里面有metric
            if test_score[0][0] > best_score:
                best_score = test_score[0][0]
                # best_model只是暂时保存了目前为止的最佳模型，并未参与训练（因为是深拷贝，不管self.model怎么变，bestmodel不变）
                # 参与训练的只有self.model
                best_model = copy.deepcopy(self.model)
                # params = list(best_model.named_parameters())
                # print('best model:',params)

            print('Epoch: {}, CE Loss: {}, AdvLoss: {}, {}: {}'.format(
                epoch, epoch_ce_loss.item(), epoch_adv_loss.item(),
                metric, test_score[0]))

            self.save2(model_dir,epoch)
            saveToExcel(epoch_ce_loss.item(),epoch_adv_loss.item(),epoch,trtime)
        self.model = copy.deepcopy(best_model)



    def test(self,test_loader, metric):
        """Evaluate the model on the given test dataset.

        Parameters
        ----------
        test_loader: Dataloader object for the test dataset.
        metric: Metric used for evaluation (AUC / F1).
        """        
        self.model.eval()
        label_score = []
        batch_idx = -1
        # for data, target in test_loader:
        # i=0
        for data,target in test_loader:
            # if i<20:
            #     target=torch.from_numpy(np.zeros((len(data))))
            # else:
            #     target=torch.from_numpy(np.ones((len(data))))
            # i+=1
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)
            # data = data[:, 1:3, :, :].to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            # sigmoid_logits = torch.sigmoid(logits)
            scores = logits
            # logits = torch.squeeze(logits, dim=1)
            # logits=logits.detach().numpy().reshape(len(logits),1)
            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            scores.tolist()))
        # Compute test score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        if metric == 'F1':
            # Evaluation based on https://openreview.net/forum?id=BJJLHbb0-
            thresh = np.percentile(scores, 50)
            y_pred = np.where(scores >= thresh, 1, 0)#scores >= thresh取1，scores <= thresh取0,原
            prec, recall, test_metric, _ = precision_recall_fscore_support(
                labels, y_pred)
            # test_metric=2*prec[0]*recall[0]/(prec[0]+recall[0])
            acc = accuracy_score(labels, y_pred)

            print("F1", test_metric)
            print("acc", acc)
            # 所有无火样本中，被判为有火的概率
            print("误警率:", np.sum(y_pred[np.where(labels == 1)] == 0) / np.sum(labels == 1))
            # 所有有火样本中，被判为有火的概率,也就是召回率，TP/(TP+FN)=TP/P
            print("火焰检测率:", recall[0])
        if metric == 'AUC':
            # Evaluation based on https://openreview.net/forum?id=BJJLHbb0-
            thresh = np.percentile(scores, 50)
            y_pred = np.where(scores >= thresh, 1, 0)#scores >= thresh取1，scores <= thresh取0,原
            # y_pred = np.where(scores <= thresh, 1, 0)
            # y_pred=scores
            # y=np.sum(y_pred)
            prec, recall, test_metric, _ = precision_recall_fscore_support(
                labels, y_pred)
            #所有无火样本中，被判为有火的概率
            print("误警率:",np.sum(y_pred[np.where(labels==1)]==0)/np.sum(labels==1))
            #所有有火样本中，被判为有火的概率,也就是召回率，TP/(TP+FN)=TP/P
            print("火焰检测率:",recall[0])
            # test_metric = roc_auc_score(labels, scores)  # labels真实标签，scores预测标签
            # saveToExcel(np.sum(y_pred[np.where(labels == 1)] == 0) / np.sum(labels == 1), recall[0], epoch,test_metric)
        return test_metric,thresh,y_pred,scores
        
    
    def one_class_adv_loss(self,logits, x_train_data,epoch,batch_idx):
        """Computes the adversarial loss:
        0为火焰类，视为正常类，1为无火类，视为异常类
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r) 
            classified as +ve (label=0). This is done by maximizing 
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R 
            (set N_i(r))
        4) Pass the calculated adversarial points through the model, 
            and calculate the CE loss wrt target class 0
        
        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points

        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        # img = x_adv_sampled[0]
        # img = img.cpu().detach().numpy()
        # img = np.transpose(img, (1, 2, 0))  # C*H*W -> H*W*C
        # plt.imshow(img)
        # plt.show()

        for step in range(self.ascent_num_steps):
            # 按公式不断更新x_adv_sampled，每10步按公式投影一次x_adv_sampled到Ni(r)集
            with torch.enable_grad():#允许计算梯度
                # new_targets = torch.zeros(batch_size, 1).to(self.device)
                new_targets = torch.ones(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)
                # new_targets=torch.reshape(new_targets,[1])
                
                logits = self.model(x_adv_sampled)         
                logits = torch.squeeze(logits, dim = 1)
                # 公式Adversarial search:第一步，计算网络相对于负标签（异常点）的损失
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]#原代码，公式Adversarial search:第二步，梯度上升找对抗点，先求梯度
                grad_norm = torch.norm(grad, p=2, dim = tuple(range(1, grad.dim())))#公式Adversarial search:第二步求梯度的二范数
                grad_norm = grad_norm.view(-1, *[1]*(grad.dim()-1))
                grad_normalized = grad/grad_norm #公式Adversarial search:第二步，梯度除以梯度的二范数
            with torch.no_grad():#公式Adversarial search:第二步，结果相加，就是沿梯度上升，ascent_step_size就是η，
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10==0:#每10步投影一次异常点到Ni(r)集，公式Adversarial search中第三步：投影
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h**2,
                                                dim=tuple(range(1, h.dim()))))#点之间的距离
                # clamp(输入张量,min,max)函数的功能将输入input张量每个元素的值压缩到区间[min, max]，并返回结果到一个新张量
                alpha = torch.clamp(norm_h, self.radius,
                                    self.gamma * self.radius).to(self.device)#论文，算法步骤Adversarial search中第三步，where α=
                # Make use of broadcast to project h
                proj = (alpha/norm_h).view(-1, *[1] * (h.dim()-1))#公式Adversarial search中第三步，proj压缩比
                h = proj * h#论文，算法步骤Adversarial search中第三步
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere


        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        # adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))#原始
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 1))


        # if epoch==0 or epoch==2 or epoch==4:
        #     path = r'F:\videofireimg\adv_epoch{}.xls'.format(epoch)
        #     # workbook = xlwt.Workbook(encoding='utf-8')
        #     workbook = xlrd.open_workbook(path)
        #     sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
        #     worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
        #     rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
        #     new_workbook = xlcopy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
        #     new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
        #     k=0
        #     for o in range(x_adv_sampled.shape[0]):
        #         img = x_adv_sampled[o]
        #         new_worksheet.write(k+rows_old, 0,float(img[0][0]))
        #         new_worksheet.write(k+rows_old, 1,float(img[0][1]))
        #         k=k+1
                # img2 = x_train_data[o]
                # for row in range(4):
                #     for col in range(4):
                #         new_worksheet.write(k, 0,float(img[0][row,col:col+1]))
                #         new_worksheet.write(k, 1,float(img[1][row,col:col+1]))
                #         new_worksheet.write(k, 2,float(img[2][row,col:col+1]))
                #         new_worksheet.write(k, 3, float(img2[0][row, col:col + 1]))
                #         new_worksheet.write(k, 4, float(img2[1][row, col:col + 1]))
                #         new_worksheet.write(k, 5, float(img2[2][row, col:col + 1]))
                #         k+=1
            # new_workbook.save(path)
                # img = img.cpu().detach().numpy()
        #         # img = np.transpose(img, (1, 2, 0))  # C*H*W -> H*W*C
        #         # min_val = np.min(img)
        #         # max_val = np.max(img)
        #         # img_data_clamped = (img - min_val) / (max_val - min_val)
        #         x_train_da=x_train_data[o]
        #         # x_train_da=np.transpose(x_train_da,(1,2,0))
        #         # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        #         # img = transform(cropped_image)
        #         img = img.reshape((1, 3, 4, 4))
        #         img = img.to(self.device)
        #         img = img.to(torch.float)
        #         logits_adv=self.model(img)
        #
        #         x_train_da = x_train_da.reshape((1, 3, 4, 4))
        #         x_train_da = x_train_da.to(self.device)
        #         x_train_da = x_train_da.to(torch.float)
        #         logits_fire=self.model(x_train_da)
        #         for row in range(4):
        #             for col in range(4):
        #                 worksheet.write(k, 0,float(img[0][row,col:col+1]))
        #                 worksheet.write(k, 1,float(img[1][row,col:col+1]))
        #                 worksheet.write(k, 2,float(img[2][row,col:col+1]))
        #                 worksheet.write(k, 0, float(logits_adv))
        #
        #                 # worksheet.write(k, 7, float(x_train_da[0][row, col:col + 1]))
        #                 # worksheet.write(k, 8, float(x_train_da[1][row, col:col + 1]))
        #                 # worksheet.write(k, 9, float(x_train_da[2][row, col:col + 1]))
        #                 worksheet.write(k, 1, float(logits_fire))
        #                 k+=1
        #     # 保存表
        #     workbook.save(path)

                # plt.imsave("/mnt/videofireimg/cropvideo2-onecls/four_four/fake/fire{}.png".format(o), img_data_clamped)
                # plt.imsave("/mnt/videofireimg/cropvideo2-onecls/four_four/true/fire{}.png".format(o), x_train_da)
            # print("epochPPPPPPPPPPPPPPPPPPPPPPPP",epoch)
        return adv_loss,x_adv_sampled

    def save(self, path,metric):
        torch.save(self.model.state_dict(),os.path.join(path, '4px-AUC-SGD-lr0.001-r0.2-gamma1.0.pt'.format(metric)))

    def save2(self, path,i):
        torch.save(self.model.state_dict(),os.path.join(path, '4px-STMDA_{}.pt'.format(i)))

    def load(self, path,str):
        self.model.load_state_dict(torch.load(os.path.join(path,str)))


    def val(self,path_src,s,vthresh):
        scorebox = np.array([])

        for i in range(1,len(os.listdir(path_src))+1):
            img = Image.open(path_src + '{}_'.format(s)+str(i)+'.png')
            print('{}_'.format(s)+str(i)+'.png')

            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            img = transform(img)
            img = torch.reshape(img, (1, 3, 4, 4))
            img = img.to(self.device)
            img = img.to(torch.float)

            logits = self.model(img)
            logits = torch.squeeze(logits, dim=1)

            scores = logits

            # scores.cpu().data.numpy()-----GPU数据转到CPU，再将tensor转为numpy
            scores = np.array(scores.cpu().data.numpy())
            scorebox = np.hstack((scorebox, scores))
        print(scorebox)
        thresh =vthresh
        print('val thresh', thresh)
        y_pred = np.where(scorebox >= thresh, 1, 0)  # scores >= thresh取1，scores <= thresh取0
        print(y_pred)
        fire_index=np.where(y_pred==0)+np.ones((1,np.size(np.where(y_pred==0))))
        return fire_index

    def real_time_val(self,f,vthresh,cutsize,frame):


        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 宽度854再加42，左右各加21，高度288再加48，上下各加24,分块56x56
        # borderType = cv2.BORDER_REFLECT指边界反射填充(以边界为轴对称填充)
        # top_btm = 0
        # left_rgt = 1
        # replicate = cv2.copyMakeBorder(img, top_btm, top_btm, left_rgt, left_rgt, borderType=cv2.BORDER_REFLECT)
        # replicate = cv2.cvtColor(replicate, cv2.COLOR_BGR2RGB)
        # imgori = replicate
        imgori=img

        high, width, channel = imgori.shape
        row=high
        col=width
        thresh = vthresh
        print('val thresh', thresh)
        w=h = cutsize
        y_pred=[]
        new_img = Image.new('RGB', (width, high))
        runtime=0
        for i in range(row // cutsize):
            for j in range(col // cutsize):
                cropped_image = imgori[i * cutsize:cutsize * (i + 1),
                                cutsize * j:cutsize * (j + 1)]  # Slicing to crop the image

                transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                img = transform(cropped_image)
                img = img.reshape((1, 3, cutsize, cutsize))
                img = img.to(self.device)
                img = img.to(torch.float)
                s_time = time.time()
                logits = self.model(img)
                logits = torch.squeeze(logits, dim=1)

                scores = logits
                if scores>=thresh:#原
                # if scores <= thresh:
                    imgn = Image.new("RGB", (w, h), "black")  # 新建图像
                    y_pred.append(1)
                    e_time = time.time()
                else:
                    # img_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                    img_pil = Image.fromarray(cropped_image)
                    imgn = img_pil
                    y_pred.append(0)
                    e_time = time.time()
                runtime = runtime+e_time - s_time
                new_img.paste(imgn, (j * h, i * w))
        # new_img.save(r'F:\videofireimg\ST-MDA\frame{}.png'.format(f))
        return runtime,y_pred


    def batch_val(self,label,cutsize,frame):
        s_time = time.time()
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 宽度854再加42，左右各加21，高度288再加48，上下各加24,分块56x56
        # borderType = cv2.BORDER_REFLECT指边界反射填充(以边界为轴对称填充)
        # top_btm = 0
        # left_rgt = 1
        # replicate = cv2.copyMakeBorder(img, top_btm, top_btm, left_rgt, left_rgt, borderType=cv2.BORDER_REFLECT)
        # replicate = cv2.cvtColor(replicate, cv2.COLOR_BGR2RGB)
        # imgori = replicate
        imgori = frame

        high, width, channel = imgori.shape
        row = high
        col = width
        k1=1
        k2=1
        for i in range(row // cutsize):
            for j in range(col // cutsize):
                cropped_image = imgori[i * cutsize:cutsize * (i + 1),
                                cutsize * j:cutsize * (j + 1)]  # Slicing to crop the image

                # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                # img = transform(cropped_image)
                # img = img.reshape((1, 3, 4, 4))
                # img = img.to(self.device)
                # img = img.to(torch.float)

                if label[i*320+j] == 0:
                    cv2.imwrite(r'/mnt/videofireimg/flamevideo-onecls/four_four/tempval/fire/4px-frame{}.png'.format(k1),cropped_image)
                    k1=k1+1
                else:
                    cv2.imwrite(r'/mnt/videofireimg/flamevideo-onecls/four_four/tempval/nofire/4px-frame{}.png'.format(k2),cropped_image)
                    k2 = k2 + 1

        e_time = time.time()
        runtime = e_time - s_time

        return runtime


    def test2(self,test_loader, metric,vthresh):
        s_time = time.time()
        self.model.eval()
        label_score = []
        batch_idx = -1

        for data,target in test_loader:
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            sigmoid_logits = torch.sigmoid(logits)
            scores = logits
            logits = torch.squeeze(logits, dim=1)

            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            scores.tolist()))
        # Compute test score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        if metric == 'AUC':
            thresh = vthresh
            y_pred = np.where(scores >= thresh, 1, 0)#scores >= thresh取1，scores <= thresh取0
            prec, recall, test_metric, _ = precision_recall_fscore_support(
                labels, y_pred)
            #所有无火样本中，被判为有火的概率
            fp=np.sum(y_pred[np.where(labels == 1)] == 0) / np.sum(labels == 1)
            print("误警率:",fp)
            #所有有火样本中，被判为有火的概率,也就是召回率，TP/(TP+FN)=TP/P
            print("火焰检测率:",recall[0])
            test_metric = roc_auc_score(labels, scores)  # labels真实标签，scores预测标签
        e_time = time.time()
        runtime = e_time - s_time

        return test_metric,fp,recall[0],runtime

    def one_class_adv_lossL1(self, logits, x_train_data, epoch, batch_idx,r,gamma):
        batch_size = len(x_train_data)
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            # 按公式不断更新x_adv_sampled，每10步按公式投影一次x_adv_sampled到Ni(r)集
            with torch.enable_grad():  # 允许计算梯度
                new_targets = torch.ones(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                logits = self.model(x_adv_sampled)
                logits = torch.squeeze(logits, dim=1)
                # 公式Adversarial search:第一步，计算网络相对于负标签（异常点）的损失
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]  # 公式Adversarial search:第二步，梯度上升找对抗点，先求梯度
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))  # 公式Adversarial search:第二步求梯度的二范数
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_normalized = grad / grad_norm  # 公式Adversarial search:第二步，梯度除以梯度的二范数

                # ****************************
            with torch.no_grad():  # 公式Adversarial search:第二步，结果相加，就是沿梯度上升，ascent_step_size就是η，
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)


            if (step + 1) %10 == 0:  # 每10步投影一次异常点到Ni(r)集，公式Adversarial search中第三步：投影
                # Project the normal points to the set N_i(r)
                x_adv_sampled=torch.from_numpy(self.project_to_l1_ball(x_adv_sampled,r,gamma)).type(torch.float32).detach().requires_grad_()
                # 方式1：设置三维图形模式
                # plt.rcParams['font.sans-serif'] = ['SimHei']
                # plt.rcParams['axes.unicode_minus'] = False
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(x_train_data[0:10, 0, :, :].reshape(-1).detach().numpy(),
                #            x_train_data[0:10, 1, :, :].reshape(-1).detach().numpy(),
                #            x_train_data[0:10, 2, :, :].reshape(-1).detach().numpy(),s=3,c='b', marker='o')  # 画出(xs1,ys1,zs1)的散点图。
                # ax.set_xlabel('X')  # 画出坐标轴
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # plt.show()



                # h = x_adv_sampled - x_train_data
                # norm_h = torch.sum(abs(h),dim=tuple(range(1, h.dim())))
                # alpha = torch.clamp(norm_h, self.radius,
                #                     self.gamma * self.radius).to(self.device)  # 论文，算法步骤Adversarial search中第三步，where α=
                # proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))  # 公式Adversarial search中第三步
                # h = proj * h  # 论文，算法步骤Adversarial search中第三步
                # x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere



                # ax.scatter(x_adv_sampled[0:10, 0, :, :].reshape(-1).detach().numpy(),
                #            x_adv_sampled[0:10, 1, :, :].reshape(-1).detach().numpy(),
                #            x_adv_sampled[0:10, 2, :, :].reshape(-1).detach().numpy(), s=10,c='y', marker='^')
                # ax.scatter(x_adv_sampled2[0:10, 0, :, :].reshape(-1).detach().numpy(),
                #            x_adv_sampled2[0:10, 1, :, :].reshape(-1).detach().numpy(),
                #            x_adv_sampled2[0:10, 2, :, :].reshape(-1).detach().numpy(), s=15,c='r', marker='*')
                # plt.legend(['train data', 'L1 norm','L2 norm'])
                # plt.show()
        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 1))
        return adv_loss, x_adv_sampled


    def one_class_adv_loss3(self, logits, data, epoch, batch_idx):
        nofire_pred = self.model(data)
        nofire_pred = torch.squeeze(nofire_pred, dim=1)

        new_targets = torch.ones(data.shape[0], 1).to(self.device)
        new_targets = torch.squeeze(new_targets)
        new_targets = new_targets.to(torch.float)
        adv_loss = F.binary_cross_entropy_with_logits(nofire_pred, (new_targets * 1))
        return adv_loss,nofire_pred

    def project_to_ellipsoid(self,x,center):
        # x: point to be projected
        # center: center of the ellipsoid
        # A: positive definite matrix defining the ellipsoid
        # center = np.array([0.0, 0.0])
        A = np.array([[1, 0], [0, 0.5]])# Defines an ellipsoid with different axis lengths
        A = np.expand_dims(A, 0)
        A = A.repeat(x.shape[0], axis=0)

        diff = (x - center).detach().numpy()
        a=np.linalg.inv(A)
        dist = np.sqrt(diff.T @ np.linalg.inv(A) @ diff)
        if dist <= 1:
            return 0  # Already inside the ellipsoid
        # projected_point = center + diff / dist  # Scale the difference vector
        return dist


    def one_class_adv_loss_ellipsoid(self, logits, x_train_data, epoch, batch_idx):
        batch_size = len(x_train_data)
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():
                new_targets = torch.ones(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                logits = self.model(x_adv_sampled)
                logits = torch.squeeze(logits, dim=1)

                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]  #公式Adversarial search:第二步，梯度上升找对抗点，先求梯度
                grad_norm = torch.norm(grad, p=1, dim=tuple(range(1, grad.dim())))  #公式Adversarial search:第二步求梯度的二范数
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_normalized = grad / grad_norm  #公式Adversarial search:第二步，梯度除以梯度的二范数
            with torch.no_grad():  #公式Adversarial search:第二步，结果相加，就是沿梯度上升，ascent_step_size就是η，
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10 == 0:  # 每10步投影一次异常点到Ni(r)集，公式Adversarial search中第三步：投影
                h=self.project_to_ellipsoid(x_adv_sampled,x_train_data)
                x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 1))
        return adv_loss, x_adv_sampled




    def train2(self, size, train_loader, test_loader, learning_rate, lr_scheduler, total_epochs,only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50, model_dir='temp',metric='AUC'):
        best_score = -np.inf
        best_model = None
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size

        data = []
        temp = torch.tensor(data)
        for epoch in range(total_epochs):
            # Make the weights trainable
            self.model.train()
            lr_scheduler(epoch, total_epochs, only_ce_epochs, learning_rate, self.optimizer)

            # Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  # AdvLoss
            epoch_ce_loss = 0  # Cross entropy Loss

            # 每轮训练结束后，模型迭代，分类器能力增强，生成伪火焰样本的能力也需要增强
            batch_idx = -1
            for data, target in train_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                if epoch == 0:
                    temp = torch.cat((temp, data), 0)
                target = target.to(torch.float)
                target = torch.squeeze(target)

                self.optimizer.zero_grad()

                logits = self.model(data)
                logits = torch.squeeze(logits, dim=1)
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                epoch_ce_loss += ce_loss

                if epoch >= only_ce_epochs:
                    data = data[target == 0]
                    adv_lossbox = self.one_class_adv_loss3(logits, data, epoch, batch_idx)
                    adv_loss = adv_lossbox[0]
                    epoch_adv_loss += adv_loss
                    loss = ce_loss + adv_loss * self.lamda
                else:
                    # If only CE based training has to be done
                    loss = ce_loss
                # Backprop
                loss.backward()  # 求导，计算损失函数梯度，以便确定参数更新的方向和大小，实现反向传播
                # 根据计算得到的梯度，更新网络器参数，更新后的参数将被用于下一次的前向传递计算和反向传播计算
                self.optimizer.step()

            epoch_ce_loss = epoch_ce_loss / (batch_idx + 1)  # Average CE Loss
            epoch_adv_loss = epoch_adv_loss / (batch_idx + 1)  # Average AdvLoss


            print('Epoch: {}, CE Loss: {}, AdvLoss: {}'.format(
                epoch, epoch_ce_loss.item(), epoch_adv_loss.item()))

        print("test**************")
        logits = self.model(temp)
        logits = torch.squeeze(logits, dim=1)
        clf = svm.OneClassSVM(nu=0.0001, kernel="linear", gamma='auto')
        clf.fit(logits.detach().numpy().reshape(-1, 1))

        data = []
        temptest = torch.tensor(data)
        temptarget = torch.tensor(data)
        for data, t in test_loader:
            data, t = data.to(self.device), t.to(self.device)
            data = data.to(torch.float)
            temptest = torch.cat((temptest, data), 0)
            temptarget = torch.cat((temptarget, t), 0)  # 0火，1非火,  1改为-1,0改为1
        logitstest = self.model(temptest)
        r = torch.where(temptarget == 1, -1, temptarget)
        target = torch.where(r == 0, 1, r)
        y_pred_test = clf.predict(logitstest.detach().numpy())
        accuracy = accuracy_score(target, y_pred_test)
        print("准确率:", accuracy)
        joblib.dump(clf, './log/svm_model{}.pkl'.format(epoch))
        self.save2(model_dir, epoch)
        # print("误警率:", )
        # print("火焰检测率:", )

    def real_time_val2(self, f, test_loader, cutsize, frame):
        data = []
        temptest = torch.tensor(data)
        temptarget = torch.tensor(data)
        for data, t in test_loader:
            data, t = data.to(self.device), t.to(self.device)
            data = data.to(torch.float)
            temptest = torch.cat((temptest, data), 0)
            temptarget = torch.cat((temptarget, t), 0)  # 0火，1非火,  1改为-1,0改为1
        logitstest = self.model(temptest)
        clf = KMeans(n_clusters=2)
        label = clf.fit(logitstest.detach().numpy())
        label = label.labels_
        accuracy = accuracy_score(temptarget, label)
        print("准确率:", accuracy)


    def project_to_l1_ball(self,x, r,gamma):
        """
        Project a point x onto the L1 ball of radius r.
        Parameters:
        x (np.ndarray): The input point.
        r (float): The radius of the L1 ball.
        Returns:
        np.ndarray: The projected point.
        """
        x=x.detach().numpy()
        sample=np.empty([x.shape[0],3,x.shape[2],x.shape[3]])
        # if np.sum(np.abs(x)) <= r:
        #     return x
        for i in range(x.shape[0]):
            dim0=x[i][0]
            dim1 = x[i][1]
            dim2 = x[i][2]
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    p=[]
                    p=np.append(p,[dim0[j][k],dim1[j][k],dim2[j][k]])
                    if gamma*r > np.sum(np.abs(p)) > r:
                        sample[i, 0, j, k] = dim0[j][k]
                        sample[i, 1, j, k] = dim1[j][k]
                        sample[i, 2, j, k] = dim2[j][k]
                        continue
                    if np.sum(np.abs(p)) > gamma*r:
                        r = gamma * r
                    # Sort the absolute values in descending order
                    u = np.sort(np.abs(p))[::-1]  # 对数组进行降序
                    # Find the critical k
                    sv = np.cumsum(u)
                    rho = np.where(u > (sv - r) / np.arange(1, len(u) + 1))[0][-1]
                    # Calculate the threshold value
                    theta = (sv[rho] - r) / (rho + 1)
                    # Compute the projection
                    y = np.sign(p) * np.maximum(np.abs(p) - theta, 0)
                    sample[i,0,j,k]=y[0]
                    sample[i, 1, j, k] = y[1]
                    sample[i, 2, j, k] = y[2]
        return sample

    def testsinx(self,test_loader, metric,thresh):

        self.model.eval()
        label_score = []
        batch_idx = -1

        for data,target in test_loader:
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)

            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            sigmoid_logits = torch.sigmoid(logits)
            scores = logits
            logits = torch.squeeze(logits, dim=1)
            # logits=logits.detach().numpy().reshape(len(logits),1)
            # scores = svm.predict(logits)
            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            scores.tolist()))
        # Compute test score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        if metric == 'AUC':
            y_pred = np.where(scores >= thresh, 1, 0)#scores >= thresh取1，scores <= thresh取0
            # y_pred=scores
            # y=np.sum(y_pred)
            # prec, recall, test_metric, _ = precision_recall_fscore_support(
            #     labels, y_pred)
            # #所有无火样本中，被判为有火的概率
            # print("误警率:",np.sum(y_pred[np.where(labels==1)]==0)/np.sum(labels==1))
            # #所有有火样本中，被判为有火的概率,也就是召回率，TP/(TP+FN)=TP/P
            # print("火焰检测率:",recall[0])

            # test_metric = roc_auc_score(labels, scores)  # labels真实标签，scores预测标签
            test_metric=0
            # saveToExcel(np.sum(y_pred[np.where(labels == 1)] == 0) / np.sum(labels == 1), recall[0], advloss, epoch,test_metric)
        return test_metric,thresh,y_pred

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(1, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def compute_fisher_diagonal(self,model, loss):
        fisher_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

        # 计算 batch 内每个样本的梯度
        for i in range(loss.shape[0]):  # 遍历 batch 维度
            grad = torch.autograd.grad(loss[i], model.parameters(), retain_graph=True)

            for (name, param), g in zip(model.named_parameters(), grad):
                fisher_diag[name] += g.pow(2)  # 梯度平方

        # 计算均值（batch 归一化）
        for name in fisher_diag:
            fisher_diag[name] /= loss.shape[0]
        return fisher_diag


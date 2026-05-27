import math
import os
import shutil

import cv2
import imageio
import numpy as np
import torch
import torchvision
from PIL import Image



# jpg转png
from matplotlib import pyplot as plt


def jpg2png(origin_folder):
    # 设置输入文件夹路径和目标格式
    target_format = 'png'
    # 循环遍历输入文件夹中的所有文件
    for filename in os.listdir(origin_folder):
        # 检查文件是否为jpg格式
        if filename.endswith('.jpg'):
            # 组合新的文件名和路径
            img_path_jpg = os.path.join(origin_folder, filename)
            img_path_png = os.path.splitext(img_path_jpg)[0] + '.' + target_format

            # 读取JPG格式图片并保存为PNG格式
            with Image.open(img_path_jpg) as img:
                img.save(img_path_png)
            # shutil.move( img_path_jpg,  input_folder)
        # 删除原始JPG格式图片
        # os.remove(img_path_jpg)

# origin_folder=path_src = r'F:\train2\jpg'
# jpg2png(origin_folder)






# png图片转为视频
def png2video(filename, origin_folder, targetPath):
    # 读取所有 PNG 图片
    images = []
    for file_name in os.listdir(origin_folder):
        if file_name.endswith('.png'):
            images.append(Image.open(origin_folder + '\\' + file_name))
    # 将图片转换为视频
    fps = 25  # 每秒钟30帧
    filePath = os.path.join(targetPath, filename)
    with imageio.get_writer(filePath, fps=fps) as video:
        for image in images:
            frame = image.convert('RGB')
            video.append_data(np.array(frame))

# 设置生成的视频文件名和路径
# filename = 'output.mp4'
# origin_folder = r'F:\DJfire\png'
# targetPath = r'F:\DJfire\video'
# png2video(filename,origin_folder,targetPath)







def img_merge(path,fire_index,s):
    # 打开图片并获取大小
    # 需要合成width=896，height=336大小图片
    w, h = Image.open(path + '/' + '{}_1.png'.format(s)).size
    picrow,piccol=Image.open(r'/mnt/videofireimg/cropvideo2-onecls/four_four/val/fourPixel_{}.png'.format(s)).size

    # 合并图像
    new_img = Image.new('RGB', (picrow, piccol))
    num = 1
    for row in range(piccol // h):
        for col in range(picrow // w):
            if num in fire_index:
                img = Image.open(path + '/' + '{}_{}.png'.format(s,num))
            else:
                img= Image.new("RGB", (w, h), "black") #新建图像
            # 创建新的图像,二元组(x, y) 则为 im 左上角在此图片中的坐标
            new_img.paste(img, (col * h, row * w))
            num = num + 1

    # 保存新的图像
    new_img.save(r'/mnt/videofireimg/cropvideo2-onecls/four_four/val/4px_result_{}.png'.format(s))

# fire_index=np.array([2,17,33,49,50,54,55,56,57,74])
# path = r'F:\videofireimg\cropvideo2-onecls\val\first'
# img_merge(path,fire_index)







def cut_image(path, cutsize, storagePath,flag,device,model):
    # 文件名字：first_1,half_2
    img = cv2.imread(path)
    row,col,channel=img.shape
    num=1
    label=np.zeros([row,col])
    pred=np.zeros([row//cutsize,col//cutsize])
    for i in range(row//cutsize):
        for j in range(col//cutsize):
            if flag=="getLabel":
                label[i * cutsize:cutsize * (i + 1), cutsize * j:cutsize * (j + 1)]=np.ones([cutsize,cutsize])*num
            elif flag=="getPred":
                cropped_image =img[i * cutsize:cutsize * (i + 1), cutsize * j:cutsize * (j + 1)]
                transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                img = transform(cropped_image)
                img = img.reshape((1, 3, cutsize, cutsize))
                img = img.to(device)
                img = img.to(torch.float)
                logits = model(img)
                logits = torch.squeeze(logits, dim=1)
                pred[i,j]=logits
            else:
                cropped_image = img[i * cutsize:cutsize * (i + 1),cutsize * j:cutsize * (j + 1)]
                # cv2.imwrite(storagePath + '/' + 'video3_'+ str(num) + '.png', cropped_image)
            num=num+1
    if flag == "getLabel":
        return label
    if flag == "getPred":
        return pred









def border_img(path,cutsize):
    img = cv2.imread(path)
    row,col,_=img.shape
    if row%cutsize==0 and col%cutsize==0:
        return path
    else:
        top_btm=math.ceil(row//cutsize)*cutsize-row
        left_rgt=math.ceil(col//cutsize)*cutsize-col
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #宽度854再加42，左右各加21，高度288再加48，上下各加24,分块56x56
        # borderType = cv2.BORDER_REFLECT指边界反射填充(以边界为轴对称填充)
        replicate = cv2.copyMakeBorder(img, top_btm, top_btm, left_rgt, left_rgt, borderType=cv2.BORDER_REFLECT)
        replicate = cv2.cvtColor(replicate, cv2.COLOR_BGR2RGB)
        # print(replicate.shape)
        # plt.imshow(replicate)
        # plt.show()
        border_path=r'F:\videofireimg\ST-MDA\Video2Nofire4_border.jpg'
        cv2.imwrite(border_path, replicate)
        print("ok")
        return border_path

def drawPic(tpbox,fpbox,f):
    # 解决中文乱码
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    frames=f
    plt.figure()
    plt.ylim(0, 100)
    plt.xlabel('Frames')
    plt.ylabel('Fire detection rate')
    plt.plot(frames, tpbox, ls='-', c='r', label='TP')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))


    plt.figure()
    plt.ylim(0, 50)
    plt.xlabel('Frames')
    plt.ylabel('Error Warning rate')
    plt.plot(frames, fpbox, ls='-', c='r', label='FP')
    # 显示图例
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))  # bbox_to_anchor=(1,0.8)图例位置，x越大越往右，y越大越往上
    # 显示柱状图
    plt.show()
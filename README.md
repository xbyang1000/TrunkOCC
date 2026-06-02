# TrunkOCC:  Tree trunk detection by using deep one-class classification (OCC)

Our trunk data was collected from Huanghai National Forest Park, located at Dongtai City, Jiangsu Province, China.

## Main Features

- PyTorch platform
- Supporting `4x4` or `8x8`pixel-input for small object detection
- Data labels stored in mat-format file


## Model Training

by Running
```bash
python main.py --batch_size xx --epochs xx --lr xx, # e.g., "python main.py --batch_size 80 --epochs 100 --lr 0.03"
```



## Real-Time Validation

项目支持对图像帧进行实时块级验证。流程包括：

1. 读取输入图像或视频帧
2. 按设定块大小切分
3. 使用训练好的模型逐块判断
4. 统计预测结果并计算评估指标

使用时请根据本地路径修改：

- 训练数据路径
- 测试数据路径
- `.mat` 标签文件路径
- 实时输入图像路径
- 结果输出 `.xls` 路径



## 个人设备

4060 ，16G内存


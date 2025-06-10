# DAY2实习课程笔记

### (下列内容为本日实习课程笔记)

---

# 1 GitHub 推送教程（含 PyCharm 操作）

## 1.1 第一步：新建 GitHub 仓库
1. 登录 GitHub，点击右上角 + 创建新仓库。
2. 填写仓库名称，选择公开或私有，**不要勾选初始化 README**。
3. 点击 **Create repository**。

---

## 1.2 第二步：准备本地项目
1. 新建一个本地文件夹，将你的项目文件放入其中。
2. 在该文件夹内 **右键打开 Git Bash**。
3. 初始化 Git 仓库：
```bash
git init
```

---

## 1.3 第三步：使用 PyCharm 进行版本管理
1. 打开 PyCharm，导入项目。
2. 在右下角点击 **Version Control**，查看 Git 状态和日志。
3. 选择未被版本控制的文件，点击 **+** 添加到 Git。
4. 提交更改。

---

## 1.4 第四步：连接远程仓库并推送
1. 添加远程仓库地址：
```bash
git remote add origin https://github.com/你的用户名/你的仓库名.git
```
2. 设置代理（如有需要）。
3. 推送代码：
```bash
git push -u origin master
```

---

## 1.5 第五步：修改并提交新代码
1. 修改代码后，重复以下命令：
```bash
git add .
git commit -m "更新说明"
git push
```

---

# 2 深度学习基础

## 2.1 深度学习训练套路
### 2.1.1 完整训练流程包括：
1. 数据准备
2. 模型定义
3. 损失函数
4. 优化器
5. 训练循环（epoch）
6. 验证与测试

### 2.1.2 欠拟合 vs 过拟合
- **欠拟合**：训练集和验证集表现都不好。
- **过拟合**：训练集表现很好，验证集表现差。


---

## 2.2 卷积神经网络（CNN）
#### 卷积神经网络（Convolutional Neural Network, CNN）是一类专门用于处理具有类似网格结构的数据（如图像）的深度学习模型。它在图像识别、目标检测、语音识别等任务中表现出色。

### 2.2.1 卷积层（Convolutional Layer）
- **作用**：提取局部特征（如边缘、纹理、形状等）。
- **参数**：
  - `in_channels`：输入通道数（如 RGB 图像为 3）。
  - `out_channels`：输出通道数（即卷积核个数）。
  - `kernel_size`：卷积核大小（如 3x3）。
  - `stride`：步长，控制卷积核滑动的步幅。
  - `padding`：边缘填充，防止尺寸缩小。

> 卷积操作的本质是滑动窗口的加权求和。

---

### 2.2.2 激活函数（Activation Function）
- 常用的激活函数：
  - `ReLU`（Rectified Linear Unit）：$$f(x) = \max(0, x)$$
  - `Leaky ReLU`：允许负值通过，缓解“神经元死亡”问题。
  - `Sigmoid` 和 `Tanh`：在深层网络中容易导致梯度消失，使用较少。

> ReLU 是 CNN 中最常用的激活函数，计算简单，收敛快。

---

### 2.2.3 池化层（Pooling Layer）
- **作用**：降低特征图尺寸，减少参数，防止过拟合。
- **类型**：
  - 最大池化（Max Pooling）：取窗口内最大值。
  - 平均池化（Average Pooling）：取窗口内平均值。
- **参数**：
  - `kernel_size`：池化窗口大小。
  - `stride`：滑动步长。

> 池化操作不改变通道数，只改变宽高。

---

### 2.2.4 批归一化（Batch Normalization）
- **作用**：加速训练，提高稳定性，缓解梯度消失。
- **原理**：对每一层的输入进行标准化处理，使其均值为 0，方差为 1。

---

### 2.2.5 全连接层（Fully Connected Layer）
- **作用**：将提取到的特征映射到最终的分类空间。
- 通常在卷积层和池化层之后使用。

---

### 2.2.6 Dropout 层
- **作用**：在训练过程中随机“丢弃”一部分神经元，防止过拟合。
- **原理**：每次训练时随机屏蔽部分神经元的输出。

---

### 2.2.7 CNN 的结构示意图
- **一个典型的 CNN 网络结构如下：**
```
输入图像 → 卷积层 → ReLU → 池化层 → 卷积层 → ReLU → 池化层 → 全连接层 → 输出
```

---

### 2.2.8 参数计算公式

- **输出尺寸计算**（不考虑 dilation）：

  ### Output Size = floor((W - K + 2P) / S) + 1


  - \( W \)：输入尺寸
  - \( K \)：卷积核尺寸
  - \( P \)：填充大小
  - \( S \)：步长

---

### 2.2.9 卷积操作示例

```python
import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = input.reshape(1, 1, 5, 5)
kernel = kernel.reshape(1, 1, 3, 3)

output = F.conv2d(input, kernel, stride=1)
output2 = F.conv2d(input, kernel, stride=2)
output3 = F.conv2d(input, kernel, stride=1, padding=1)
```

---

### 2.2.10 CIFAR10 卷积网络示例

```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                      ataloader = DataLoader(dataset, batch_size=64)

class CHEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)

    def forward(self, x):
        return self.conv1(x)

chen = CHEN()
writer = SummaryWriter("conv_logs")

step = 0
for data in dataloader:
    imgs, _ = data
    output = chen(imgs)
    writer.add_images("input", imgs, step)
    output = output.reshape(-1, 3, 30, 30)
    writer.add_images("output", output, step)
    step += 1
```

---

## 2.2.11 TensorBoard 可视化
1. 安装：
```bash
pip install tensorboard
```
2. 启动：
```bash
tensorboard --logdir=conv_logs
```
3. 打开浏览器访问提示的地址，查看训练过程可视化。

---

## 2.2.12 池化层
**常见的池化层：**
- 平均池化（average pooling）：计算图像区域的平均值作为该区域池化后的值。
- 最大池化（max pooling）:选图像区域的最大值作为该区域池化后的值。
- （代码里面是最大池化，还有平均池化）
```python
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#
dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

# # 最大池化没法对long整形进行池化
# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]], dtype = torch.float)
# input =torch.reshape(input,(-1,1,5,5))
# print(input.shape)


class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_1 = MaxPool2d(kernel_size=3,
                                   ceil_mode=False)
    def forward(self,input):
        output = self.maxpool_1(input)
        return output

chen = Chen()

writer = SummaryWriter("maxpool_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = chen(imgs)
    writer.add_images("ouput",output,step)
    step += 1
writer.close()

#
# output = chen(input)
# print(output)
```

---



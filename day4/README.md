# DAY4实习课程笔记

### (下列内容为本日实习课程笔记)

---

# 1 训练自己的数据集

## 1.1 数据集预处理
- 使用 `deal_with_datasets.py` 将原始数据集划分为训练集和验证集：
```python
from sklearn.model_selection import train_test_split
import os, shutil

# 设置路径
dataset_dir = r'D:\dataset\image2'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

# 创建目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历类别文件夹
for class_name in os.listdir(dataset_dir):
    if class_name not in ['train', 'val']:
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
        train_imgs, val_imgs = train_test_split(images, train_size=0.7, random_state=42)

        # 创建子目录并移动图片
        for img in train_imgs:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_imgs:
            shutil.move(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

        shutil.rmtree(class_path)
```

---

## 1.2 生成训练/验证路径文件
- 使用 `prepare.py` 生成 `train.txt` 和 `val.txt`：
```python
def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                f.write(f"{img_path} {label}\n")
```

---

## 1.3 自定义数据集类
```python
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, transform):
        self.imgs_path, self.labels = [], []
        with open(txt_path, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.imgs_path.append(path)
                self.labels.append(int(label))
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.imgs_path)
```

---

## 1.4 加载数据集
```python
from torchvision import transforms

train_data = ImageTxtDataset('train.txt', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]))
```

---

# 2 使用 GPU 加速训练

## 2.1 检查 GPU 是否可用

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

## 2.2 将模型和数据迁移到 GPU

```python
model = model.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```

## 2.3 训练循环示例

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

# 2.4 常见问题排查

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| 模型输出维度不匹配 | 类别数设置错误 | 修改 `ViT` 中 `num_classes` |
| 图片路径错误 | `txt` 文件路径不对 | 检查路径是否为绝对路径 |
| GPU 不可用 | CUDA 未安装或驱动问题 | 安装正确版本的 PyTorch 和 CUDA |
| 图像尺寸不匹配 | patch size 与 image size 不整除 | 调整 `patch_size` 或 `image_size` |

---

# 3 Vision Transformer（ViT）

## 3.1 ViT 简介
Vision Transformer（ViT）是 Google 于 2020 年提出的一种将 Transformer 架构应用于图像分类任务的模型。它的核心思想是：
> **将图像切分为小块（patch），再将这些 patch 当作“词”一样输入 Transformer 模型中进行处理。**
与传统 CNN 不同，ViT 不使用卷积提取局部特征，而是依赖自注意力机制建模全局关系。

---

## 3.2 ViT 架构详解

### 3.2.1 Patch Embedding（图像切块）
- 将输入图像（如 224×224）划分为固定大小的 patch（如 16×16）。
- 每个 patch 展平后通过线性层映射为向量（token）。
- 假设图像大小为 224×224，patch 大小为 16×16，则有：
  $$
  \frac{224 \times 224}{16 \times 16} = 196 \text{ 个 patch}
  $$

### 3.2.2 Position Embedding（位置编码）
- Transformer 不具备位置信息，因此需要为每个 patch 添加位置编码。
- 类似 NLP 中的 BERT，还添加一个 `cls_token`，用于最终分类。

### 3.2.3 Transformer Encoder
- 多层堆叠的 Transformer，每层包含：
  - 多头自注意力（Multi-Head Attention）
  - 前馈神经网络（FeedForward）
  - 残差连接 + LayerNorm

### 3.2.4 MLP Head（分类器）
- 最终只使用 `cls_token` 的输出，输入到 MLP 层进行分类。

---

## 3.3 ViT 模型代码解析
以下是一个简化版的 ViT 实现（使用 PyTorch + einops）：
```python
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        ...
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        ...
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        ...
        x = self.transformer(x)
        x = x[:, 0]  # 取 cls_token
        return self.mlp_head(x)
```
import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def normalize_band(band, min_val=0, max_val=10000):
    """归一化波段数据到 0-255"""
    band = np.clip(band, min_val, max_val)
    return ((band - min_val) / (max_val - min_val)) * 255

def shuchu(tif_file):
    """读取原始图像并转换为 RGB"""
    with rasterio.open(tif_file) as src:
        bands = src.read()  # 假设波段顺序为 B02, B03, B04, B08, B12
        red, green, blue = map(normalize_band, [bands[2], bands[1], bands[0]])
        rgb_image = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return rgb_image

def process_remote_sensing_image(input_path):
    """处理遥感图像并显示对比结果"""
    with rasterio.open(input_path) as src:
        data = src.read().astype(np.float32)

        # 计算 2%-98% 分位数拉伸
        min_vals = np.nanpercentile(data, 2, axis=(1, 2))
        max_vals = np.nanpercentile(data, 98, axis=(1, 2))
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # 避免除零错误

        # 归一化到 0-255
        scaled_data = (data - min_vals[:, None, None]) / range_vals[:, None, None] * 255
        scaled_data = np.clip(scaled_data, 0, 255).astype(np.uint8)

        # 构建 RGB 图像
        rgb_data = np.stack([scaled_data[2], scaled_data[1], scaled_data[0]], axis=-1)

    # 显示图像对比
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].imshow(shuchu(input_path))
    axs[0].axis("off")
    axs[0].set_title("原始 RGB 图像", fontsize=14)

    axs[1].imshow(rgb_data)
    axs[1].axis("off")
    axs[1].set_title("处理后 RGB 图像", fontsize=14)

    plt.tight_layout()
    plt.show()

# 示例调用
input_image = "2019_1101_nofire_B2348_B12_10m_roi.tif"
process_remote_sensing_image(input_image)

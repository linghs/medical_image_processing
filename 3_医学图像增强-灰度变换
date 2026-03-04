# 导入必要的库（需提前安装：pip install simpleitk matplotlib numpy）
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np  # 用于统计灰度值，计算直方图

# ---------------------- 原有核心函数（复用，不修改）----------------------
def read_dicom_and_plot_histogram(dicom_path, title_suffix=""):
    """
    读取DICOM文件、展示图像、绘制灰度直方图，并打印关键元信息
    核心功能：贴合灰度直方图章节知识点，直观呈现医学图像的灰度分布
    参数:
        dicom_path: DICOM文件的路径（如 "./*.dcm"，支持绝对/相对路径）
        title_suffix: 图像标题后缀（用于区分不同灰度变换效果）
    """
    # 检查文件是否存在，避免报错（提升代码健壮性）
    if not os.path.exists(dicom_path):
        print(f"错误：文件 {dicom_path} 不存在！请检查路径是否正确。")
        return
    
    try:
        # 1. 读取DICOM文件（医学图像专用读取方式，适配CT/MRI等模态）
        dicom_image = sitk.ReadImage(dicom_path)
        
        # 2. 获取DICOM元数据
        patient_id = dicom_image.GetMetaData("0010|0020")
        image_rows = dicom_image.GetMetaData("0028|0010")
        image_cols = dicom_image.GetMetaData("0028|0011")
        modality = dicom_image.GetMetaData("0008|0060")  # 模态：CT/MRI/X光等
        
        # 3. 将SimpleITK图像转换为numpy数组（便于后续灰度计算和可视化）
        image_array = sitk.GetArrayFromImage(dicom_image)
        image_array = image_array.squeeze()  # 去掉维度为1的轴（DICOM通常为(1,高,宽)）
        
        # 4. 打印DICOM关键信息
        print("="*50)
        print(f"DICOM 文件关键信息{title_suffix}：")
        print(f"患者ID: {patient_id}")
        print(f"影像模态: {modality}（决定图像灰度特征，如CT为HU值）")
        print(f"图像尺寸: {image_rows} x {image_cols}")
        print(f"像素数组形状: {image_array.shape}")
        print(f"像素值范围: {image_array.min()} ~ {image_array.max()}")
        print("="*50)
        
        # 5. 可视化：同时显示医学图像和对应的灰度直方图
        plt.figure(figsize=(12, 6))
        # 5.1 显示医学图像（左侧）
        plt.subplot(1, 2, 1)
        plt.imshow(image_array, cmap="gray")
        plt.axis("off")
        plt.title(f"DICOM Image {title_suffix} (Modality: {modality})", fontsize=12)
        # 5.2 绘制灰度直方图
        plt.subplot(1, 2, 2)
        hist, bins = np.histogram(image_array.flatten(), bins=np.arange(image_array.min(), image_array.max()+2))
        plt.plot(bins[:-1], hist, color='black', linewidth=1.5, label='Gray Level Histogram')
        # 标注峰值
        peak_idx = np.argmax(hist)
        peak_gray = bins[peak_idx]
        plt.axvline(x=peak_gray, color='red', linestyle='--', label=f'Peak: {peak_gray}')
        # 标注灰度范围
        plt.axvline(x=image_array.min(), color='blue', linestyle=':', label=f'Range: {image_array.min()}~{image_array.max()}')
        plt.axvline(x=image_array.max(), color='blue', linestyle=':')
        plt.xlabel('Gray Level (HU for CT, Signal Intensity for MRI)', fontsize=11)
        plt.ylabel('Number of Pixels', fontsize=11)
        plt.title(f'Gray Level Histogram {title_suffix}', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return image_array, modality  # 返回图像数组和模态，用于后续灰度变换
    
    except Exception as e:
        print(f"读取/处理DICOM文件失败：{e}")
        print("提示：请确保DICOM文件路径正确，且安装了所需库（simpleitk、matplotlib、numpy）。")
        return None, None

# ---------------------- 新增：灰度变换核心函数（重点）----------------------
def gray_transform(image_array, transform_type, gamma=1.5, threshold=100):
    """
    灰度变换核心函数，实现5种灰度变换方法，适配医学图像数组
    参数:
        image_array: 输入的图像numpy数组（从DICOM读取转换而来）
        transform_type: 变换类型（可选：stretch/normalize/gamma/log/invert/threshold）
        gamma: 伽马校正的gamma值（默认1.5，<1增强亮部，>1增强暗部）
        threshold: 阈值化的阈值（默认100，可根据图像灰度范围调整）
    返回:
        transformed_array: 灰度变换后的图像数组
    """
    # 先将图像数组转为float类型，避免整数运算溢出
    img_float = image_array.astype(np.float32)
    transformed_array = None
    
    # 1. 灰度拉伸（Stretch）：将原有灰度范围映射到0~255，增强对比度
    if transform_type == "stretch":
        min_val = img_float.min()
        max_val = img_float.max()
        # 核心公式：(原灰度值 - 最小值) / (最大值 - 最小值) * 255
        transformed_array = (img_float - min_val) / (max_val - min_val) * 255
        print(f"\n【灰度拉伸】：将灰度范围从 {min_val:.0f}~{max_val:.0f} 映射到 0~255")
    
    # 2. 归一化（Normalize）：映射到0~1或0~255（按需选择）
    elif transform_type == "normalize":
        min_val = img_float.min()
        max_val = img_float.max()
        # 映射到0~1（默认，适合后续AI模型输入）
        transformed_array = (img_float - min_val) / (max_val - min_val)
        # 若需映射到0~255，取消注释下方代码
        # transformed_array = ((img_float - min_val) / (max_val - min_val)) * 255
        print(f"\n【归一化】：将灰度范围从 {min_val:.0f}~{max_val:.0f} 映射到 0~1")
    
    # 3. 伽马校正（Gamma Correction）：调整图像亮度，增强暗部/亮部
    elif transform_type == "gamma":
        # 先归一化到0~1，再进行伽马变换，最后映射回原范围
        min_val = img_float.min()
        max_val = img_float.max()
        img_norm = (img_float - min_val) / (max_val - min_val)
        # 核心公式：output = img_norm ^ (1/gamma)（gamma>1增强暗部，gamma<1增强亮部）
        transformed_array = np.power(img_norm, 1/gamma)
        # 映射回原灰度范围，保持与原图一致性
        transformed_array = transformed_array * (max_val - min_val) + min_val
        print(f"\n【伽马校正】：gamma={gamma}，{'增强暗部' if gamma>1 else '增强亮部'}")
    
    # 4. 对数变换（Log Transform）：重点增强暗部细节（医学图像常用，如CT暗部组织）
    elif transform_type == "log":
        # 核心公式：output = c * log(1 + 原灰度值)，c为缩放系数（确保灰度范围合理）
        c = 255 / np.log(1 + img_float.max())  # 缩放系数，使结果映射到0~255
        transformed_array = c * np.log(1 + img_float)
        print(f"\n【对数变换】：增强暗部细节，缩放系数c={c:.2f}")
    
    # 5. 阈值化（Threshold）
    elif transform_type == "threshold":
        # 阈值化核心：灰度值>threshold设为255（白），≤threshold设为0（黑），用于分割目标（如病灶）
        transformed_array = np.where(img_float > threshold, 255, 0)
        print(f"\n【阈值化】：阈值={threshold}，灰度值>{threshold}设为255，否则设为0")
    
    else:
        print("错误：transform_type参数错误，可选值：stretch/normalize/gamma/log/invert/threshold")
        return image_array  # 返回原图，避免报错
    
    # 将数组转为uint8类型（适合图像显示，0~255范围）
    return transformed_array.astype(np.uint8) if transform_type != "normalize" else transformed_array

# ---------------------- 新增：一起显示所有图像的函数 ----------------------
def show_all_transformations(original_img, transformations_dict):
    """
    将原图和所有灰度变换后的图像一起显示，便于对比
    参数:
        original_img: 原始图像数组
        transformations_dict: 包含变换名称和对应图像的字典
    """
    n_transforms = len(transformations_dict) + 1  # +1 是原图
    # 计算合适的网格大小（尽可能接近方形）
    n_cols = min(3, n_transforms)  # 最多3列
    n_rows = (n_transforms + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    # 显示原图
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_img, cmap="gray")
    plt.axis("off")
    plt.title("Original Image", fontsize=10, fontweight='bold')
    
    # 显示各种变换后的图像
    for i, (title, img) in enumerate(transformations_dict.items(), start=2):
        plt.subplot(n_rows, n_cols, i)
        
        # 处理归一化图像（范围0-1）的显示
        if title == "Normalize (0-1)":
            display_img = img * 255  # 临时映射到0-255显示
        else:
            display_img = img
            
        plt.imshow(display_img, cmap="gray")
        plt.axis("off")
        plt.title(title, fontsize=10)
    
    plt.suptitle("Comparison of Different Gray Level Transformations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 主程序入口（可直接运行，替换DICOM路径即可）
if __name__ == "__main__":
    # 1. 替换为你的DICOM文件路径（绝对路径/相对路径均可）
    dicom_file_path = "image-00000.dcm"
    
    # 2. 读取DICOM原图，获取图像数组和模态（复用原有函数）
    original_img, modality = read_dicom_and_plot_histogram(dicom_file_path, title_suffix="(Original)")
    if original_img is None:
        exit()  # 若读取失败，退出程序
    
    # 3. 对原图应用所有灰度变换，并保存结果
    transformations = {}
    
    # 3.1 灰度拉伸
    stretch_img = gray_transform(original_img, transform_type="stretch")
    transformations["Stretch (0-255)"] = stretch_img
    
    # 3.2 归一化（默认0~1）
    normalize_img = gray_transform(original_img, transform_type="normalize")
    transformations["Normalize (0-1)"] = normalize_img
    
    # 3.3 伽马校正（gamma=1.5，增强暗部）
    gamma_img = gray_transform(original_img, transform_type="gamma", gamma=1.5)
    transformations["Gamma (γ=1.5)"] = gamma_img
    
    # 3.4 对数变换
    log_img = gray_transform(original_img, transform_type="log")
    transformations["Log Transform"] = log_img
    
    # 3.5 阈值化（阈值可调整，根据原图灰度范围修改）
    # 计算原图灰度均值作为阈值参考
    threshold_value = int(original_img.mean())  # 使用均值作为阈值
    threshold_img = gray_transform(original_img, transform_type="threshold", threshold=threshold_value)
    transformations[f"Threshold (T={threshold_value})"] = threshold_img
    
    # 4. 一起显示所有图像
    print("\n" + "="*60)
    print("所有灰度变换完成，正在生成对比图...")
    print("="*60)
    show_all_transformations(original_img, transformations)
    

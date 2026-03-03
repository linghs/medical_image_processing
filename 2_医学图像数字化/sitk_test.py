# 导入必要的库（需提前安装：pip install simpleitk matplotlib numpy）
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np  # 用于统计灰度值，计算直方图

def read_dicom_and_plot_histogram(dicom_path):
    """
    读取DICOM文件、展示图像、绘制灰度直方图，并打印关键元信息
    核心功能：贴合灰度直方图章节知识点，直观呈现医学图像的灰度分布
    参数:
        dicom_path: DICOM文件的路径（如 "./*.dcm"，支持绝对/相对路径）
    """
    # 检查文件是否存在，避免报错（提升代码健壮性）
    if not os.path.exists(dicom_path):
        print(f"错误：文件 {dicom_path} 不存在！请检查路径是否正确。")
        return
    
    try:
        # 1. 读取DICOM文件（医学图像专用读取方式，适配CT/MRI等模态）
        dicom_image = sitk.ReadImage(dicom_path)
        
        # 2. 获取DICOM元数据
        # 常用DICOM标签说明：
        # (0010,0020) 患者ID | (0028,0010) 图像行数 | (0028,0011) 图像列数
        # (0028,0100) 像素位深 | (0008,0060) 模态（CT/MRI/US等）
        patient_id = dicom_image.GetMetaData("0010|0020")  # 分隔符为 |，注意与逗号区分
        image_rows = dicom_image.GetMetaData("0028|0010")
        image_cols = dicom_image.GetMetaData("0028|0011")
        modality = dicom_image.GetMetaData("0008|0060")  # 模态：CT/MRI/X光等
        
        # 3. 将SimpleITK图像转换为numpy数组（便于后续灰度直方图计算和可视化）
        image_array = sitk.GetArrayFromImage(dicom_image)
        # DICOM图像数组形状通常为 (1, 高, 宽)，去掉维度为1的轴（简化数据）
        image_array = image_array.squeeze()
        
        # 4. 打印DICOM关键信息（呼应章节4.1 DICOM格式的核心优势：存储元数据）
        print("="*50)
        print("DICOM 文件关键信息（贴合章节4.1知识点）：")
        print(f"患者ID: {patient_id}")
        print(f"影像模态: {modality}（决定图像灰度特征，如CT为HU值）")
        print(f"图像尺寸: {image_rows} x {image_cols}（对应章节2.3的空间分辨率）")
        print(f"像素数组形状: {image_array.shape}")
        print(f"像素值范围: {image_array.min()} ~ {image_array.max()}（对应章节5.2灰度范围）")
        print("="*50)
        
        # 5. 可视化：同时显示医学图像和对应的灰度直方图（核心功能，贴合章节5）
        plt.figure(figsize=(12, 6))  # 设置画布大小，便于对比查看
        
        # 5.1 显示医学图像（左侧）
        plt.subplot(1, 2, 1)  # 1行2列，第1个子图
        plt.imshow(image_array, cmap="gray")  # 灰度色板，适配医学影像显示
        plt.axis("off")  # 隐藏坐标轴，聚焦图像本身
        plt.title(f"DICOM Image (Modality: {modality})", fontsize=12)  # 已为英文标题
        
        # 5.2 绘制灰度直方图
        plt.subplot(1, 2, 2)  # 1行2列，第2个子图
        # 计算灰度直方图：统计每个灰度值的像素数量（对应章节5.1直方图定义）
        # bins：直方图的灰度级数量，根据像素值范围自适应，确保覆盖所有灰度
        hist, bins = np.histogram(image_array.flatten(), bins=np.arange(image_array.min(), image_array.max()+2))
        # 绘制直方图曲线，标注核心参数（对应章节5.2的峰值、灰度范围）
        plt.plot(bins[:-1], hist, color='black', linewidth=1.5, label='Gray Level Histogram')  # 英文标签
        # 标注峰值
        peak_idx = np.argmax(hist)
        peak_gray = bins[peak_idx]
        plt.axvline(x=peak_gray, color='red', linestyle='--', label=f'Peak Gray Level: {peak_gray}')  # 英文标签
        # 标注灰度范围
        plt.axvline(x=image_array.min(), color='blue', linestyle=':', label=f'Gray Level Range: {image_array.min()}~{image_array.max()}')  # 英文标签
        plt.axvline(x=image_array.max(), color='blue', linestyle=':')
        
        # 直方图标注
        plt.xlabel('Gray Level (HU for CT, Signal Intensity for MRI)', fontsize=11)  # 英文x轴标签
        plt.ylabel('Number of Pixels', fontsize=11)  # 英文y轴标签
        plt.title('Gray Level Histogram (Consistent with Chapter 5 Knowledge)', fontsize=12)  # 英文标题
        plt.legend()  # 显示图例，便于解读
        plt.tight_layout()  # 调整布局，避免重叠
        plt.show()
        
        # 补充直方图解读
        print("\n【灰度直方图解读（贴合章节5知识点）】")
        print(f"1. 峰值灰度值 {peak_gray}：对应图像中出现频率最高的组织（如CT中骨骼灰度值高、软组织居中）；")
        print(f"2. 灰度范围 {image_array.min()}~{image_array.max()}：范围越大，图像对比度越高（如CT可区分骨骼、软组织、空气）；")
        print(f"3. 若直方图峰值集中、谷值不明显，说明图像对比度低（对应章节5.3，可通过直方图均衡化改善）。")
        
    except Exception as e:
        print(f"读取/处理DICOM文件失败：{e}")
        print("提示：请确保DICOM文件路径正确，且安装了所需库（simpleitk、matplotlib、numpy）。")

# 主程序入口（可直接运行，替换路径即可）
if __name__ == "__main__":
    # 替换为你的DICOM文件路径（支持绝对路径，如"C:/dicom/image-00000.dcm"；或相对路径）
    dicom_file_path = "image-00000.dcm"
    # 调用函数：读取DICOM、显示图像、绘制灰度直方图
    read_dicom_and_plot_histogram(dicom_file_path)

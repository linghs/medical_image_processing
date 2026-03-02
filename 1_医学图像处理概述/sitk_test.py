# 导入必要的库
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

def read_and_show_dicom(dicom_path):
    """
    读取DICOM文件并展示图像，同时打印关键元信息
    
    参数:
        dicom_path: DICOM文件的路径（如 "./*.dcm"）
    """
    # 检查文件是否存在
    if not os.path.exists(dicom_path):
        print(f"错误：文件 {dicom_path} 不存在！")
        return
    
    try:
        # 1. 读取DICOM文件
        dicom_image = sitk.ReadImage(dicom_path)
        
        # 2. 获取DICOM元数据（标签对应的值）
        # 常用标签说明：
        # (0010,0020) 患者ID | (0028,0010) 图像行数 | (0028,0011) 图像列数
        # (0028,0100) 像素位深 | (0008,0060) 模态（CT/MRI/US等）
        patient_id = dicom_image.GetMetaData("0010|0020")  # 注意分隔符是 | 而非 ,
        image_rows = dicom_image.GetMetaData("0028|0010")
        image_cols = dicom_image.GetMetaData("0028|0011")
        modality = dicom_image.GetMetaData("0008|0060")
        
        # 3. 将SimpleITK图像转换为numpy数组（便于可视化）
        image_array = sitk.GetArrayFromImage(dicom_image)
        # DICOM数组形状通常是 (1, 高, 宽)，去掉维度为1的轴
        image_array = image_array.squeeze()
        
        # 4. 打印关键信息
        print("="*50)
        print("DICOM 文件关键信息：")
        print(f"患者ID: {patient_id}")
        print(f"影像模态: {modality}")
        print(f"图像尺寸: {image_rows} x {image_cols}")
        print(f"像素数组形状: {image_array.shape}")
        print(f"像素值范围: {image_array.min()} ~ {image_array.max()}")
        print("="*50)
        
        # 5. 可视化图像（调整灰度范围适配医学影像显示）
        plt.figure(figsize=(8, 8))
        plt.imshow(image_array, cmap="gray")  # 灰度色板适配医学影像
        plt.axis("off")  # 隐藏坐标轴
        plt.title(f"DICOM Image - Modality: {modality}", fontsize=12)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"读取/展示DICOM文件失败：{e}")

# 主程序入口
if __name__ == "__main__":
    # 替换为你的DICOM文件路径（支持绝对路径/相对路径）
    dicom_file_path = "image-00000.dcm"
    # 调用函数读取并展示
    read_and_show_dicom(dicom_file_path)

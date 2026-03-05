# 导入必要的库（需提前安装：pip install simpleitk matplotlib numpy scipy）
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import convolve
import warnings
warnings.filterwarnings('ignore')

def apply_filters(image_array):
    """
    Apply mean and Gaussian filters to the image
    
    Parameters:
        image_array: Input numpy array image
    
    Returns:
        Dictionary with original, mean_filtered, gaussian_filtered images
    """
    # Ensure image is float type for calculation
    image_float = image_array.astype(np.float32)
    
    # 1. Mean filter - using 3x3 uniform kernel
    mean_kernel = np.ones((3, 3)) / 9.0  # 3x3 template, sum of weights = 1
    mean_filtered = convolve(image_float, mean_kernel)
    
    # 2. Gaussian filter - using 3x3 Gaussian kernel
    # Standard Gaussian kernel weights: higher weight closer to center
    gaussian_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0  # Divided by sum of weights (16) to make total weight = 1
    gaussian_filtered = convolve(image_float, gaussian_kernel)
    
    # Convert back to original data type (uint8/uint16 etc.)
    mean_filtered = np.clip(mean_filtered, image_array.min(), image_array.max()).astype(image_array.dtype)
    gaussian_filtered = np.clip(gaussian_filtered, image_array.min(), image_array.max()).astype(image_array.dtype)
    
    return {
        'original': image_array,
        'mean_filtered': mean_filtered,
        'gaussian_filtered': gaussian_filtered
    }

def calculate_filter_stats(original, filtered, filter_name):
    """
    Calculate statistical metrics before and after filtering
    
    Parameters:
        original: Original image array
        filtered: Filtered image array
        filter_name: Name of the filter
    
    Returns:
        Dictionary of statistical information
    """
    # Calculate difference
    diff = filtered.astype(np.float32) - original.astype(np.float32)
    
    # Statistical metrics
    stats = {
        'filter_name': filter_name,
        'original_mean': np.mean(original),
        'filtered_mean': np.mean(filtered),
        'original_std': np.std(original),
        'filtered_std': np.std(filtered),
        'max_change': np.max(np.abs(diff)),
        'mean_abs_change': np.mean(np.abs(diff)),
        'original_range': original.max() - original.min(),
        'filtered_range': filtered.max() - filtered.min()
    }
    
    return stats

def read_dicom_and_plot_comparison(dicom_path):
    """
    Read DICOM file, display images, and compare original with filtered effects
    Core function: Demonstrates knowledge points of smoothing filters
    """
    # Check if file exists
    if not os.path.exists(dicom_path):
        print(f"Error: File {dicom_path} does not exist! Please check the path.")
        return
    
    try:
        # 1. Read DICOM file
        dicom_image = sitk.ReadImage(dicom_path)
        
        # 2. Get DICOM metadata
        try:
            patient_id = dicom_image.GetMetaData("0010|0020")
        except:
            patient_id = "Unknown"
        
        try:
            image_rows = dicom_image.GetMetaData("0028|0010")
            image_cols = dicom_image.GetMetaData("0028|0011")
        except:
            image_size = dicom_image.GetSize()
            image_rows, image_cols = image_size[1], image_size[0]
        
        try:
            modality = dicom_image.GetMetaData("0008|0060")
        except:
            modality = "Unknown"
        
        # 3. Convert SimpleITK image to numpy array
        image_array = sitk.GetArrayFromImage(dicom_image)
        # DICOM image array shape is usually (1, height, width) or (height, width)
        if len(image_array.shape) == 3:
            image_array = image_array.squeeze()
        
        print("="*60)
        print("DICOM File Key Information:")
        print(f"Patient ID: {patient_id}")
        print(f"Modality: {modality}")
        print(f"Image Size: {image_rows} x {image_cols}")
        print(f"Pixel Array Shape: {image_array.shape}")
        print(f"Pixel Value Range: {image_array.min()} ~ {image_array.max()}")
        print(f"Pixel Data Type: {image_array.dtype}")
        print("="*60)
        
        # 4. Apply filters
        filtered_images = apply_filters(image_array)
        
        # 5. Calculate statistical metrics
        mean_stats = calculate_filter_stats(image_array, filtered_images['mean_filtered'], "Mean Filter")
        gaussian_stats = calculate_filter_stats(image_array, filtered_images['gaussian_filtered'], "Gaussian Filter")
        
        # 6. Print filter comparison information
        print("\n【Smoothing Filter Effect Comparison (Knowledge Point Validation)】")
        print("-"*50)
        print(f"Mean Filter Effect:")
        print(f"  - Original Mean: {mean_stats['original_mean']:.2f} → Filtered Mean: {mean_stats['filtered_mean']:.2f}")
        print(f"  - Original Std: {mean_stats['original_std']:.2f} → Filtered Std: {mean_stats['filtered_std']:.2f}")
        print(f"  - Gray Range Change: {mean_stats['original_range']} → {mean_stats['filtered_range']}")
        print(f"  - Max Pixel Change: {mean_stats['max_change']:.2f}")
        print(f"  - Mean Absolute Change: {mean_stats['mean_abs_change']:.2f}")
        
        print(f"\nGaussian Filter Effect:")
        print(f"  - Original Mean: {gaussian_stats['original_mean']:.2f} → Filtered Mean: {gaussian_stats['filtered_mean']:.2f}")
        print(f"  - Original Std: {gaussian_stats['original_std']:.2f} → Filtered Std: {gaussian_stats['filtered_std']:.2f}")
        print(f"  - Gray Range Change: {gaussian_stats['original_range']} → {gaussian_stats['filtered_range']}")
        print(f"  - Max Pixel Change: {gaussian_stats['max_change']:.2f}")
        print(f"  - Mean Absolute Change: {gaussian_stats['mean_abs_change']:.2f}")
        print("-"*50)
        
        print("\n【Smoothing Filter Theory Validation】")
        print("1. Standard Deviation Reduction: Decreased std after filtering indicates smoother gray level variation (noise suppression)")
        print("2. Gray Range Reduction: Filtered gray range typically narrows, demonstrating smoothing effect")
        print("3. Mean Filter vs Gaussian Filter:")
        print("   - Mean Filter: Larger pixel changes (higher mean absolute change), more blurred edges")
        print("   - Gaussian Filter: Preserves more details (smaller changes), better edge preservation")
        
        # 7. Visualization: Original image, filtered results (without histograms)
        plt.figure(figsize=(18, 6))
        
        # 7.1 Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image_array, cmap="gray")
        plt.axis("off")
        plt.title(f"Original Image\n(Modality: {modality})", fontsize=12)
        
        # Add text annotation for original image stats
        plt.text(0.5, -0.1, f"Mean: {mean_stats['original_mean']:.1f}, Std: {mean_stats['original_std']:.1f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=10)
        
        # 7.2 Mean filter result
        plt.subplot(1, 3, 2)
        plt.imshow(filtered_images['mean_filtered'], cmap="gray")
        plt.axis("off")
        plt.title("Mean Filter (3×3 Kernel)\nUniform Weighting", fontsize=12)
        
        # Add text annotation for mean filter stats
        plt.text(0.5, -0.1, f"Mean: {mean_stats['filtered_mean']:.1f}, Std: {mean_stats['filtered_std']:.1f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=10)
        plt.text(0.5, -0.18, f"Mean Abs Change: {mean_stats['mean_abs_change']:.2f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=9, color='blue')
        
        # 7.3 Gaussian filter result
        plt.subplot(1, 3, 3)
        plt.imshow(filtered_images['gaussian_filtered'], cmap="gray")
        plt.axis("off")
        plt.title("Gaussian Filter (3×3 Kernel)\nCenter-Weighted", fontsize=12)
        
        # Add text annotation for Gaussian filter stats
        plt.text(0.5, -0.1, f"Mean: {gaussian_stats['filtered_mean']:.1f}, Std: {gaussian_stats['filtered_std']:.1f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=10)
        plt.text(0.5, -0.18, f"Mean Abs Change: {gaussian_stats['mean_abs_change']:.2f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=9, color='red')
        
        plt.suptitle("Medical Image Smoothing Filter Comparison", fontsize=14, y=1.05)
        plt.tight_layout()
        plt.show()
        
        # 8. Local magnification comparison (showing filtering effect in central region)
        # Select a central region for magnification
        h, w = image_array.shape
        center_h, center_w = h // 2, w // 2
        crop_size = min(120, h // 3, w // 3)
        
        # Ensure crop region doesn't exceed image boundaries
        crop_h_start = max(0, center_h - crop_size // 2)
        crop_h_end = min(h, center_h + crop_size // 2)
        crop_w_start = max(0, center_w - crop_size // 2)
        crop_w_end = min(w, center_w + crop_size // 2)
        
        plt.figure(figsize=(16, 5))
        
        # Local original image
        plt.subplot(1, 4, 1)
        plt.imshow(image_array[crop_h_start:crop_h_end, crop_w_start:crop_w_end], cmap="gray")
        plt.axis("off")
        plt.title("Original\n(Central Region)", fontsize=11)
        
        # Local mean filter
        plt.subplot(1, 4, 2)
        plt.imshow(filtered_images['mean_filtered'][crop_h_start:crop_h_end, crop_w_start:crop_w_end], cmap="gray")
        plt.axis("off")
        plt.title("Mean Filter\n(More Smoothing)", fontsize=11)
        
        # Local Gaussian filter
        plt.subplot(1, 4, 3)
        plt.imshow(filtered_images['gaussian_filtered'][crop_h_start:crop_h_end, crop_w_start:crop_w_end], cmap="gray")
        plt.axis("off")
        plt.title("Gaussian Filter\n(Better Edge Preservation)", fontsize=11)
        
        # Difference map (Gaussian - Mean)
        plt.subplot(1, 4, 4)
        diff_image = np.abs(filtered_images['gaussian_filtered'].astype(np.float32) - 
                           filtered_images['mean_filtered'].astype(np.float32))
        
        # Normalize difference for better visualization
        if diff_image.max() > 0:
            diff_display = diff_image / diff_image.max()
        else:
            diff_display = diff_image
            
        im = plt.imshow(diff_display[crop_h_start:crop_h_end, crop_w_start:crop_w_end], cmap="hot")
        plt.axis("off")
        plt.title("Gaussian vs Mean\nDifference Map", fontsize=11)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Normalized Difference")
        
        plt.suptitle("Local Region Comparison (Magnified View)", fontsize=14, y=1.05)
        plt.tight_layout()
        plt.show()
        
        # Additional interpretation
        print("\n【Detailed Filter Effect Analysis】")
        print("1. Mean Filter Effect:")
        print("   - All pixels participate equally in averaging, providing strong noise suppression")
        print(f"   - Gray range reduced from {mean_stats['original_range']} to {mean_stats['filtered_range']}")
        print("   - Disadvantage: Edges and details are blurred equally")
        
        print("\n2. Gaussian Filter Effect:")
        print("   - Center pixel has highest weight, preserving more original information")
        print(f"   - Mean absolute change: {gaussian_stats['mean_abs_change']:.2f} < Mean filter's {mean_stats['mean_abs_change']:.2f}")
        print("   - Advantage: Better edge preservation while denoising")
        
        print("\n3. Statistical Comparison:")
        print(f"   - Mean filter reduces std by: {mean_stats['original_std'] - mean_stats['filtered_std']:.2f}")
        print(f"   - Gaussian filter reduces std by: {gaussian_stats['original_std'] - gaussian_stats['filtered_std']:.2f}")
        
        if mean_stats['mean_abs_change'] > gaussian_stats['mean_abs_change']:
            print("   - Mean filter causes more aggressive smoothing (higher pixel changes)")
        else:
            print("   - Gaussian filter preserves more original pixel values")
            
    except Exception as e:
        print(f"Error reading/processing DICOM file: {e}")
        import traceback
        traceback.print_exc()

# Main program entry
if __name__ == "__main__":
    # Replace with your DICOM file path
    dicom_file_path = "image-00000.dcm"  # Please modify to your actual DICOM file path
    
    print("="*60)
    print("Medical Image Smoothing Filter Effect Demonstration")
    print("Knowledge Points: Mean Filter vs Gaussian Filter")
    print("="*60)
    
    # Call function
    read_dicom_and_plot_comparison(dicom_file_path)

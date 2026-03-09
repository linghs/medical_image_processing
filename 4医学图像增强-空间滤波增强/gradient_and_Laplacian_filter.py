# Import necessary libraries
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import convolve
import warnings
warnings.filterwarnings('ignore')

def apply_sharpening_filters(image_array):
    """
    Apply gradient (Sobel) and Laplacian sharpening filters to the image
    """
    # Ensure image is float type for calculation
    image_float = image_array.astype(np.float32)
    
    # 1. Sobel gradient (edge detection)
    # Sobel operator - horizontal
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    # Sobel operator - vertical
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    # Calculate horizontal and vertical gradients
    gradient_x = convolve(image_float, sobel_x)
    gradient_y = convolve(image_float, sobel_y)
    
    # Calculate gradient magnitude (edge strength)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize gradient magnitude to 0-255 range for display
    if gradient_magnitude.max() > 0:
        gradient_normalized = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    else:
        gradient_normalized = gradient_magnitude.astype(np.uint8)
    
    # 2. Laplacian sharpening
    # Laplacian operator (4-neighborhood)
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    
    # Calculate Laplacian response
    laplacian_response = convolve(image_float, laplacian_kernel)
    
    # Laplacian sharpening: original - Laplacian response
    laplacian_sharpened = image_float - laplacian_response
    
    # Clip to valid range
    laplacian_sharpened = np.clip(laplacian_sharpened, image_array.min(), image_array.max())
    laplacian_sharpened = laplacian_sharpened.astype(image_array.dtype)
    
    # Absolute Laplacian response (for display)
    laplacian_abs = np.abs(laplacian_response)
    if laplacian_abs.max() > 0:
        laplacian_abs_normalized = (laplacian_abs / laplacian_abs.max() * 255).astype(np.uint8)
    else:
        laplacian_abs_normalized = laplacian_abs.astype(np.uint8)
    
    return {
        'original': image_array,
        'gradient_magnitude': gradient_normalized,
        'laplacian_sharpened': laplacian_sharpened,
        'laplacian_response': laplacian_abs_normalized,
        'gradient_x': gradient_x,
        'gradient_y': gradient_y
    }

def read_dicom_and_plot_sharpening(dicom_path):
    """
    Read DICOM file and display gradient vs Laplacian sharpening results
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
            modality = dicom_image.GetMetaData("0008|0060")
        except:
            modality = "Unknown"
        
        # 3. Convert SimpleITK image to numpy array
        image_array = sitk.GetArrayFromImage(dicom_image)
        if len(image_array.shape) == 3:
            image_array = image_array.squeeze()
        
        print("="*60)
        print("DICOM FILE INFORMATION")
        print("="*60)
        print(f"Patient ID: {patient_id}")
        print(f"Modality: {modality}")
        print(f"Image Size: {image_array.shape}")
        print(f"Pixel Range: {image_array.min()} ~ {image_array.max()}")
        print(f"Data Type: {image_array.dtype}")
        print("="*60)
        
        # 4. Apply sharpening filters
        results = apply_sharpening_filters(image_array)
        
        # 5. Calculate statistics
        original_mean = np.mean(image_array)
        original_std = np.std(image_array)
        gradient_mean = np.mean(results['gradient_magnitude'])
        laplacian_mean = np.mean(results['laplacian_sharpened'])
        laplacian_std = np.std(results['laplacian_sharpened'])
        
        print("\n【SHARPENING FILTER STATISTICS】")
        print("-"*50)
        print("Original Image:")
        print(f"  - Mean: {original_mean:.2f}")
        print(f"  - Std Dev: {original_std:.2f}")
        
        print("\nGradient Method (Sobel):")
        print(f"  - Edge Intensity Mean: {gradient_mean:.2f}")
        print(f"  - Edge Intensity Range: {results['gradient_magnitude'].min()} ~ {results['gradient_magnitude'].max()}")
        
        print("\nLaplacian Sharpening:")
        print(f"  - Sharpened Mean: {laplacian_mean:.2f}")
        print(f"  - Sharpened Std Dev: {laplacian_std:.2f}")
        print("-"*50)
        
        # 6. Visualization: Original, Gradient, Laplacian Response, Sharpened
        plt.figure(figsize=(20, 5))
        
        # 6.1 Original image
        plt.subplot(1, 4, 1)
        plt.imshow(image_array, cmap="gray")
        plt.axis("off")
        plt.title("Original Image", fontsize=14)
        plt.text(0.5, -0.1, f"Mean: {original_mean:.1f}, Std: {original_std:.1f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=11)
        
        # 6.2 Gradient magnitude (Sobel)
        plt.subplot(1, 4, 2)
        plt.imshow(results['gradient_magnitude'], cmap="gray")
        plt.axis("off")
        plt.title("Gradient Method (Sobel)\nFirst Derivative", fontsize=14)
        plt.text(0.5, -0.1, f"Edge Intensity: {gradient_mean:.1f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=11)
        
        # 6.3 Laplacian response
        plt.subplot(1, 4, 3)
        plt.imshow(results['laplacian_response'], cmap="gray")
        plt.axis("off")
        plt.title("Laplacian Response\nSecond Derivative", fontsize=14)
        plt.text(0.5, -0.1, f"Response Mean: {np.mean(results['laplacian_response']):.1f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=11)
        
        # 6.4 Laplacian sharpened
        plt.subplot(1, 4, 4)
        plt.imshow(results['laplacian_sharpened'], cmap="gray")
        plt.axis("off")
        plt.title("Laplacian Sharpened\nf - ∇²f", fontsize=14)
        plt.text(0.5, -0.1, f"Mean: {laplacian_mean:.1f}, Std: {laplacian_std:.1f}", 
                 transform=plt.gca().transAxes, ha='center', fontsize=11)
        
        plt.suptitle(f"Medical Image Sharpening Filters Comparison\nModality: {modality}", 
                    fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()
        
        
        # Print summary
        print("\n" + "="*60)
        print("SHARPENING FILTERS SUMMARY")
        print("="*60)
        
        print("\n1. Gradient Method (Sobel Operator):")
        print("   - First-order derivative")
        print("   - Detects edges by finding maximum intensity changes")
        print("   - Output: Edge strength map")
        print("   - Kernel: Gx = [-1 0 1; -2 0 2; -1 0 1] (horizontal)")
        print("            Gy = [-1 -2 -1; 0 0 0; 1 2 1] (vertical)")
        print("   - Magnitude: |G| = √(Gx² + Gy²)")
        
        print("\n2. Laplacian Operator:")
        print("   - Second-order derivative")
        print("   - Isotropic (rotation invariant)")
        print("   - Sensitive to all edge directions equally")
        print("   - Kernel: L = [0 -1 0; -1 4 -1; 0 -1 0]")
        print("   - Sharpening: g = f - ∇²f")
        
        print("\n3. Key Differences:")
        print("   - Gradient: Produces thick edges, directional information")
        print("   - Laplacian: Produces thin edges, zero-crossings, more noise sensitive")
        
        print("\n4. Medical Applications:")
        print("   - Gradient: Edge detection, organ boundary extraction, size measurement")
        print("   - Laplacian: Image sharpening, enhancing模糊病灶 boundaries")
        
    except Exception as e:
        print(f"Error processing DICOM file: {e}")
        import traceback
        traceback.print_exc()

# Main program entry
if __name__ == "__main__":
    dicom_file_path = "image-00000.dcm"  # Modify to your DICOM file path
    
    print("="*60)
    print("MEDICAL IMAGE SHARPENING FILTERS")
    print("Gradient Method (Sobel) vs Laplacian Sharpening")
    print("="*60)
    
    read_dicom_and_plot_sharpening(dicom_file_path)

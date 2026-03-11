import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import os

def dicom_fourier_transform_sitk_simple(dicom_file_path):
    """
    Read DICOM image using SimpleITK and perform Fourier transform
    Display only original image and its Fourier spectrum
    
    Parameters:
    -----------
    dicom_file_path : str
        Path to the DICOM file
    """
    
    # 1. Read DICOM file using SimpleITK
    try:
        # Read DICOM file
        image = sitk.ReadImage(dicom_file_path)
        
        # Convert to numpy array
        image_array = sitk.GetArrayFromImage(image)
        
        # Handle 3D images by taking the middle slice
        if len(image_array.shape) == 3:
            print(f"3D image detected, shape: {image_array.shape}")
            if image_array.shape[0] > 1:
                middle_slice = image_array.shape[0] // 2
                image_array = image_array[middle_slice, :, :]
                print(f"Using middle slice {middle_slice} for display")
            else:
                image_array = image_array[0, :, :]
        elif len(image_array.shape) == 2:
            print(f"2D image detected, shape: {image_array.shape}")
        
        # Ensure image is 2D
        if len(image_array.shape) != 2:
            print(f"Cannot process {len(image_array.shape)}D image")
            return None
        
        # Convert to float type
        image_array = image_array.astype(float)
        
        print(f"\nImage info:")
        print(f"Size: {image_array.shape}")
        print(f"Pixel type: {image.GetPixelIDTypeAsString()}")
        print(f"Spacing: {image.GetSpacing()}")
        
    except Exception as e:
        print(f"Error reading DICOM file with SimpleITK: {e}")
        return None
    
    # 2. Perform 2D Fourier transform
    # Compute Fourier transform
    f_transform = fftpack.fft2(image_array)
    
    # Shift zero frequency to center
    f_transform_shifted = fftpack.fftshift(f_transform)
    
    # 3. Calculate magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)
    
    # 4. Apply log transform for better visualization
    # Add small constant to avoid log(0)
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)  # log(1 + magnitude)
    
    # 5. Display only original image and Fourier spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    im1 = axes[0].imshow(image_array, cmap='gray')
    axes[0].set_title('Original DICOM Image', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Pixel Intensity')
    
    # Fourier spectrum (log transformed)
    im2 = axes[1].imshow(magnitude_spectrum_log, cmap='gray')
    axes[1].set_title('Fourier Spectrum\n(Log Transformed)', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Log Magnitude')
    
    plt.suptitle(f'Fourier Transform of DICOM Image\nFile: {os.path.basename(dicom_file_path)}', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print basic statistics
    print("\n" + "="*50)
    print("FOURIER TRANSFORM STATISTICS")
    print("="*50)
    print(f"Original image - Min: {np.min(image_array):.2f}, Max: {np.max(image_array):.2f}, Mean: {np.mean(image_array):.2f}")
    print(f"Fourier spectrum - Min: {np.min(magnitude_spectrum):.2f}, Max: {np.max(magnitude_spectrum):.2f}")
    print(f"Spectrum dynamic range: {np.max(magnitude_spectrum)/np.maximum(np.min(magnitude_spectrum), 1):.2e}")
    
    # Calculate energy concentration in low frequencies
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    center_energy = np.sum(magnitude_spectrum[
        center_h-10:center_h+10, 
        center_w-10:center_w+10
    ])
    total_energy = np.sum(magnitude_spectrum)
    print(f"Energy in center 20x20 region: {center_energy/total_energy*100:.2f}% of total")
    
    return {
        'original': image_array,
        'original_sitk': image,
        'fourier': f_transform_shifted,
        'magnitude': magnitude_spectrum,
        'magnitude_log': magnitude_spectrum_log
    }

# Alternative version with even simpler display (just the two images without colorbars)
def dicom_fourier_transform_ultra_simple(dicom_file_path):
    """
    Ultra simple version - displays only original image and Fourier spectrum
    No colorbars, minimal styling
    """
    
    try:
        # Read DICOM
        image = sitk.ReadImage(dicom_file_path)
        image_array = sitk.GetArrayFromImage(image)
        
        # Handle 3D
        if len(image_array.shape) == 3:
            middle_slice = image_array.shape[0] // 2
            image_array = image_array[middle_slice, :, :]
        elif len(image_array.shape) != 2:
            print(f"Unsupported image dimension: {len(image_array.shape)}")
            return None
        
        image_array = image_array.astype(float)
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Fourier transform
    f_transform = fftpack.fft2(image_array)
    f_transform_shifted = fftpack.fftshift(f_transform)
    magnitude_spectrum_log = np.log1p(np.abs(f_transform_shifted))
    
    # Simple display
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum_log, cmap='gray')
    plt.title('Fourier Spectrum', fontsize=12)
    plt.axis('off')
    
    plt.suptitle(f'Fourier Transform: {os.path.basename(dicom_file_path)}', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    return image_array, magnitude_spectrum_log

# Main program
if __name__ == "__main__":
    dicom_file_path = "image-00000.dcm"
    
    # Check if file exists
    if not os.path.exists(dicom_file_path):
        print(f"Error: File '{dicom_file_path}' not found!")
        print("Please ensure the DICOM file is in the current directory.")
        
        # List .dcm files in current directory
        dcm_files = [f for f in os.listdir('.') if f.lower().endswith('.dcm')]
        if dcm_files:
            print(f"\nDICOM files found in current directory:")
            for i, f in enumerate(dcm_files[:5]):  # Show first 5
                print(f"  {i+1}. {f}")
            print("\nPlease modify the filename in the code to one of these.")
    else:
        print("="*60)
        print("FOURIER TRANSFORM OF DICOM IMAGE")
        print("="*60)
        print(f"Processing: {dicom_file_path}")
        
        # Run the simple version
        result = dicom_fourier_transform_sitk_simple(dicom_file_path)
        
        # Uncomment below for ultra simple version
        # result = dicom_fourier_transform_ultra_simple(dicom_file_path)

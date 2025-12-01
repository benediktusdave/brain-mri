"""
Brain MRI Tumor Detection using Classical Image Processing
Pipeline: FFT Sharpening -> Segmentation -> Morphology
Streamlit Web Application
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# ==================== FUNCTIONS ====================

def fft_sharpen(image, hpf_radius=20):
    """
    Apply FFT-based sharpening using High-Pass Filter (Unsharp Masking)
    
    Parameters:
    - image: grayscale input image (uint8)
    - hpf_radius: radius for high-pass filter
    
    Returns:
    - sharpened image (uint8)
    """
    # Convert to float and normalize
    img_float = image.astype(np.float32) / 255.0
    
    # Apply FFT
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)
    
    # Create High-Pass Filter mask
    rows, cols = img_float.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create circular mask (low-pass)
    mask = np.ones((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    mask[distance <= hpf_radius] = 0  # Block low frequencies
    
    # Apply high-pass filter
    fft_filtered = fft_shifted * mask
    
    # Inverse FFT
    fft_ishifted = np.fft.ifftshift(fft_filtered)
    img_filtered = np.fft.ifft2(fft_ishifted)
    img_filtered = np.real(img_filtered)
    
    # Unsharp masking: original + high-pass
    img_sharpened = img_float + img_filtered
    
    # Normalize back to 0-255
    img_sharpened = np.clip(img_sharpened * 255, 0, 255).astype(np.uint8)
    
    return img_sharpened


def segment_tumor(image, threshold_value=200):
    """
    Segment potential tumor regions using simple thresholding
    
    Parameters:
    - image: grayscale image (uint8)
    - threshold_value: threshold for segmentation
    
    Returns:
    - binary mask (uint8)
    """
    _, mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return mask


def morphology_cleanup(mask, kernel_size=3):
    """
    Apply morphological opening to remove noise
    
    Parameters:
    - mask: binary mask (uint8)
    - kernel_size: size of morphological kernel
    
    Returns:
    - cleaned mask (uint8)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cleaned


def process_and_plot(image_path, title, ax_row, hpf_radius=20, threshold_value=200):
    """
    Process a single MRI image through the complete pipeline and plot results
    
    Parameters:
    - image_path: path to the image file
    - title: title for the plot
    - ax_row: matplotlib axes array for this row
    - hpf_radius: radius for FFT high-pass filter
    - threshold_value: threshold for segmentation
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Step 1: FFT Sharpening
    sharpened = fft_sharpen(image, hpf_radius=hpf_radius)
    
    # Step 2: Segmentation
    segmented = segment_tumor(sharpened, threshold_value=threshold_value)
    
    # Step 3: Morphology cleanup
    final_mask = morphology_cleanup(segmented, kernel_size=3)
    
    # Plotting
    ax_row[0].imshow(image, cmap='gray')
    ax_row[0].set_title(f'{title}\nOriginal')
    ax_row[0].axis('off')
    
    ax_row[1].imshow(sharpened, cmap='gray')
    ax_row[1].set_title('FFT Sharpened')
    ax_row[1].axis('off')
    
    ax_row[2].imshow(segmented, cmap='gray')
    ax_row[2].set_title(f'Segmented\n(Threshold > {threshold_value})')
    ax_row[2].axis('off')
    
    ax_row[3].imshow(final_mask, cmap='gray')
    ax_row[3].set_title('Tumor Detection\n(After Morphology)')
    ax_row[3].axis('off')
    
    # Overlay tumor detection on original
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay[final_mask > 0] = [255, 0, 0]  # Red overlay for detected tumor
    
    ax_row[4].imshow(overlay)
    ax_row[4].set_title('Overlay')
    ax_row[4].axis('off')


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Define dataset paths
    DATASET_ROOT = "../dataset/Brain MRI Images/Train"
    NORMAL_FOLDER = os.path.join(DATASET_ROOT, "Normal")
    TUMOR_FOLDER = os.path.join(DATASET_ROOT, "Tumor")
    
    # Check if folders exist
    if not os.path.exists(NORMAL_FOLDER):
        print(f"Error: Normal folder not found at {NORMAL_FOLDER}")
        exit(1)
    
    if not os.path.exists(TUMOR_FOLDER):
        print(f"Error: Tumor folder not found at {TUMOR_FOLDER}")
        exit(1)
    
    # Get list of images
    normal_images = [os.path.join(NORMAL_FOLDER, f) for f in os.listdir(NORMAL_FOLDER) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tumor_images = [os.path.join(TUMOR_FOLDER, f) for f in os.listdir(TUMOR_FOLDER) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(normal_images) < 2:
        print(f"Error: Need at least 2 normal images, found {len(normal_images)}")
        exit(1)
    
    if len(tumor_images) < 2:
        print(f"Error: Need at least 2 tumor images, found {len(tumor_images)}")
        exit(1)
    
    # Randomly select 2 images from each category
    selected_normal = random.sample(normal_images, 2)
    selected_tumor = random.sample(tumor_images, 2)
    
    print("=" * 60)
    print("Brain MRI Tumor Detection Pipeline")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - THRESHOLD_VALUE: {THRESHOLD_VALUE}")
    print(f"  - HPF_RADIUS: {HPF_RADIUS}")
    print(f"\nDataset:")
    print(f"  - Total Normal images: {len(normal_images)}")
    print(f"  - Total Tumor images: {len(tumor_images)}")
    print(f"\nSelected images:")
    print("Normal:")
    for i, img in enumerate(selected_normal, 1):
        print(f"  {i}. {os.path.basename(img)}")
    print("Tumor:")
    for i, img in enumerate(selected_tumor, 1):
        print(f"  {i}. {os.path.basename(img)}")
    print("=" * 60)
    
    # Create figure for visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Brain MRI Tumor Detection: Normal vs. Tumor Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Process Normal images (rows 0 and 1)
    for i, img_path in enumerate(selected_normal):
        filename = os.path.basename(img_path)
        process_and_plot(img_path, f"NORMAL #{i+1}\n{filename}", 
                        axes[i], HPF_RADIUS, THRESHOLD_VALUE)
    
    # Process Tumor images (rows 2 and 3)
    for i, img_path in enumerate(selected_tumor):
        filename = os.path.basename(img_path)
        process_and_plot(img_path, f"TUMOR #{i+1}\n{filename}", 
                        axes[i+2], HPF_RADIUS, THRESHOLD_VALUE)
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization complete!")
    print("\nExpected Results:")
    print("  - NORMAL images: Tumor Detection mask should be mostly BLACK (empty)")
    print("  - TUMOR images: Tumor Detection mask should show WHITE tumor regions")

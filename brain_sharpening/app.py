"""
Brain MRI Tumor Detection using Classical Image Processing
Pipeline: FFT Sharpening -> Skull Stripping -> Hybrid Binarization -> Watershed -> Geometric Filtering
Streamlit Web Application

Metode Terbukti Bagus:
- Skull Stripping (hapus tengkorak)
- Hybrid Binarization: Top-hat (texture) + Brightness threshold
- Watershed Segmentation
- Geometric Properties Filtering: Solidity > 0.6 (bentuk padat = tumor, bukan lipatan otak)
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import pandas as pd
from scipy import ndimage

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.4rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #dee2e6;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #dee2e6;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNCTIONS ====================

def fft_sharpen(image, hpf_radius=20):
    """
    Apply FFT-based sharpening using High-Pass Filter (Unsharp Masking)
    
    Parameters:
    - image: grayscale input image (uint8)
    - hpf_radius: radius for high-pass filter
    
    Returns:
    - sharpened image (uint8)
    - fft_magnitude_original: FFT spectrum sebelum filter
    - fft_magnitude_filtered: FFT spectrum setelah HPF
    - hpf_mask: High-Pass Filter mask
    """
    # Convert to float and normalize
    img_float = image.astype(np.float32) / 255.0
    
    # Apply FFT
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)
    
    # Calculate magnitude spectrum (for visualization)
    magnitude_spectrum_original = np.abs(fft_shifted)
    magnitude_spectrum_original_log = np.log1p(magnitude_spectrum_original)  # Log scale for better visualization
    
    # Normalize to 0-255 for display
    fft_magnitude_original = cv2.normalize(magnitude_spectrum_original_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create High-Pass Filter mask
    rows, cols = img_float.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create circular mask (low-pass)
    mask = np.ones((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    mask[distance <= hpf_radius] = 0  # Block low frequencies
    
    # Visualize HPF mask
    hpf_mask_display = (mask * 255).astype(np.uint8)
    
    # Apply high-pass filter
    fft_filtered = fft_shifted * mask
    
    # Calculate magnitude spectrum after filtering
    magnitude_spectrum_filtered = np.abs(fft_filtered)
    magnitude_spectrum_filtered_log = np.log1p(magnitude_spectrum_filtered)
    fft_magnitude_filtered = cv2.normalize(magnitude_spectrum_filtered_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Inverse FFT
    fft_ishifted = np.fft.ifftshift(fft_filtered)
    img_filtered = np.fft.ifft2(fft_ishifted)
    img_filtered = np.real(img_filtered)
    
    # Unsharp masking: original + high-pass
    img_sharpened = img_float + img_filtered
    
    # Normalize back to 0-255
    img_sharpened = np.clip(img_sharpened * 255, 0, 255).astype(np.uint8)
    
    return img_sharpened, fft_magnitude_original, fft_magnitude_filtered, hpf_mask_display


def enhanced_segmentation(image, threshold_value=200):
    """
    Enhanced segmentation with better contrast and preprocessing
    
    Parameters:
    - image: grayscale image (uint8)
    - threshold_value: threshold for segmentation
    
    Returns:
    - binary mask (uint8)
    """
    # Step 1: Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Step 2: Denoise sedikit untuk mengurangi noise
    denoised = cv2.medianBlur(enhanced, 3)
    
    # Step 3: Thresholding
    _, mask = cv2.threshold(denoised, threshold_value, 255, cv2.THRESH_BINARY)
    
    return mask, enhanced, denoised


def morphology_cleanup(mask, open_kernel=3, close_kernel=5):
    """
    Apply morphological operations to clean up the mask
    
    Parameters:
    - mask: binary mask (uint8)
    - open_kernel: size for opening (remove small noise)
    - close_kernel: size for closing (fill small holes)
    
    Returns:
    - cleaned mask (uint8)
    """
    # Opening: remove small noise/artifacts
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Closing: fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    return cleaned


def tophat_filtering(image, kernel_size=15):
    """
    Apply Top-hat filtering to enhance bright structures (tumors) on dark background
    
    Top-hat transform = Original - Opening
    This highlights small bright regions that are smaller than the structuring element
    
    Parameters:
    - image: grayscale image (uint8)
    - kernel_size: size of structuring element (odd number)
    
    Returns:
    - top-hat filtered image (uint8)
    - opened image (for visualization)
    """
    # Create structuring element (disk/ellipse shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply morphological opening
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Top-hat transform: Original - Opening
    # This extracts bright objects smaller than the structuring element
    tophat = cv2.subtract(image, opened)
    
    # Normalize and enhance the result for better visualization
    # Top-hat results are often very dim, so we need to stretch the contrast
    if tophat.max() > 0:
        # Normalize to full range 0-255
        tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    
    return tophat, opened


def anisotropic_diffusion(image, iterations=15, kappa=50, gamma=0.1, option=1):
    """
    Anisotropic Diffusion Filter (Perona-Malik)
    Edge-preserving smoothing - sesuai artikel penelitian
    
    Menghilangkan noise TANPA blur edges tumor - lebih baik dari median/gaussian blur
    
    Parameters:
    - iterations: jumlah iterasi diffusion (default: 15)
    - kappa: gradient threshold untuk detect edges (default: 50)
    - gamma: diffusion speed (default: 0.1, must be <= 0.25 for stability)
    - option: 1 untuk favor high-contrast edges, 2 untuk wide regions
    
    Reference: Perona & Malik (1990) - "Scale-Space and Edge Detection Using Anisotropic Diffusion"
    """
    img = image.astype(np.float32)
    
    for _ in range(iterations):
        # Calculate gradients in 4 directions (North, South, East, West)
        nabla_N = np.roll(img, 1, axis=0) - img  # North
        nabla_S = np.roll(img, -1, axis=0) - img  # South
        nabla_E = np.roll(img, -1, axis=1) - img  # East
        nabla_W = np.roll(img, 1, axis=1) - img  # West
        
        # Conduction coefficients - PRESERVE edges (high gradient = low diffusion)
        if option == 1:
            # Exponential function: c(x) = exp(-(||∇I||/K)^2)
            # Favors high-contrast edges
            c_N = np.exp(-(nabla_N/kappa)**2)
            c_S = np.exp(-(nabla_S/kappa)**2)
            c_E = np.exp(-(nabla_E/kappa)**2)
            c_W = np.exp(-(nabla_W/kappa)**2)
        else:
            # Rational function: c(x) = 1 / (1 + (||∇I||/K)^2)
            # Favors wide regions over smaller ones
            c_N = 1.0 / (1.0 + (nabla_N/kappa)**2)
            c_S = 1.0 / (1.0 + (nabla_S/kappa)**2)
            c_E = 1.0 / (1.0 + (nabla_E/kappa)**2)
            c_W = 1.0 / (1.0 + (nabla_W/kappa)**2)
        
        # Update image - diffusion equation
        img += gamma * (c_N * nabla_N + c_S * nabla_S + c_E * nabla_E + c_W * nabla_W)
    
    return np.clip(img, 0, 255).astype(np.uint8)


def skull_stripping(image):
    """
    Skull Stripping / Brain Extraction (Metode Ketat)
    Menghapus tengkorak dan menyisakan hanya brain tissue
    
    Returns: brain-only image (skull removed), brain_mask
    """
    # Step 1: Median blur untuk noise reduction
    smooth = cv2.medianBlur(image, 5)
    
    # Step 2: Otsu threshold untuk separate brain dari background hitam
    _, thresh = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3: Find largest contour (brain)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    brain_mask = np.zeros_like(image)
    if contours:
        # Largest contour = brain
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(brain_mask, [largest_contour], -1, 255, -1)
    
    # Step 4: Erosi pinggiran agar tulang tengkorak tidak ikut
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    brain_mask = cv2.erode(brain_mask, kernel_erode, iterations=2)
    
    # Step 5: Apply mask ke smoothed image
    skull_stripped = cv2.bitwise_and(smooth, smooth, mask=brain_mask)
    
    return skull_stripped, brain_mask


def calculate_geometric_properties(contour):
    """
    Menghitung properti geometri untuk membedakan Tumor vs Noise (Lipatan Otak)
    
    Returns:
    - solidity: Seberapa padat bentuknya? (Tumor > 0.8, Lipatan otak < 0.6)
    - circularity: Seberapa bulat? (Tumor mendekati 1, Garis lurus mendekati 0)
    """
    area = cv2.contourArea(contour)
    if area == 0:
        return 0, 0
    
    # Solidity: Area / ConvexHull Area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Circularity: 4π × Area / Perimeter²
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    return solidity, circularity


def watershed_segmentation(image, threshold_value=200, min_area=300, tophat_kernel=50, sensitivity=0.4):
    """
    Refined Watershed Segmentation dengan Geometric Properties Filtering
    
    Pipeline (Based on proven best method):
    0. Skull Stripping - hapus tengkorak
    1. Hybrid Binarization - Top-hat (texture) + Brightness threshold
    2. Watershed segmentation  
    3. Intelligent Region Filtering - gunakan Solidity & Circularity untuk filter tumor vs noise
    4. Post-processing morphology
    
    Parameters:
    - threshold_value: brightness threshold (default: 180)
    - min_area: minimum area in pixels (default: 300)
    - tophat_kernel: kernel size for top-hat (default: 50, besar agar tumor besar tidak bolong)
    - sensitivity: watershed threshold 0.1-0.9 (default: 0.4)
    
    Returns: markers, colored_output, regions_info, tophat_filtered, enhanced, binary_combined, brain_extracted
    """
    # Step 0: SKULL STRIPPING (Metode Ketat)
    brain_extracted, brain_mask = skull_stripping(image)
    
    # Step 1: HYBRID BINARIZATION
    
    # A. Jalur Top-Hat (Untuk Tekstur Tumor)
    # Kernel besar (50) agar tumor besar tidak bolong tengahnya
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_kernel, tophat_kernel))
    tophat = cv2.morphologyEx(brain_extracted, cv2.MORPH_TOPHAT, kernel_tophat)
    tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    
    # CLAHE untuk enhance
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(tophat_norm)
    
    # Otsu threshold
    _, binary_tophat = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # B. Jalur Brightness (Untuk Tumor Sangat Terang)
    # Threshold 180 agar tumor yang agak gelap/pudar tetap kena
    _, binary_bright = cv2.threshold(brain_extracted, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Gabungkan (OR)
    binary_combined = cv2.bitwise_or(binary_tophat, binary_bright)
    
    # Apply brain mask
    binary_combined = cv2.bitwise_and(binary_combined, binary_combined, mask=brain_mask)
    
    # Bersihkan noise awal
    kernel_clean = np.ones((3,3), np.uint8)
    binary_clean = cv2.morphologyEx(binary_combined, cv2.MORPH_OPEN, kernel_clean, iterations=2)
    
    # Step 2: WATERSHED SEGMENTATION
    sure_bg = cv2.dilate(binary_clean, kernel_clean, iterations=3)
    dist_transform = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)
    
    # Threshold distance transform (sensitivity) agar benih tumor terambil
    _, sure_fg = cv2.threshold(dist_transform, sensitivity * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers_initial = cv2.connectedComponents(sure_fg)
    markers_initial = markers_initial + 1
    markers_initial[unknown == 255] = 0
    
    # Apply watershed
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers_result = cv2.watershed(image_color, markers_initial)
    
    # Step 3: INTELLIGENT REGION FILTERING (GEOMETRIC PROPERTIES)
    # Filter berdasarkan bentuk (Solidity & Circularity), bukan cuma luas/terang
    
    final_mask = np.zeros_like(brain_extracted)
    unique_markers = np.unique(markers_result)
    valid_regions = []
    
    for marker in unique_markers:
        if marker <= 1:  # Skip background
            continue
        
        # Create temp mask untuk region ini
        temp_mask = np.zeros_like(brain_extracted)
        temp_mask[markers_result == marker] = 255
        
        # Find contour
        contours_reg, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_reg:
            continue
        
        cnt = contours_reg[0]
        
        # Hitung properti
        area = cv2.contourArea(cnt)
        mean_val = cv2.mean(brain_extracted, mask=temp_mask)[0]
        solidity, circularity = calculate_geometric_properties(cnt)
        
        # --- LOGIKA FILTERING ---
        # Syarat Tumor:
        # 1. Area > min_area (Bukan noise kecil)
        # 2. Mean > 60 (Cukup terang, bukan background hitam)
        # 3. Solidity > 0.6 (Bentuknya padat/gumpalan, bukan garis/lipatan otak yang berongga)
        is_tumor = False
        
        if area > min_area and mean_val > 60:
            if solidity > 0.6:  # Tumor biasanya sangat solid (>0.8), set 0.6 biar aman
                is_tumor = True
        
        if is_tumor:
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
            
            # Calculate centroid
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0
            
            valid_regions.append({
                'id': marker,
                'area_px': int(area),
                'centroid': (cx, cy),
                'mask': temp_mask,
                'mean_intensity': mean_val,
                'solidity': solidity,
                'circularity': circularity
            })
    
    # Step 4: POST-PROCESSING MORPHOLOGY (Sesuai Artikel)
    # Finishing untuk memuluskan tepi dan menambal lubang
    kernel_finish = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    # A. Closing: Tambal lubang di dalam tumor
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_finish, iterations=2)
    # B. Opening: Hapus sisa-sisa noise di pinggir tumor
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_finish, iterations=1)
    
    # Update markers dengan final mask
    _, markers_final = cv2.connectedComponents(final_mask)
    markers_final = markers_final + 1
    markers_final[final_mask == 0] = 1
    
    # Sort by area (largest first)
    valid_regions.sort(key=lambda x: x['area_px'], reverse=True)
    
    # Create colored output
    colored_output = create_colored_watershed(image, markers_final, valid_regions)
    
    return markers_final, colored_output, valid_regions, tophat_norm, enhanced, binary_combined, brain_extracted


def create_colored_watershed(image, markers, regions_info):
    """
    Create beautiful colored visualization of watershed regions with VIBRANT contrasting colors
    Hanya menggunakan warna, tanpa label angka
    """
    # Copy original image as base
    gray_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    output = gray_rgb.copy()
    
    # VIBRANT, HIGH CONTRAST colors (mudah dibedakan dari tumor putih/abu)
    colors = [
        (0, 255, 0),        # Lime Green
        (255, 0, 255),      # Magenta
        (0, 255, 255),      # Cyan
        (255, 165, 0),      # Orange
        (255, 0, 0),        # Red
        (0, 100, 255),      # Blue
        (255, 255, 0),      # Yellow
        (128, 0, 255),      # Purple
        (0, 255, 128),      # Spring Green
        (255, 20, 147),     # Deep Pink
    ]
    
    # Apply colors to each region dengan semi-transparent overlay
    overlay = output.copy()
    
    for idx, region in enumerate(regions_info):
        color = colors[idx % len(colors)]
        region_mask = (markers == region['id'])
        overlay[region_mask] = color
        # Simpan warna ke region info untuk tabel
        region['color'] = color
        region['color_name'] = get_color_name(color)
    
    # Blend: 50% original, 50% color untuk visibility yang bagus
    output = cv2.addWeighted(output, 0.5, overlay, 0.5, 0)
    
    # Draw boundaries dengan warna gelap agar terlihat
    boundary_mask = (markers == -1)
    output[boundary_mask] = [0, 0, 0]  # Black boundaries untuk kontras
    
    # HANYA draw small circle di centroid dengan warna region (tanpa label angka)
    for idx, region in enumerate(regions_info):
        cx, cy = region['centroid']
        color = colors[idx % len(colors)]
        
        # Draw small circle at centroid dengan warna region
        cv2.circle(output, (cx, cy), 6, (0, 0, 0), -1)  # Black fill
        cv2.circle(output, (cx, cy), 5, color, -1)  # Color center
    
    return output


def get_color_name(color):
    """Convert BGR color tuple to readable name"""
    color_map = {
        (0, 255, 0): 'Lime Green',
        (255, 0, 255): 'Magenta',
        (0, 255, 255): 'Cyan',
        (255, 165, 0): 'Orange',
        (255, 0, 0): 'Red',
        (0, 100, 255): 'Blue',
        (255, 255, 0): 'Yellow',
        (128, 0, 255): 'Purple',
        (0, 255, 128): 'Spring Green',
        (255, 20, 147): 'Deep Pink',
    }
    return color_map.get(color, 'Unknown')


def process_image(image, hpf_radius=20, threshold_value=200, open_kernel=3, close_kernel=5, 
                 apply_segmentation=True, pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                 method='threshold', tophat_kernel=15, watershed_sensitivity=0.6, min_tumor_area=500):
    """
    Process a single MRI image through the complete pipeline
    
    Parameters:
    - apply_segmentation: If False, only do sharpening (for normal images)
    - pixel_spacing_x, pixel_spacing_y: mm per pixel
    - method: 'threshold' or 'watershed'
    - tophat_kernel: kernel size for top-hat filtering (watershed only)
    - watershed_sensitivity: sensitivity for watershed distance transform (higher = less regions)
    - min_tumor_area: minimum area in pixels to be considered as tumor
    
    Returns: dict with all intermediate results
    """
    results = {}
    
    # Step 1: FFT Sharpening (ALWAYS applied)
    results['sharpened'], results['fft_original'], results['fft_filtered'], results['hpf_mask'] = fft_sharpen(image, hpf_radius=hpf_radius)
    
    if apply_segmentation:
        if method == 'watershed':
            # Watershed segmentation with Geometric Properties Filtering
            markers, colored_output, regions_info, tophat_filtered, enhanced, binary_combined, brain_extracted = watershed_segmentation(
                results['sharpened'], threshold_value=threshold_value, min_area=min_tumor_area, 
                tophat_kernel=tophat_kernel, sensitivity=watershed_sensitivity
            )
            
            results['markers'] = markers
            results['colored_watershed'] = colored_output
            results['regions_info'] = regions_info
            results['tophat_filtered'] = tophat_filtered
            results['enhanced'] = enhanced
            results['binary_combined'] = binary_combined  # Hybrid binary (tophat + brightness)
            results['brain_extracted'] = brain_extracted  # Skull-stripped result
            
            # Calculate total area from all regions
            total_area_px = sum(r['area_px'] for r in regions_info)
            results['tumor_area_px'] = total_area_px
            results['tumor_area_mm2'] = total_area_px * pixel_spacing_x * pixel_spacing_y
            results['num_regions'] = len(regions_info)
            
            # Create combined mask for overlay
            combined_mask = np.zeros_like(image)
            for region in regions_info:
                combined_mask = cv2.bitwise_or(combined_mask, region['mask'])
            results['final_mask'] = combined_mask
            results['overlay'] = colored_output
            
            # Create regions dataframe with color information
            regions_data = []
            for idx, region in enumerate(regions_info):
                area_mm2 = region['area_px'] * pixel_spacing_x * pixel_spacing_y
                regions_data.append({
                    'Region Color': region.get('color_name', 'Unknown'),
                    'Area (mm²)': f"{area_mm2:.2f}",
                    'Area (px)': region['area_px'],
                    'Centroid': f"({region['centroid'][0]}, {region['centroid'][1]})"
                })
            results['regions_df'] = pd.DataFrame(regions_data) if regions_data else None
            results['regions_colors'] = [region.get('color', (255, 255, 255)) for region in regions_info]
            
        else:
            # Simple threshold segmentation
            results['segmented'], results['enhanced'], results['denoised'] = enhanced_segmentation(
                results['sharpened'], threshold_value=threshold_value
            )
            
            # Step 3: Morphology cleanup
            results['final_mask'] = morphology_cleanup(
                results['segmented'], 
                open_kernel=open_kernel,
                close_kernel=close_kernel
            )
            
            # Step 4: Create overlay dengan warna kontras tinggi
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            overlay[results['final_mask'] > 0] = [0, 255, 0]  # Lime Green - kontras tinggi dengan tumor putih
            results['overlay'] = overlay
            
            # Calculate statistics in mm²
            tumor_pixels = np.count_nonzero(results['final_mask'])
            results['tumor_area_px'] = tumor_pixels
            results['tumor_area_mm2'] = tumor_pixels * pixel_spacing_x * pixel_spacing_y
            results['num_regions'] = 1 if tumor_pixels > 0 else 0
            results['regions_info'] = None
            results['regions_df'] = None
        
    else:
        # No segmentation for normal images
        results['segmented'] = None
        results['enhanced'] = None
        results['denoised'] = None
        results['final_mask'] = np.zeros_like(image)
        results['overlay'] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        results['tumor_area_px'] = 0
        results['tumor_area_mm2'] = 0.0
        results['num_regions'] = 0
        results['regions_info'] = None
        results['regions_df'] = None
    
    return results


def quick_tumor_detection(image, threshold_value=180):
    """
    Quick tumor detection untuk menentukan apakah gambar mengandung tumor
    Menggunakan simple brightness analysis
    
    Returns: (is_tumor, bright_ratio)
    - is_tumor: True jika terdeteksi tumor
    - bright_ratio: persentase pixel terang (> threshold)
    """
    # Apply simple threshold
    _, bright_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Hitung persentase pixel terang
    total_pixels = image.size
    bright_pixels = np.count_nonzero(bright_mask)
    bright_ratio = (bright_pixels / total_pixels) * 100
    
    # Kriteria tumor: Ada area terang yang signifikan (> 1%)
    # Normal MRI biasanya < 0.5% pixel sangat terang
    is_tumor = bright_ratio > 1.0
    
    return is_tumor, bright_ratio


def load_random_images(normal_folder, tumor_folder, num_each=2):
    """Load random images from normal and tumor folders"""
    # Get image lists
    normal_images = [os.path.join(normal_folder, f) for f in os.listdir(normal_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tumor_images = [os.path.join(tumor_folder, f) for f in os.listdir(tumor_folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(normal_images) < num_each or len(tumor_images) < num_each:
        return None, None, None, None
    
    # Random selection
    selected_normal = random.sample(normal_images, num_each)
    selected_tumor = random.sample(tumor_images, num_each)
    
    return selected_normal, selected_tumor, len(normal_images), len(tumor_images)


# ==================== STREAMLIT APP ====================

st.markdown('<div class="main-header">🧠 Brain MRI Tumor Detection</div>', unsafe_allow_html=True)

st.markdown("""
Aplikasi untuk deteksi tumor pada gambar MRI otak menggunakan **Classical Image Processing** (tanpa AI/ML).

**Pipeline:**
- **Normal Images:** FFT Sharpening only
- **Tumor - Simple Threshold:** FFT → CLAHE → Threshold → Morphology
- **Tumor - Watershed:** FFT → CLAHE → Top-hat → Manual Threshold → Aggressive Morphology → Watershed
""")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Parameters")

mode = st.sidebar.radio("Mode", ["Random Dataset", "Upload Image"])

st.sidebar.markdown("---")
st.sidebar.subheader("FFT Sharpening")
hpf_radius = st.sidebar.slider("HPF Radius", 5, 50, 20, 
                                help="Radius untuk High-Pass Filter. Lebih besar = lebih gentle")

st.sidebar.subheader("🔬 Watershed Segmentation (Tumor Only)")
st.sidebar.info("Metode: Geometric Filtering (Solidity > 0.6)")

threshold_value = st.sidebar.slider("Brightness Threshold", 100, 255, 180,
                                     help="Brightness threshold untuk hybrid binarization (180 = tumor agak gelap tetap terdeteksi)")

tophat_kernel = st.sidebar.slider("Top-hat Kernel Size", 15, 80, 50, step=5,
                                   help="Kernel BESAR (50+) agar tumor besar tidak bolong tengahnya. Hybrid: Top-hat (texture) + Brightness")

watershed_sensitivity = st.sidebar.slider("Watershed Sensitivity", 0.2, 0.8, 0.4, step=0.05,
                                           help="Distance transform threshold. 0.4 = balanced (detect tumor seeds accurately)")

min_tumor_area = st.sidebar.number_input("Min Tumor Area (px)", min_value=100, max_value=2000, value=300, step=50,
                                          help="Min area untuk dianggap tumor (default: 300px). Filter berdasarkan Solidity > 0.6 (bentuk padat).")

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Pixel Spacing (untuk mm²)")
pixel_spacing_x = st.sidebar.number_input("Pixel Spacing X (mm/pixel)", 
                                          min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                          help="Ukuran pixel dalam milimeter (biasanya 0.5-1.0 mm)")
pixel_spacing_y = st.sidebar.number_input("Pixel Spacing Y (mm/pixel)", 
                                          min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                          help="Ukuran pixel dalam milimeter (biasanya 0.5-1.0 mm)")

# ==================== MAIN CONTENT ====================

if mode == "Random Dataset":
    st.markdown('<div class="section-header">📁 Dataset Mode</div>', unsafe_allow_html=True)
    
    # Dataset paths
    DATASET_ROOT = "../dataset/Brain MRI Images/Train"
    NORMAL_FOLDER = os.path.join(DATASET_ROOT, "Normal")
    TUMOR_FOLDER = os.path.join(DATASET_ROOT, "Tumor")
    
    # Check folders
    if not os.path.exists(NORMAL_FOLDER) or not os.path.exists(TUMOR_FOLDER):
        st.error(f"❌ Dataset tidak ditemukan di: `{DATASET_ROOT}`")
        st.info("Pastikan struktur folder: `../dataset/Brain MRI Images/Train/Normal/` dan `.../Tumor/`")
    else:
        num_images = st.sidebar.number_input("Jumlah gambar per kategori", 1, 5, 2)
        
        if st.button("🎲 Load Random Images", type="primary"):
            with st.spinner("Loading images..."):
                selected_normal, selected_tumor, total_normal, total_tumor = load_random_images(
                    NORMAL_FOLDER, TUMOR_FOLDER, num_images
                )
                
                if selected_normal is None:
                    st.error("❌ Tidak cukup gambar di dataset")
                else:
                    st.success(f"✅ Dataset: {total_normal} Normal, {total_tumor} Tumor images")
                    
                    # Store in session state
                    st.session_state.selected_normal = selected_normal
                    st.session_state.selected_tumor = selected_tumor
        
        # Display results if loaded
        if 'selected_normal' in st.session_state:
            st.markdown('<div class="section-header">✅ Results - NORMAL Images (Sharpening Only)</div>', unsafe_allow_html=True)
            st.info("💡 Normal images: FFT Sharpening dengan visualisasi Frequency Spectrum (FFT + HPF)")
            
            for i, img_path in enumerate(st.session_state.selected_normal):
                st.markdown(f"### Normal Image #{i+1}: `{os.path.basename(img_path)}`")
                
                # Read and process (NO SEGMENTATION)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                results = process_image(image, hpf_radius, threshold_value, 3, 5, 
                                       apply_segmentation=False, 
                                       pixel_spacing_x=pixel_spacing_x, 
                                       pixel_spacing_y=pixel_spacing_y,
                                       tophat_kernel=tophat_kernel,
                                       watershed_sensitivity=watershed_sensitivity,
                                       min_tumor_area=min_tumor_area)
                
                # Display - FFT Spectrum visualization
                st.markdown("#### 🔬 FFT Frequency Domain Analysis")
                cols = st.columns(5)
                with cols[0]:
                    st.image(image, caption="1. Original Image", use_container_width=True, clamp=True)
                with cols[1]:
                    st.image(results['fft_original'], caption="2. FFT Spectrum\n(Before HPF)", use_container_width=True, clamp=True, channels="GRAY")
                with cols[2]:
                    st.image(results['hpf_mask'], caption="3. HPF Mask\n(Block Center)", use_container_width=True, clamp=True, channels="GRAY")
                with cols[3]:
                    st.image(results['fft_filtered'], caption="4. FFT Spectrum\n(After HPF)", use_container_width=True, clamp=True, channels="GRAY")
                with cols[4]:
                    st.image(results['sharpened'], caption="5. Sharpened Result", use_container_width=True, clamp=True)
                
                # Show Before/After comparison
                st.markdown("#### 📊 Before vs After Comparison")
                cols2 = st.columns(2)
                with cols2[0]:
                    fig1, ax1 = plt.subplots(figsize=(6, 6))
                    ax1.imshow(image, cmap='gray')
                    ax1.set_title('Before Sharpening', fontsize=14, fontweight='bold')
                    ax1.axis('off')
                    st.pyplot(fig1)
                with cols2[1]:
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    ax2.imshow(results['sharpened'], cmap='gray')
                    ax2.set_title('After Sharpening (FFT + HPF)', fontsize=14, fontweight='bold')
                    ax2.axis('off')
                    st.pyplot(fig2)
                    plt.close()
                
                # Explanation
                st.markdown("""
                **📖 Penjelasan FFT Sharpening:**
                1. **FFT Spectrum (Before)**: Frekuensi rendah di tengah (terang), frekuensi tinggi di pinggir
                2. **HPF Mask**: Lingkaran hitam di tengah = BLOCK frekuensi rendah (background smooth)
                3. **FFT Spectrum (After)**: Center gelap = frekuensi rendah dibuang, hanya edges/detail tersisa
                4. **Sharpened Result**: Inverse FFT → edges diperkuat, detail lebih jelas
                
                **Rumus**: `Sharpened = Original + High-Frequency Component`
                """)
                
                st.markdown("---")
            
            st.markdown('<div class="section-header">🔴 Results - TUMOR Images (Full Pipeline)</div>', unsafe_allow_html=True)
            st.info("💡 Tumor images: FFT Sharpening + Watershed Segmentation (Geometric Filtering) + Pengukuran area dalam mm²")
            
            for i, img_path in enumerate(st.session_state.selected_tumor):
                st.markdown(f"### Tumor Image #{i+1}: `{os.path.basename(img_path)}`")
                
                # Read and process (WITH SEGMENTATION)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                results = process_image(image, hpf_radius, threshold_value, 3, 5, 
                                       apply_segmentation=True,
                                       pixel_spacing_x=pixel_spacing_x, 
                                       pixel_spacing_y=pixel_spacing_y,
                                       method='watershed',
                                       tophat_kernel=tophat_kernel,
                                       watershed_sensitivity=watershed_sensitivity,
                                       min_tumor_area=min_tumor_area)
                
                # Display full pipeline (Watershed)
                cols = st.columns(6)
                with cols[0]:
                    st.image(image, caption="Original", use_container_width=True, clamp=True)
                with cols[1]:
                    st.image(results['sharpened'], caption="FFT Sharpened", use_container_width=True, clamp=True)
                with cols[2]:
                    st.image(results['brain_extracted'], caption="🧠 Brain Only", use_container_width=True, clamp=True)
                with cols[3]:
                    st.image(results['tophat_filtered'], caption="🎩 Top-hat Norm", use_container_width=True, clamp=True)
                with cols[4]:
                    st.image(results['binary_combined'], caption="🔲 Hybrid Binary", use_container_width=True, clamp=True)
                with cols[5]:
                    st.image(results['colored_watershed'], caption="🌈 Watershed + Filter", use_container_width=True, clamp=True)
                
                # Metrics with mm²
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Total Tumor Area", f"{results['tumor_area_mm2']:.2f} mm²")
                with metric_cols[1]:
                    st.metric("Number of Regions", f"{results['num_regions']}")
                with metric_cols[2]:
                    img_size_mm2 = image.size * pixel_spacing_x * pixel_spacing_y
                    percentage = (results['tumor_area_mm2'] / img_size_mm2) * 100 if img_size_mm2 > 0 else 0
                    st.metric("% of Image", f"{percentage:.2f}%")
                with metric_cols[3]:
                    if results['tumor_area_mm2'] > 0:
                        st.error("🔴 Tumor Detected")
                    else:
                        st.info("ℹ️ No tumor detected")
                
                # Show regions table with color styling
                if results.get('regions_df') is not None and results.get('regions_colors') is not None:
                    st.markdown("#### 📊 Individual Tumor Regions (by Color)")
                    
                    # Style dataframe dengan warna
                    def color_rows(row):
                        colors_list = results.get('regions_colors', [])
                        idx = row.name
                        if idx < len(colors_list):
                            bgr_color = colors_list[idx]
                            # Convert BGR to RGB hex
                            rgb_hex = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"
                            return [f'background-color: {rgb_hex}; color: black; font-weight: bold'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = results['regions_df'].style.apply(color_rows, axis=1)
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")

else:  # Upload Image mode
    st.markdown('<div class="section-header">📤 Upload Image Mode</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload MRI Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            st.error("❌ Failed to read image")
        else:
            # AUTO-DETECTION: Quick check apakah ada tumor
            with st.spinner("🔍 Analyzing image..."):
                is_tumor, bright_ratio = quick_tumor_detection(image, threshold_value)
                
                # Info untuk user
                if is_tumor:
                    st.info(f"🔴 **Tumor Detected** (Bright pixels: {bright_ratio:.2f}%) → Running full segmentation...")
                    apply_seg = True
                else:
                    st.info(f"✅ **Normal Image** (Bright pixels: {bright_ratio:.2f}%) → Sharpening only...")
                    apply_seg = False
            
            # Process
            with st.spinner("Processing..."):
                results = process_image(image, hpf_radius, threshold_value, 3, 5,
                                       apply_segmentation=apply_seg,
                                       pixel_spacing_x=pixel_spacing_x,
                                       pixel_spacing_y=pixel_spacing_y,
                                       method='watershed',
                                       tophat_kernel=tophat_kernel,
                                       watershed_sensitivity=watershed_sensitivity,
                                       min_tumor_area=min_tumor_area)
            
            st.success("✅ Processing complete!")
            
            if apply_seg:
                # Full pipeline for tumor (Watershed)
                st.markdown("### 🔴 Tumor Detection Pipeline")
                cols = st.columns(6)
                with cols[0]:
                    st.image(image, caption="1. Original", use_container_width=True, clamp=True)
                with cols[1]:
                    st.image(results['sharpened'], caption="2. FFT Sharpened", use_container_width=True, clamp=True)
                with cols[2]:
                    st.image(results['brain_extracted'], caption="3. 🧠 Brain Only", use_container_width=True, clamp=True)
                with cols[3]:
                    st.image(results['tophat_filtered'], caption="4. 🎩 Top-hat", use_container_width=True, clamp=True)
                with cols[4]:
                    st.image(results['binary_combined'], caption="5. 🔲 Hybrid Binary", use_container_width=True, clamp=True)
                with cols[5]:
                    st.image(results['colored_watershed'], caption="6. 🌈 Filtered", use_container_width=True, clamp=True)
                
                st.markdown("### Final Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True, clamp=True)
                with col2:
                    st.image(results['overlay'], caption="Tumor Detection", use_container_width=True, clamp=True)
                
                # Metrics with mm²
                st.markdown("### Analysis Results")
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Image Size", f"{image.shape[1]} x {image.shape[0]}")
                with metric_cols[1]:
                    st.metric("Total Tumor Area", f"{results['tumor_area_mm2']:.2f} mm²")
                with metric_cols[2]:
                    st.metric("Number of Regions", f"{results['num_regions']}")
                with metric_cols[3]:
                    if results['tumor_area_mm2'] > 0:
                        st.error("🔴 Tumor Detected")
                    else:
                        st.success("✅ No tumor detected")
                
                # Show regions table with color styling
                if results.get('regions_df') is not None and results.get('regions_colors') is not None:
                    st.markdown("#### 📊 Individual Tumor Regions (by Color)")
                    
                    # Style dataframe dengan warna
                    def color_rows(row):
                        colors_list = results.get('regions_colors', [])
                        idx = row.name
                        if idx < len(colors_list):
                            bgr_color = colors_list[idx]
                            # Convert BGR to RGB hex
                            rgb_hex = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"
                            return [f'background-color: {rgb_hex}; color: black; font-weight: bold'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = results['regions_df'].style.apply(color_rows, axis=1)
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                # Sharpening only for normal
                st.markdown("### ✅ Normal Image - Sharpening Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True, clamp=True)
                with col2:
                    st.image(results['sharpened'], caption="FFT Sharpened", use_container_width=True, clamp=True)
                
                st.markdown("### Analysis Results")
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Image Size", f"{image.shape[1]} x {image.shape[0]}")
                with metric_cols[1]:
                    st.metric("Tumor Area", "0 mm²")
                with metric_cols[2]:
                    st.success("✅ Normal (No tumor detected)")

# ==================== INFO & HELP ====================
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Help & Info"):
    st.markdown("""
    **Segmentation Methods:**
    
    🔹 **Simple Threshold**
    - Fast, single region detection
    - Good for simple cases
    
    🌊 **Watershed (Recommended!)**
    - Multi-region detection
    - Color-coded tumors
    - Individual area per tumor
    - More accurate
    
    **Parameter Guide:**
    - **HPF Radius**: 15-25 untuk MRI normal
    - **Threshold**: 180-220 untuk tumor
    - **Opening**: 3-5 untuk noise removal
    - **Closing**: 5-9 untuk hole filling
    - **Pixel Spacing**: 0.5-1.0 mm/pixel untuk MRI
    
    **Processing:**
    - **Normal**: Hanya sharpening
    - **Tumor**: Full pipeline + area dalam mm²
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Brain MRI Tumor Detection v1.0")

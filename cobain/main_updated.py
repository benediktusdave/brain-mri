import streamlit as st
import cv2
import numpy as np
from scipy import fftpack, ndimage
from skimage.segmentation import active_contour, morphological_chan_vese
from skimage import measure
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="MRI FFT Sharpening & Segmentation",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
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
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
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
        margin: 0.5rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .alert-danger {
        background-color: #ffe6e6;
        padding: 1.5rem;
        border-radius: 10px;
        border: 3px solid #dc3545;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #e6ffe6;
        padding: 1.5rem;
        border-radius: 10px;
        border: 3px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 MRI FFT Sharpening & Abnormality Detection</div>', unsafe_allow_html=True)

# ================== FUNGSI FFT SHARPENING ==================

def create_high_pass_filter(shape, cutoff, filter_type='gaussian', order=2):
    """
    Membuat berbagai jenis high-pass filter
    
    Parameters:
    - shape: ukuran filter (rows, cols)
    - cutoff: frekuensi cutoff
    - filter_type: 'gaussian', 'butterworth', 'ideal', 'laplacian'
    - order: order untuk butterworth filter
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Buat grid jarak dari center
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    if filter_type == 'gaussian':
        # Gaussian High-Pass Filter
        filter_mask = 1 - np.exp(-(distance**2) / (2 * (cutoff**2)))
        
    elif filter_type == 'butterworth':
        # Butterworth High-Pass Filter
        filter_mask = 1 / (1 + (cutoff / (distance + 1e-8))**(2 * order))
        
    elif filter_type == 'ideal':
        # Ideal High-Pass Filter
        filter_mask = np.where(distance <= cutoff, 0, 1)
        
    elif filter_type == 'laplacian':
        # Laplacian Filter (frequency domain)
        filter_mask = -4 * np.pi**2 * (distance**2)
        filter_mask = filter_mask / np.max(np.abs(filter_mask))  # Normalize
        
    else:
        filter_mask = np.ones(shape)
    
    return filter_mask

def fft_sharpen(image, cutoff=30, filter_type='gaussian', boost=1.5, order=2):
    """
    Sharpening menggunakan FFT dengan berbagai metode filter
    
    Returns: sharpened_image, magnitude_spectrum, filter_visualization
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Preprocessing: CLAHE untuk meningkatkan kontras lokal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_enhanced = clahe.apply(image)
    
    # Denoising ringan
    image_clean = cv2.fastNlMeansDenoising(image_enhanced, h=10)
    
    # Normalisasi ke 0-1
    image_float = image_clean.astype(np.float32) / 255.0
    
    # FFT
    fft = fftpack.fft2(image_float)
    fft_shifted = fftpack.fftshift(fft)
    
    # Magnitude spectrum untuk visualisasi
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
    
    # Buat high-pass filter
    hp_filter = create_high_pass_filter(
        image_float.shape, 
        cutoff, 
        filter_type, 
        order
    )
    
    # Apply filter dengan boost factor
    if filter_type == 'laplacian':
        # Untuk Laplacian, gunakan metode yang berbeda
        fft_filtered = fft_shifted * (1 + boost * hp_filter)
    else:
        fft_filtered = fft_shifted * (1 + boost * hp_filter)
    
    # Inverse FFT
    fft_ishifted = fftpack.ifftshift(fft_filtered)
    image_sharpened = np.real(fftpack.ifft2(fft_ishifted))
    
    # Clip dan convert kembali
    image_sharpened = np.clip(image_sharpened * 255, 0, 255).astype(np.uint8)
    
    # Magnitude spectrum setelah filtering
    magnitude_filtered = np.log(np.abs(fft_filtered) + 1)
    
    return image_sharpened, magnitude_spectrum, magnitude_filtered, hp_filter

def calculate_sharpness(image):
    """Menghitung sharpness menggunakan Laplacian variance"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

# ================== FUNGSI SEGMENTASI ==================

def preprocess_for_segmentation(image):
    """Preprocessing khusus untuk segmentasi"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Median blur
    cleaned = cv2.medianBlur(denoised, 3)
    
    return cleaned

def segment_brain_region(image):
    """Segmentasi region otak dari background"""
    processed = preprocess_for_segmentation(image)
    
    # Otsu thresholding
    _, brain_mask = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = np.ones((5,5), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find largest contour (brain)
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        brain_contour = max(contours, key=cv2.contourArea)
        brain_only_mask = np.zeros_like(processed)
        cv2.drawContours(brain_only_mask, [brain_contour], -1, 255, -1)
        return brain_only_mask, brain_contour
    
    return brain_mask, None

def detect_tumor_active_contour(image, brain_mask, alpha=0.015, beta=10, gamma=0.001, iterations=100):
    """
    Deteksi tumor menggunakan Active Contour (Snake)
    
    Parameters:
    - alpha: Snake length shape parameter (continuity)
    - beta: Snake smoothness shape parameter (smoothness)
    - gamma: Explicit time stepping parameter
    - iterations: Number of iterations
    
    Returns: abnormal_mask, contours_list, features_list
    """
    processed = preprocess_for_segmentation(image)
    
    # Apply brain mask
    brain_only = cv2.bitwise_and(processed, processed, mask=brain_mask)
    
    # Enhance edges menggunakan gradient magnitude
    sobelx = cv2.Sobel(brain_only, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(brain_only, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Deteksi kandidat tumor menggunakan adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        brain_only, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )
    adaptive = cv2.bitwise_and(adaptive, brain_mask)
    
    # Morphological operations untuk mendapatkan blob kandidat
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Remove border artifacts
    kernel_erode = np.ones((5,5), np.uint8)
    brain_mask_eroded = cv2.erode(brain_mask, kernel_erode, iterations=8)
    cleaned = cv2.bitwise_and(cleaned, brain_mask_eroded)
    
    # Find initial contours
    initial_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    features_list = []
    final_mask = np.zeros_like(brain_only)
    
    # Normalize image untuk active contour (0-1)
    image_normalized = brain_only.astype(np.float32) / 255.0
    
    # Process each candidate blob dengan Active Contour
    for init_contour in initial_contours:
        area = cv2.contourArea(init_contour)
        
        # Filter: area minimum dan maximum
        if area < 150 or area > cv2.countNonZero(brain_mask) * 0.4:
            continue
        
        # Create initial snake dari contour
        # Resample contour untuk mendapatkan jumlah point yang konsisten
        if len(init_contour) < 10:
            continue
        
        # Convert contour ke format yang sesuai untuk active_contour
        contour_points = init_contour.squeeze()
        if len(contour_points.shape) == 1:
            continue
        
        # Resample untuk mendapatkan ~100 points
        num_points = min(100, len(contour_points))
        indices = np.linspace(0, len(contour_points)-1, num_points, dtype=int)
        snake = contour_points[indices]
        
        # Flip koordinat (active_contour uses row, col)
        snake = np.fliplr(snake)
        
        try:
            # Apply Active Contour
            snake_refined = active_contour(
                gradient_magnitude,
                snake,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                max_iterations=iterations,
                coordinates='rc'
            )
            
            # Convert back ke format OpenCV (x, y)
            snake_refined = np.fliplr(snake_refined).astype(np.int32)
            
            # Create refined contour
            refined_contour = snake_refined.reshape((-1, 1, 2))
            
            # Calculate features dari refined contour
            refined_area = cv2.contourArea(refined_contour)
            
            if refined_area < 100:
                continue
            
            perimeter = cv2.arcLength(refined_contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * refined_area / (perimeter * perimeter)
            
            # Tumor detection criteria
            if circularity < 0.2 or circularity > 0.98:
                continue
            
            x, y, w, h = cv2.boundingRect(refined_contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
            
            if aspect_ratio > 4.0:
                continue
            
            hull = cv2.convexHull(refined_contour)
            hull_area = cv2.contourArea(hull)
            solidity = refined_area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.5:
                continue
            
            # Intensity analysis
            mask_temp = np.zeros_like(brain_only)
            cv2.drawContours(mask_temp, [refined_contour], -1, 255, -1)
            
            roi_pixels = processed[mask_temp == 255]
            if len(roi_pixels) == 0:
                continue
            
            mean_val = np.mean(roi_pixels)
            std_val = np.std(roi_pixels)
            
            # Calculate contrast dengan surrounding
            kernel_dilate = np.ones((20,20), np.uint8)
            surrounding_mask = cv2.dilate(mask_temp, kernel_dilate, iterations=1)
            surrounding_mask = cv2.bitwise_and(surrounding_mask, brain_mask)
            surrounding_mask = cv2.bitwise_xor(surrounding_mask, mask_temp)
            
            surrounding_pixels = processed[surrounding_mask == 255]
            if len(surrounding_pixels) > 0:
                surrounding_mean = np.mean(surrounding_pixels)
                contrast = abs(mean_val - surrounding_mean)
                
                # Tumor harus berbeda dari sekitarnya
                if contrast < 10:
                    continue
            else:
                continue
            
            # Classify abnormality
            brain_pixels = processed[brain_mask == 255]
            mean_intensity = np.mean(brain_pixels)
            std_intensity = np.std(brain_pixels)
            
            intensity_diff = mean_val - mean_intensity
            
            if intensity_diff > 1.5 * std_intensity:
                abnormality_type = 'hemorrhage'
                severity = 'high' if refined_area > 500 else 'medium'
            elif intensity_diff < -1.5 * std_intensity:
                abnormality_type = 'stroke'
                severity = 'high' if refined_area > 600 else 'medium'
            elif circularity > 0.4 and refined_area > 200:
                abnormality_type = 'tumor'
                severity = 'high' if refined_area > 800 else 'medium'
            else:
                abnormality_type = 'anomaly'
                severity = 'medium' if refined_area > 300 else 'low'
            
            # Store features
            features = {
                'contour': refined_contour,
                'area': refined_area,
                'circularity': circularity,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'mean_intensity': mean_val,
                'std_intensity': std_val,
                'contrast': contrast,
                'type': abnormality_type,
                'severity': severity,
                'bounding_box': (x, y, w, h)
            }
            
            valid_contours.append(refined_contour)
            features_list.append(features)
            cv2.drawContours(final_mask, [refined_contour], -1, 255, -1)
            
        except Exception as e:
            # Skip jika active contour gagal
            continue
    
    return final_mask, valid_contours, features_list

def detect_tumor_chan_vese(image, brain_mask, iterations=100, smoothing=3):
    """
    Deteksi tumor menggunakan Chan-Vese (Morphological Active Contour)
    Metode ini bagus untuk deteksi region dengan boundary tidak jelas
    
    Returns: abnormal_mask, contours_list, features_list
    """
    processed = preprocess_for_segmentation(image)
    brain_only = cv2.bitwise_and(processed, processed, mask=brain_mask)
    
    # Normalize ke 0-1
    image_normalized = brain_only.astype(np.float32) / 255.0
    
    # Create initial level set (inisialisasi di tengah)
    rows, cols = image_normalized.shape
    center_y, center_x = rows // 2, cols // 2
    radius = min(rows, cols) // 4
    
    y, x = np.ogrid[:rows, :cols]
    init_ls = np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius
    
    # Apply Chan-Vese
    try:
        cv_result = morphological_chan_vese(
            image_normalized,
            num_iter=iterations,
            init_level_set=init_ls,
            smoothing=smoothing
        )
        
        # Convert ke binary mask
        result_mask = (cv_result > 0).astype(np.uint8) * 255
        
        # Apply brain mask
        result_mask = cv2.bitwise_and(result_mask, brain_mask)
        
        # Clean up
        kernel = np.ones((5,5), np.uint8)
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Remove border artifacts
        kernel_erode = np.ones((5,5), np.uint8)
        brain_mask_eroded = cv2.erode(brain_mask, kernel_erode, iterations=8)
        result_mask = cv2.bitwise_and(result_mask, brain_mask_eroded)
        
        # Find contours
        contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        features_list = []
        final_mask = np.zeros_like(brain_only)
        
        total_brain_area = cv2.countNonZero(brain_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 150 or area > total_brain_area * 0.4:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < 0.2 or circularity > 0.98:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
            
            if aspect_ratio > 4.0:
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.5:
                continue
            
            # Intensity analysis
            mask_temp = np.zeros_like(brain_only)
            cv2.drawContours(mask_temp, [contour], -1, 255, -1)
            
            roi_pixels = processed[mask_temp == 255]
            if len(roi_pixels) == 0:
                continue
            
            mean_val = np.mean(roi_pixels)
            std_val = np.std(roi_pixels)
            
            # Contrast analysis
            kernel_dilate = np.ones((20,20), np.uint8)
            surrounding_mask = cv2.dilate(mask_temp, kernel_dilate, iterations=1)
            surrounding_mask = cv2.bitwise_and(surrounding_mask, brain_mask)
            surrounding_mask = cv2.bitwise_xor(surrounding_mask, mask_temp)
            
            surrounding_pixels = processed[surrounding_mask == 255]
            if len(surrounding_pixels) > 0:
                surrounding_mean = np.mean(surrounding_pixels)
                contrast = abs(mean_val - surrounding_mean)
                
                if contrast < 10:
                    continue
            else:
                continue
            
            # Classify
            brain_pixels = processed[brain_mask == 255]
            mean_intensity = np.mean(brain_pixels)
            std_intensity = np.std(brain_pixels)
            
            intensity_diff = mean_val - mean_intensity
            
            if intensity_diff > 1.5 * std_intensity:
                abnormality_type = 'hemorrhage'
                severity = 'high' if area > 500 else 'medium'
            elif intensity_diff < -1.5 * std_intensity:
                abnormality_type = 'stroke'
                severity = 'high' if area > 600 else 'medium'
            elif circularity > 0.4 and area > 200:
                abnormality_type = 'tumor'
                severity = 'high' if area > 800 else 'medium'
            else:
                abnormality_type = 'anomaly'
                severity = 'medium' if area > 300 else 'low'
            
            features = {
                'contour': contour,
                'area': area,
                'circularity': circularity,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'mean_intensity': mean_val,
                'std_intensity': std_val,
                'contrast': contrast,
                'type': abnormality_type,
                'severity': severity,
                'bounding_box': (x, y, w, h)
            }
            
            valid_contours.append(contour)
            features_list.append(features)
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
        
        return final_mask, valid_contours, features_list
        
    except Exception as e:
        st.error(f"Chan-Vese error: {str(e)}")
        return np.zeros_like(image), [], []

def detect_abnormalities(image, brain_mask, sensitivity=5):
    """
    Deteksi area abnormal: tumor, pendarahan, gumpalan
    Menggunakan deteksi blob dan analisis texture variance
    
    Returns: abnormal_mask, contours_list, features_list
    """
    processed = preprocess_for_segmentation(image)
    
    # Ambil hanya region otak
    brain_only = cv2.bitwise_and(processed, processed, mask=brain_mask)
    
    # Hitung statistik intensitas di region otak
    brain_pixels = processed[brain_mask == 255]
    if len(brain_pixels) == 0:
        return np.zeros_like(image), [], []
    
    mean_intensity = np.mean(brain_pixels)
    std_intensity = np.std(brain_pixels)
    
    # === METODE 1: Deteksi area dengan intensitas SANGAT berbeda (outliers) ===
    # Hanya deteksi yang benar-benar ekstrem (bukan struktur normal)
    sensitivity_factor = sensitivity / 10.0  # Normalize 0.1 - 1.0
    
    # Threshold yang lebih ketat: 2.5-3.5 std (tergantung sensitivity)
    high_threshold = mean_intensity + (2.5 + sensitivity_factor) * std_intensity
    low_threshold = mean_intensity - (2.5 + sensitivity_factor) * std_intensity
    
    # Deteksi area sangat terang (pendarahan, kalsifikasi)
    _, very_bright = cv2.threshold(brain_only, high_threshold, 255, cv2.THRESH_BINARY)
    
    # Deteksi area sangat gelap (edema, nekrosis) - tapi di dalam brain region
    _, very_dark = cv2.threshold(brain_only, low_threshold, 255, cv2.THRESH_BINARY_INV)
    very_dark = cv2.bitwise_and(very_dark, brain_mask)
    
    # === METODE 2: Deteksi blob berbentuk bulat/oval (tumor tipikal) ===
    # Gunakan adaptive threshold untuk deteksi lokal
    adaptive = cv2.adaptiveThreshold(
        brain_only, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -5  # Negative C untuk deteksi area lebih gelap
    )
    adaptive = cv2.bitwise_and(adaptive, brain_mask)
    
    # === METODE 3: Variance-based detection ===
    # Area tumor biasanya memiliki texture variance berbeda dari jaringan normal
    kernel_variance = 15
    mean_filtered = cv2.blur(brain_only.astype(np.float32), (kernel_variance, kernel_variance))
    squared = cv2.blur((brain_only.astype(np.float32))**2, (kernel_variance, kernel_variance))
    variance = squared - mean_filtered**2
    variance = np.sqrt(np.abs(variance)).astype(np.uint8)
    
    # Threshold variance (area dengan variance sangat tinggi atau rendah)
    _, high_variance = cv2.threshold(variance, np.percentile(variance[brain_mask > 0], 90), 255, cv2.THRESH_BINARY)
    high_variance = cv2.bitwise_and(high_variance, brain_mask)
    
    # === Kombinasi deteksi ===
    # Hanya ambil area yang terdeteksi oleh MULTIPLE metode (lebih reliable)
    combined = np.zeros_like(brain_only)
    
    # Brightness-based (sangat kuat)
    combined = cv2.bitwise_or(combined, very_bright)
    combined = cv2.bitwise_or(combined, very_dark)
    
    # Blob dengan variance tinggi (indikasi massa)
    blob_and_variance = cv2.bitwise_and(adaptive, high_variance)
    combined = cv2.bitwise_or(combined, blob_and_variance)
    
    # === Morphological cleaning yang lebih agresif ===
    # Remove noise kecil
    kernel_open = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Close small gaps dalam blob
    kernel_close = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Remove border artifacts (struktur di tepi brain)
    kernel_erode = np.ones((3,3), np.uint8)
    brain_mask_eroded = cv2.erode(brain_mask, kernel_erode, iterations=5)
    cleaned = cv2.bitwise_and(cleaned, brain_mask_eroded)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # === Analisis dan filter contours dengan kriteria KETAT ===
    valid_contours = []
    features_list = []
    final_mask = np.zeros_like(brain_only)
    
    # Hitung total brain area untuk proporsi check
    total_brain_area = cv2.countNonZero(brain_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # === FILTER 1: Area ===
        # Minimum: 100 pixels (terlalu kecil = noise)
        # Maximum: 30% brain area (terlalu besar = bukan tumor spesifik)
        min_area = 100
        max_area = total_brain_area * 0.3
        
        if area < min_area or area > max_area:
            continue
        
        # === FILTER 2: Shape analysis ===
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Tumor biasanya agak bulat: circularity 0.3-0.9
        # Terlalu bulat sempurna (>0.95) = bisa artifact
        # Terlalu irregular (<0.25) = bisa noise/struktur normal
        if circularity < 0.25 or circularity > 0.95:
            continue
        
        # === FILTER 3: Bounding box ===
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
        
        # Tumor biasanya tidak terlalu elongated
        if aspect_ratio > 3.5:
            continue
        
        # === FILTER 4: Solidity ===
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Tumor biasanya cukup solid (0.6-0.98)
        # Terlalu rendah = struktur kompleks/noise
        if solidity < 0.6:
            continue
        
        # === FILTER 5: Intensity analysis ===
        mask_temp = np.zeros_like(brain_only)
        cv2.drawContours(mask_temp, [contour], -1, 255, -1)
        
        roi_pixels = processed[mask_temp == 255]
        if len(roi_pixels) == 0:
            continue
        
        mean_val = np.mean(roi_pixels)
        std_val = np.std(roi_pixels)
        
        # Hitung contrast dengan sekitar
        # Dilate untuk ambil area sekitar
        kernel_dilate = np.ones((15,15), np.uint8)
        surrounding_mask = cv2.dilate(mask_temp, kernel_dilate, iterations=1)
        surrounding_mask = cv2.bitwise_and(surrounding_mask, brain_mask)
        surrounding_mask = cv2.bitwise_xor(surrounding_mask, mask_temp)  # Hanya ring di luar
        
        surrounding_pixels = processed[surrounding_mask == 255]
        if len(surrounding_pixels) > 0:
            surrounding_mean = np.mean(surrounding_pixels)
            contrast = abs(mean_val - surrounding_mean)
            
            # Tumor harus punya contrast signifikan dengan sekitarnya
            min_contrast = 15  # Minimum difference
            if contrast < min_contrast:
                continue
        else:
            continue  # Skip jika tidak bisa hitung surrounding
        
        # === FILTER 6: Texture homogeneity ===
        # Tumor biasanya lebih homogen daripada jaringan normal
        # Tapi tidak terlalu homogen (bisa CSF/ventricle)
        if std_val < 5 or std_val > 40:
            continue
        
        # === Klasifikasi tipe abnormalitas ===
        abnormality_type = 'unknown'
        severity = 'low'
        
        # Intensitas relatif terhadap mean brain
        intensity_diff = mean_val - mean_intensity
        
        if intensity_diff > 1.5 * std_intensity:
            # Area sangat terang
            abnormality_type = 'hemorrhage'
            severity = 'high' if area > 500 else 'medium'
        elif intensity_diff < -1.5 * std_intensity:
            # Area sangat gelap
            abnormality_type = 'stroke'
            severity = 'high' if area > 600 else 'medium'
        elif circularity > 0.5 and area > 200:
            # Massa bulat dengan ukuran signifikan
            abnormality_type = 'tumor'
            severity = 'high' if area > 800 else 'medium'
        else:
            # Area abnormal tapi belum jelas klasifikasinya
            abnormality_type = 'anomaly'
            severity = 'medium' if area > 300 else 'low'
        
        # Store features
        features = {
            'contour': contour,
            'area': area,
            'circularity': circularity,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'mean_intensity': mean_val,
            'std_intensity': std_val,
            'contrast': contrast,
            'type': abnormality_type,
            'severity': severity,
            'bounding_box': (x, y, w, h)
        }
        
        valid_contours.append(contour)
        features_list.append(features)
        cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    return final_mask, valid_contours, features_list

def create_segmentation_overlay(image, features_list, brain_contour=None):
    """Membuat visualisasi overlay dengan color coding"""
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = image.copy()
    
    # Color mapping
    color_map = {
        'tumor': (255, 0, 0),       # Merah
        'hemorrhage': (0, 0, 255),  # Biru
        'stroke': (255, 255, 0),    # Kuning
        'anomaly': (255, 165, 0)    # Orange
    }
    
    # Draw brain boundary
    if brain_contour is not None:
        cv2.drawContours(vis_image, [brain_contour], -1, (0, 255, 0), 1)
    
    # Draw abnormalities
    for i, features in enumerate(features_list):
        contour = features['contour']
        abnorm_type = features['type']
        severity = features['severity']
        area = features['area']
        
        color = color_map.get(abnorm_type, (255, 255, 255))
        thickness = 3 if severity == 'high' else 2 if severity == 'medium' else 1
        
        # Draw contour
        cv2.drawContours(vis_image, [contour], -1, color, thickness)
        
        # Add label
        x, y, w, h = features['bounding_box']
        label = f"{abnorm_type.upper()[:3]} {int(area)}"
        
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        cv2.rectangle(vis_image, (x, y-text_height-5), 
                     (x+text_width, y), (0, 0, 0), -1)
        
        # Text
        cv2.putText(vis_image, label, (x, y-3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis_image

# ================== STREAMLIT UI ==================

# Informasi aplikasi
with st.expander("ℹ️ Tentang Aplikasi", expanded=False):
    st.markdown("""
    **Aplikasi ini menggunakan FFT (Fast Fourier Transform) untuk:**
    1. **Sharpening gambar MRI** dengan 4 metode berbeda:
       - Gaussian High-Pass Filter (smooth transition, less ringing)
       - Butterworth High-Pass Filter (balance smoothness & sharpness)
       - Ideal High-Pass Filter (sharp cutoff, potential ringing)
       - Laplacian Filter (edge enhancement)
    
    2. **Deteksi otomatis abnormalitas** menggunakan image processing:
       - Tumor/massa (bentuk irregular, area besar)
       - Pendarahan (area hyper-intense)
       - Stroke/iskemik (area hypo-intense)
    
    **PENTING:** Hasil ini untuk keperluan riset dan edukasi, bukan diagnosis medis!
    """)

# ================== SIDEBAR ==================

st.sidebar.markdown('<div class="section-header">⚙️ FFT Sharpening</div>', unsafe_allow_html=True)

# Method selection
fft_method = st.sidebar.selectbox(
    "Pilih Metode Filter:",
    ['gaussian', 'butterworth', 'ideal', 'laplacian'],
    format_func=lambda x: {
        'gaussian': 'Gaussian HPF (Recommended)',
        'butterworth': 'Butterworth HPF',
        'ideal': 'Ideal HPF',
        'laplacian': 'Laplacian Filter'
    }[x],
    index=0
)

# Parameters
cutoff_freq = st.sidebar.slider(
    "Cutoff Frequency",
    5, 100, 30,
    help="Frekuensi cutoff untuk filter. Nilai rendah = sharpening lebih kuat"
)

if fft_method == 'butterworth':
    butter_order = st.sidebar.slider(
        "Butterworth Order",
        1, 6, 2,
        help="Order filter. Nilai tinggi = transisi lebih tajam"
    )
else:
    butter_order = 2

boost_factor = st.sidebar.slider(
    "Boost Factor",
    0.1, 3.0, 1.5, 0.1,
    help="Faktor penguat untuk high frequency. 0.5-2.0 recommended"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-header">🔍 Segmentasi</div>', unsafe_allow_html=True)

enable_segmentation = st.sidebar.checkbox("Aktifkan Deteksi Abnormalitas", value=True)

if enable_segmentation:
    segmentation_method = st.sidebar.selectbox(
        "Metode Segmentasi:",
        ['threshold_based', 'active_contour', 'chan_vese'],
        format_func=lambda x: {
            'threshold_based': 'Threshold-Based (Fast)',
            'active_contour': 'Active Contour/Snake (Recommended)',
            'chan_vese': 'Chan-Vese (Advanced)'
        }[x],
        index=1
    )
    
    if segmentation_method == 'active_contour':
        st.sidebar.markdown("**Active Contour Parameters:**")
        ac_alpha = st.sidebar.slider("Alpha (continuity)", 0.001, 0.1, 0.015, 0.001,
                                     help="Kontrol kontinuitas kontur")
        ac_beta = st.sidebar.slider("Beta (smoothness)", 1, 20, 10, 1,
                                    help="Kontrol kehalusan kontur")
        ac_iterations = st.sidebar.slider("Iterations", 50, 300, 100, 10,
                                         help="Jumlah iterasi evolusi")
    elif segmentation_method == 'chan_vese':
        st.sidebar.markdown("**Chan-Vese Parameters:**")
        cv_iterations = st.sidebar.slider("Iterations", 50, 300, 100, 10,
                                         help="Jumlah iterasi")
        cv_smoothing = st.sidebar.slider("Smoothing", 1, 5, 3, 1,
                                        help="Level smoothing")
    else:
        sensitivity = st.sidebar.slider(
            "Sensitivity",
            1, 10, 5,
            help="Tingkat sensitivitas deteksi. Nilai tinggi = lebih sensitif. Nilai 3-5 untuk mode normal, 6-8 untuk deteksi agresif"
        )
    
    st.sidebar.markdown("**Tips Segmentasi:**")
    if segmentation_method == 'active_contour':
        st.sidebar.info("""
        ✅ **Active Contour (Recommended)**
        • Sangat baik untuk tumor dengan boundary irregular
        • Dapat menyesuaikan dengan bentuk kompleks
        • Lebih akurat daripada threshold-based
        """)
    elif segmentation_method == 'chan_vese':
        st.sidebar.info("""
        🔬 **Chan-Vese**
        • Untuk tumor dengan boundary tidak jelas (fuzzy)
        • Region-based segmentation
        • Lebih robust terhadap noise
        """)
    else:
        st.sidebar.info("""
        ⚡ **Threshold-Based**
        • Cepat dan simple
        • Sensitivity 3-5: Deteksi konservatif
        • Sensitivity 6-8: Deteksi agresif
        """)

# ================== MAIN AREA ==================

st.markdown('<div class="section-header">📁 Upload Gambar MRI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih file gambar MRI (JPG, PNG, TIFF)",
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff']
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if original_image is not None:
        # Resize jika terlalu besar
        max_size = 800
        if max(original_image.shape) > max_size:
            scale = max_size / max(original_image.shape)
            new_size = (int(original_image.shape[1] * scale), 
                       int(original_image.shape[0] * scale))
            original_image = cv2.resize(original_image, new_size)
        
        # ============= PROCESSING =============
        
        start_time = time.time()
        
        # 1. FFT Sharpening
        sharpened, mag_original, mag_filtered, hp_filter = fft_sharpen(
            original_image, 
            cutoff=cutoff_freq,
            filter_type=fft_method,
            boost=boost_factor,
            order=butter_order
        )
        
        processing_time_sharpen = (time.time() - start_time) * 1000
        
        # 2. Segmentation
        if enable_segmentation:
            start_seg = time.time()
            
            brain_mask, brain_contour = segment_brain_region(sharpened)
            
            # Pilih metode segmentasi
            if segmentation_method == 'active_contour':
                abnormal_mask, contours, features = detect_tumor_active_contour(
                    sharpened, brain_mask, 
                    alpha=ac_alpha, beta=ac_beta, iterations=ac_iterations
                )
            elif segmentation_method == 'chan_vese':
                abnormal_mask, contours, features = detect_tumor_chan_vese(
                    sharpened, brain_mask,
                    iterations=cv_iterations, smoothing=cv_smoothing
                )
            else:  # threshold_based
                abnormal_mask, contours, features = detect_abnormalities(
                    sharpened, brain_mask, sensitivity
                )
            
            overlay_image = create_segmentation_overlay(sharpened, features, brain_contour)
            
            processing_time_seg = (time.time() - start_seg) * 1000
        else:
            abnormal_mask = np.zeros_like(sharpened)
            features = []
            overlay_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            processing_time_seg = 0
        
        total_time = processing_time_sharpen + processing_time_seg
        
        # Calculate metrics
        sharpness_original = calculate_sharpness(original_image)
        sharpness_processed = calculate_sharpness(sharpened)
        improvement = ((sharpness_processed - sharpness_original) / sharpness_original * 100 
                      if sharpness_original > 0 else 0)
        
        # ============= RESULTS DISPLAY =============
        
        st.markdown('<div class="section-header">📊 Hasil Analisis</div>', unsafe_allow_html=True)
        
        # Status alert
        if enable_segmentation:
            total_abnormalities = len(features)
            total_area = sum(f['area'] for f in features)
            
            if total_abnormalities > 0:
                high_severity = sum(1 for f in features if f['severity'] == 'high')
                
                st.markdown(f'''
                <div class="alert-danger">
                    <h3 style="color: #dc3545; margin: 0;">🚨 ABNORMALITAS TERDETEKSI!</h3>
                    <p style="margin: 0.5rem 0;">
                        <strong>Total Region:</strong> {total_abnormalities} | 
                        <strong>Total Area:</strong> {total_area:.0f} pixels | 
                        <strong>High Severity:</strong> {high_severity}
                    </p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="alert-success">
                    <h3 style="color: #28a745; margin: 0;">✅ TIDAK ADA ABNORMALITAS TERDETEKSI</h3>
                    <p style="margin: 0.5rem 0;">Gambar MRI dalam rentang normal</p>
                </div>
                ''', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f'''
            <div class="metric-box">
                <div style="font-size: 0.85rem; color: #666;">Sharpness Awal</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #666;">{sharpness_original:.1f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-box">
                <div style="font-size: 0.85rem; color: #666;">Sharpness Hasil</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #2e86ab;">{sharpness_processed:.1f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            color = "green" if 10 <= improvement <= 80 else "orange" if improvement > 0 else "red"
            st.markdown(f'''
            <div class="metric-box">
                <div style="font-size: 0.85rem; color: #666;">Peningkatan</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: {color};">{improvement:+.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            if enable_segmentation:
                anomaly_color = "#dc3545" if len(features) > 0 else "#28a745"
                st.markdown(f'''
                <div class="metric-box">
                    <div style="font-size: 0.85rem; color: #666;">Abnormalitas</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: {anomaly_color};">{len(features)}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        with col5:
            st.markdown(f'''
            <div class="metric-box">
                <div style="font-size: 0.85rem; color: #666;">Waktu Proses</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #666;">{total_time:.0f} ms</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # ============= VISUALISASI =============
        
        st.markdown('<div class="section-header">🖼️ Perbandingan Gambar</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(original_image, caption="MRI Original", use_container_width=True, clamp=True)
        
        with col2:
            st.image(sharpened, caption=f"Hasil Sharpening ({fft_method.upper()})", 
                    use_container_width=True, clamp=True)
        
        with col3:
            if enable_segmentation:
                st.image(overlay_image, caption="Hasil Segmentasi", use_container_width=True, clamp=True)
            else:
                st.image(sharpened, caption="Preview", use_container_width=True, clamp=True)
        
        # FFT Spectrum & Filter
        st.markdown('<div class="section-header">📈 Analisis Frekuensi (FFT)</div>', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(mag_original, cmap='gray')
        axes[0].set_title('Spectrum Original', fontsize=11)
        axes[0].axis('off')
        
        axes[1].imshow(hp_filter, cmap='hot')
        axes[1].set_title(f'{fft_method.upper()} Filter', fontsize=11)
        axes[1].axis('off')
        
        axes[2].imshow(mag_filtered, cmap='gray')
        axes[2].set_title('Spectrum Setelah Filter', fontsize=11)
        axes[2].axis('off')
        
        # Difference map
        diff = cv2.absdiff(original_image, sharpened)
        axes[3].imshow(diff, cmap='hot')
        axes[3].set_title('Peta Perubahan (Difference)', fontsize=11)
        axes[3].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Histogram comparison
        st.markdown('<div class="section-header">📊 Histogram Intensitas</div>', unsafe_allow_html=True)
        
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        
        hist_orig = cv2.calcHist([original_image], [0], None, [256], [0, 256])
        hist_sharp = cv2.calcHist([sharpened], [0], None, [256], [0, 256])
        
        ax_hist.plot(hist_orig, color='blue', alpha=0.7, label='Original')
        ax_hist.plot(hist_sharp, color='red', alpha=0.7, label='Sharpened')
        ax_hist.set_xlabel('Pixel Intensity')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram Comparison')
        ax_hist.legend()
        ax_hist.grid(alpha=0.3)
        
        st.pyplot(fig_hist)
        
        # ============= DETAIL SEGMENTASI =============
        
        if enable_segmentation and len(features) > 0:
            with st.expander("🔍 Detail Abnormalitas Terdeteksi", expanded=True):
                # Legend
                st.markdown("""
                **Color Legend:**
                - 🔴 **MERAH** = Tumor/Massa
                - 🔵 **BIRU** = Pendarahan (Hemorrhage)
                - 🟡 **KUNING** = Stroke/Iskemik
                - 🟠 **ORANGE** = Anomali lainnya
                """)
                
                st.markdown("---")
                
                # Group by type
                tumor_count = sum(1 for f in features if f['type'] == 'tumor')
                hemorrhage_count = sum(1 for f in features if f['type'] == 'hemorrhage')
                stroke_count = sum(1 for f in features if f['type'] == 'stroke')
                other_count = sum(1 for f in features if f['type'] == 'anomaly')
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🔴 Tumor", tumor_count)
                col2.metric("🔵 Pendarahan", hemorrhage_count)
                col3.metric("🟡 Stroke", stroke_count)
                col4.metric("🟠 Lainnya", other_count)
                
                st.markdown("---")
                
                # Detail tiap abnormalitas
                st.markdown("**📋 Daftar Detail:**")
                
                for i, f in enumerate(features):
                    severity_emoji = "🔴" if f['severity'] == 'high' else "🟡" if f['severity'] == 'medium' else "🟢"
                    
                    st.markdown(f"""
                    **{severity_emoji} Region #{i+1}: {f['type'].upper()}** (Severity: {f['severity']})
                    - Area: {f['area']:.0f} pixels
                    - Circularity: {f['circularity']:.3f} {"(irregular shape)" if f['circularity'] < 0.6 else "(moderate)"}
                    - Solidity: {f['solidity']:.3f} {"(non-solid texture)" if f['solidity'] < 0.8 else "(solid)"}
                    - Mean Intensity: {f['mean_intensity']:.1f}
                    - Contrast: {f.get('contrast', 0):.1f} (difference from surrounding)
                    - Texture Std: {f.get('std_intensity', 0):.1f}
                    - Location: x={f['bounding_box'][0]}, y={f['bounding_box'][1]}
                    """)
                
                st.warning("""
                **⚠️ DISCLAIMER MEDIS:**
                - Hasil ini adalah analisis computational untuk keperluan riset dan edukasi
                - BUKAN untuk diagnosis medis klinis
                - Konsultasi dengan radiologis profesional sangat diperlukan
                - False positive/negative mungkin terjadi
                """)
        
        # ============= DOWNLOAD =============
        
        st.markdown('<div class="section-header">💾 Download Hasil</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Sharpened image
            sharpened_pil = Image.fromarray(sharpened)
            buf_sharp = io.BytesIO()
            sharpened_pil.save(buf_sharp, format='PNG')
            buf_sharp.seek(0)
            
            st.download_button(
                label="📥 Gambar Sharpened",
                data=buf_sharp,
                file_name=f"mri_sharpened_{fft_method}.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col2:
            # Overlay segmentation
            if enable_segmentation:
                overlay_pil = Image.fromarray(overlay_image)
                buf_overlay = io.BytesIO()
                overlay_pil.save(buf_overlay, format='PNG')
                buf_overlay.seek(0)
                
                st.download_button(
                    label="📥 Gambar Segmentasi",
                    data=buf_overlay,
                    file_name="mri_segmentation.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with col3:
            # Report
            report = f"""LAPORAN ANALISIS MRI
{'='*50}

FILE INFORMASI:
- Nama File: {uploaded_file.name}
- Tanggal: {time.strftime("%Y-%m-%d %H:%M:%S")}
- Ukuran: {original_image.shape[1]}x{original_image.shape[0]} pixels

SHARPENING:
- Metode: {fft_method.upper()}
- Cutoff Frequency: {cutoff_freq}
- Boost Factor: {boost_factor}
"""
            
            if fft_method == 'butterworth':
                report += f"- Butterworth Order: {butter_order}\n"
            
            report += f"""
HASIL SHARPENING:
- Sharpness Original: {sharpness_original:.2f}
- Sharpness Processed: {sharpness_processed:.2f}
- Peningkatan: {improvement:+.2f}%
- Waktu: {processing_time_sharpen:.2f} ms

"""
            
            if enable_segmentation:
                report += f"""SEGMENTASI & DETEKSI:
- Metode: {segmentation_method.upper().replace('_', ' ')}
- Waktu Segmentasi: {processing_time_seg:.2f} ms
- Total Abnormalitas: {len(features)}
- Total Area: {sum(f['area'] for f in features):.0f} pixels
"""
            
            if segmentation_method == 'active_contour':
                report += f"""- Alpha: {ac_alpha}
- Beta: {ac_beta}
- Iterations: {ac_iterations}
"""
            elif segmentation_method == 'chan_vese':
                report += f"""- Iterations: {cv_iterations}
- Smoothing: {cv_smoothing}
"""
            else:
                report += f"- Sensitivity: {sensitivity}\n"
            
            report += "\nDETAIL ABNORMALITAS:\n"
            
            if len(features) > 0:
                for i, f in enumerate(features):
                    report += f"""
Region #{i+1}:
  Type: {f['type'].upper()}
  Severity: {f['severity']}
  Area: {f['area']:.0f} pixels
  Circularity: {f['circularity']:.3f}
  Solidity: {f['solidity']:.3f}
  Mean Intensity: {f['mean_intensity']:.1f}
  Contrast: {f.get('contrast', 0):.1f}
  Texture Std: {f.get('std_intensity', 0):.1f}
  Location: ({f['bounding_box'][0]}, {f['bounding_box'][1]})
"""
            else:
                report += "  Tidak ada abnormalitas terdeteksi\n"
            
            report += f"""
{'='*50}
DISCLAIMER: Hasil untuk keperluan riset dan edukasi.
BUKAN untuk diagnosis medis klinis.
"""
            
            st.download_button(
                label="📄 Laporan Lengkap",
                data=report,
                file_name="mri_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    else:
        st.error("❌ Gagal membaca gambar. Pastikan file valid.")

else:
    # Placeholder
    st.info("👆 **Upload gambar MRI untuk memulai analisis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✨ Fitur Sharpening:**")
        st.markdown("""
        - 4 metode FFT filter berbeda
        - Perbandingan sebelum-sesudah
        - Analisis spektrum frekuensi
        - Histogram intensitas
        - Metrics sharpness otomatis
        """)
    
    with col2:
        st.markdown("**🔍 Fitur Deteksi:**")
        st.markdown("""
        - Deteksi tumor/massa
        - Deteksi pendarahan
        - Deteksi stroke/iskemik
        - Color-coded overlay
        - Detail karakteristik per region
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "🧠 MRI FFT Sharpening & Segmentation Tool | Image Processing Based | "
    "Untuk Keperluan Riset & Edukasi Saja"
    "</div>", 
    unsafe_allow_html=True
)

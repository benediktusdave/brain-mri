import streamlit as st
import cv2
import numpy as np
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="MRI Lesion Detection",
    page_icon="🏥",
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
        font-size: 1.4rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #dee2e6;
        font-weight: bold;
    }
    .lesion-box {
        background: linear-gradient(135deg, #fff0f0, #ffcccc);
        padding: 1.5rem;
        border-radius: 10px;
        border: 3px solid #ff4444;
        margin: 1rem 0;
    }
    .normal-box {
        background: linear-gradient(135deg, #f0fff0, #ccffcc);
        padding: 1.5rem;
        border-radius: 10px;
        border: 3px solid #44cc44;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown('<div class="main-header">🧠 MRI Lesion Detection & Enhancement</div>', unsafe_allow_html=True)

# ================== FUNGSI SHARPENING OPTIMAL ==================

def optimal_sharpening(image, method='adaptive'):
    """Sharpening optimal untuk MRI"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # CLAHE untuk kontras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_contrast = clahe.apply(image)
    
    if method == 'adaptive':
        # Unsharp masking dengan kontrol
        blurred = cv2.GaussianBlur(image_contrast, (0, 0), 1.5)
        detail = cv2.subtract(image_contrast, blurred)
        sharpened = cv2.addWeighted(image_contrast, 1.0, detail, 0.4, 0)
    
    elif method == 'frequency':
        # FFT-based sharpening
        img_float = image_contrast.astype(np.float32) / 255.0
        fft = fftpack.fft2(img_float)
        fft_shifted = fftpack.fftshift(fft)
        
        rows, cols = img_float.shape
        crow, ccol = rows//2, cols//2
        
        # High-pass filter
        mask = np.ones((rows, cols))
        mask[crow-25:crow+25, ccol-25:ccol+25] = 0
        
        fft_filtered = fft_shifted * (1 + 0.3 * mask)
        fft_ishifted = fftpack.ifftshift(fft_filtered)
        sharpened_float = np.real(fftpack.ifft2(fft_ishifted))
        sharpened = np.clip(sharpened_float * 255, 0, 255).astype(np.uint8)
    
    else:
        sharpened = image_contrast
    
    return sharpened

# ================== FUNGSI DETEKSI LESI/BENJOLAN ==================

def detect_lesions_accurate(image):
    """
    Deteksi lesi/benjolan dengan akurasi tinggi
    Fokus pada: massa, tumor, area abnormal yang menonjol
    """
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image.copy()
    
    # =========== STEP 1: PREPROCESSING UNTUK LESI ===========
    
    # 1.1. Contrast enhancement khusus untuk lesi
    # Lesi biasanya memiliki kontras berbeda dengan jaringan sekitarnya
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image_gray)
    
    # 1.2. Denoising yang preserve edges (lesi memiliki tepi)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # =========== STEP 2: DETEKSI AREA POTENSIAL LESI ===========
    
    # 2.1. Multi-level thresholding untuk menangkap berbagai intensitas lesi
    mean_intensity = np.mean(denoised)
    std_intensity = np.std(denoised)
    
    # Lesi bisa lebih terang ATAU lebih gelap dari jaringan sekitarnya
    # Threshold untuk lesi terang (hyperintense)
    _, thresh_bright = cv2.threshold(denoised, mean_intensity + 1.5*std_intensity, 255, cv2.THRESH_BINARY)
    
    # Threshold untuk lesi gelap (hypointense)
    _, thresh_dark = cv2.threshold(denoised, mean_intensity - 1.5*std_intensity, 255, cv2.THRESH_BINARY_INV)
    
    # Gabungkan
    potential_lesions = cv2.bitwise_or(thresh_bright, thresh_dark)
    
    # =========== STEP 3: FILTER UNTUK KARAKTERISTIK LESI ===========
    
    # 3.1. Morphological operations untuk membersihkan
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(potential_lesions, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3.2. Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # =========== STEP 4: ANALISIS KONTUR UNTUK IDENTIFIKASI LESI ===========
    
    lesion_contours = []
    lesion_features = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter 1: Area minimum (lesi harus cukup besar)
        if area < 30:  # Minimum 30 pixels
            continue
        
        # Hitung karakteristik bentuk
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity: Lesi biasanya tidak bulat sempurna
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        
        # Aspect ratio: Lesi bisa memanjang atau bulat
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Solidity: Kepadatan bentuk
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Hitung intensitas rata-rata di dalam kontur
        mask_contour = np.zeros_like(image_gray)
        cv2.drawContours(mask_contour, [contour], -1, 255, -1)
        mean_intensity_contour = cv2.mean(denoised, mask=mask_contour)[0]
        
        # Hitung intensitas di sekitar kontur (background)
        mask_dilated = cv2.dilate(mask_contour, kernel, iterations=2)
        mask_background = cv2.subtract(mask_dilated, mask_contour)
        mean_intensity_background = cv2.mean(denoised, mask=mask_background)[0]
        
        # Kontras antara lesi dan background
        contrast = abs(mean_intensity_contour - mean_intensity_background)
        
        # =========== KRITERIA UNTUK MENGIDENTIFIKASI LESI ===========
        # Lesi sejati memiliki:
        # 1. Kontras yang signifikan dengan background (> 15 intensity units)
        # 2. Bentuk tidak terlalu sempurna (circularity < 0.9)
        # 3. Ukuran yang reasonable (tidak terlalu kecil, tidak terlalu besar)
        # 4. Solidity yang reasonable (biasanya < 0.95)
        
        is_potential_lesion = (
            contrast > 15 and                    # Kontras signifikan
            circularity < 0.85 and               # Tidak bulat sempurna
            30 <= area <= 5000 and               # Ukuran reasonable
            solidity < 0.95 and                  # Tidak padat sempurna
            aspect_ratio < 3.0                   # Tidak terlalu memanjang
        )
        
        if is_potential_lesion:
            lesion_contours.append(contour)
            
            # Simpan fitur untuk analisis
            features = {
                'area': area,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'contrast': contrast,
                'mean_intensity': mean_intensity_contour,
                'bounding_box': (x, y, w, h)
            }
            lesion_features.append(features)
    
    # =========== STEP 5: POST-PROCESSING DAN VALIDASI ===========
    
    # 5.1. Group nearby contours (lesi yang berdekatan mungkin bagian dari lesi yang sama)
    if len(lesion_contours) > 1:
        # Hitung jarak antar contours
        grouped_contours = []
        used = [False] * len(lesion_contours)
        
        for i in range(len(lesion_contours)):
            if not used[i]:
                current_group = [lesion_contours[i]]
                used[i] = True
                
                for j in range(i+1, len(lesion_contours)):
                    if not used[j]:
                        # Hitung centroid
                        M1 = cv2.moments(lesion_contours[i])
                        M2 = cv2.moments(lesion_contours[j])
                        
                        if M1["m00"] != 0 and M2["m00"] != 0:
                            cx1 = int(M1["m10"] / M1["m00"])
                            cy1 = int(M1["m01"] / M1["m00"])
                            cx2 = int(M2["m10"] / M2["m00"])
                            cy2 = int(M2["m01"] / M2["m00"])
                            
                            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                            
                            # Jika jarak < 20 pixels, grupkan
                            if distance < 20:
                                current_group.append(lesion_contours[j])
                                used[j] = True
                
                # Gabungkan contours dalam grup
                if len(current_group) > 1:
                    combined_contour = np.vstack(current_group)
                    hull_combined = cv2.convexHull(combined_contour)
                    grouped_contours.append(hull_combined)
                else:
                    grouped_contours.append(current_group[0])
        
        lesion_contours = grouped_contours
    
    # 5.2. Buat mask final
    lesion_mask = np.zeros_like(image_gray)
    cv2.drawContours(lesion_mask, lesion_contours, -1, 255, -1)
    
    # 5.3. Hitung total area lesi
    total_lesion_area = sum(cv2.contourArea(c) for c in lesion_contours)
    
    return lesion_mask, lesion_contours, total_lesion_area, lesion_features

def create_lesion_visualization(original, enhanced, contours, features_list):
    """Visualisasi khusus untuk lesi"""
    if len(original.shape) == 2:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original.copy()
    
    if len(enhanced.shape) == 2:
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    else:
        enhanced_rgb = enhanced.copy()
    
    # Warna untuk lesi berdasarkan ukuran
    colors = [
        (255, 100, 100),    # Merah muda untuk lesi kecil
        (255, 50, 50),      # Merah untuk lesi medium
        (255, 0, 0)         # Merah tua untuk lesi besar
    ]
    
    # Salinan untuk visualisasi
    vis_original = original_rgb.copy()
    vis_enhanced = enhanced_rgb.copy()
    
    # Gambar setiap lesi dengan warna berdasarkan ukuran
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Pilih warna berdasarkan ukuran
        if area < 100:
            color = colors[0]  # Kecil
            thickness = 1
        elif area < 500:
            color = colors[1]  # Medium
            thickness = 2
        else:
            color = colors[2]  # Besar
            thickness = 3
        
        # Gambar contour
        cv2.drawContours(vis_original, [contour], -1, color, thickness)
        cv2.drawContours(vis_enhanced, [contour], -1, color, thickness)
        
        # Tambahkan label area dan intensitas
        if area > 50:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Ambil informasi intensitas dari features
                if i < len(features_list):
                    intensity = features_list[i]['mean_intensity']
                    label = f"A:{area:.0f}"
                else:
                    label = f"{area:.0f}"
                
                # Background putih untuk teks
                cv2.putText(vis_enhanced, label, (cx-20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                # Teks utama
                cv2.putText(vis_enhanced, label, (cx-20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis_original, vis_enhanced

# ================== STREAMLIT UI ==================

st.sidebar.markdown('<div class="section-header">⚙️ Enhancement Settings</div>', unsafe_allow_html=True)

sharpening_method = st.sidebar.selectbox(
    "Sharpening Method:",
    ["Adaptive", "Frequency Domain", "None"],
    index=0
)

st.sidebar.markdown('<div class="section-header">🔍 Lesion Detection</div>', unsafe_allow_html=True)

# Parameter khusus untuk deteksi lesi
contrast_threshold = st.sidebar.slider(
    "Contrast Threshold", 
    5, 50, 15,
    help="Minimum contrast between lesion and surrounding tissue"
)

min_lesion_size = st.sidebar.slider(
    "Minimum Lesion Size (pixels)", 
    10, 200, 30,
    help="Minimum area to be considered as a lesion"
)

max_lesion_size = st.sidebar.slider(
    "Maximum Lesion Size (pixels)", 
    500, 10000, 5000,
    help="Maximum area for a single lesion"
)

# Upload section
st.markdown('<div class="section-header">📁 Upload MRI Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop MRI image",
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff']
)

# Main processing
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if original_image is not None:
        # Resize jika perlu
        if max(original_image.shape) > 800:
            scale = 800 / max(original_image.shape)
            new_size = (int(original_image.shape[1] * scale), 
                       int(original_image.shape[0] * scale))
            original_image = cv2.resize(original_image, new_size)
        
        # ================== SHARPENING ==================
        
        if sharpening_method == "None":
            enhanced_image = original_image.copy()
        else:
            enhanced_image = optimal_sharpening(original_image, method=sharpening_method.lower())
        
        # ================== LESION DETECTION ==================
        
        lesion_mask, lesion_contours, total_area, lesion_features = detect_lesions_accurate(enhanced_image)
        
        # Filter berdasarkan parameter UI
        filtered_contours = []
        filtered_features = []
        
        for i, (contour, features) in enumerate(zip(lesion_contours, lesion_features)):
            area = features['area']
            contrast = features['contrast']
            
            if (area >= min_lesion_size and 
                area <= max_lesion_size and 
                contrast >= contrast_threshold):
                filtered_contours.append(contour)
                filtered_features.append(features)
        
        lesion_contours = filtered_contours
        lesion_features = filtered_features
        total_area = sum(f['area'] for f in lesion_features)
        
        # ================== VISUALIZATION ==================
        
        vis_original, vis_enhanced = create_lesion_visualization(
            original_image, enhanced_image, lesion_contours, lesion_features
        )
        
        # ================== DISPLAY RESULTS ==================
        
        st.markdown('<div class="section-header">📊 Lesion Analysis Results</div>', unsafe_allow_html=True)
        
        if lesion_contours:
            st.markdown(f'''
            <div class="lesion-box">
                <h3 style="color: #ff4444; margin: 0;">⚠️ LESIONS DETECTED</h3>
                <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                    <strong>Number of Lesions:</strong> {len(lesion_contours)}
                </p>
                <p style="margin: 0.5rem 0;">
                    <strong>Total Lesion Area:</strong> {total_area:.0f} pixels
                </p>
                <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #666;">
                    <em>Color coding: Pink (small), Red (medium), Dark Red (large)</em>
                </p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="normal-box">
                <h3 style="color: #44cc44; margin: 0;">✅ NO LESIONS DETECTED</h3>
                <p style="margin: 0.5rem 0;">No significant lesions/masses detected in the MRI image</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # ================== VISUALIZATION SECTION ==================
        
        st.markdown('<div class="section-header">🖼️ Lesion Visualization</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, 
                    caption="Original MRI", 
                    use_container_width=True)
        
        with col2:
            st.image(enhanced_image, 
                    caption=f"Enhanced ({sharpening_method})", 
                    use_container_width=True)
        
        st.markdown("**Lesion Detection Results:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(vis_original, 
                    caption="Lesions Marked on Original", 
                    use_container_width=True)
        
        with col2:
            st.image(vis_enhanced, 
                    caption="Lesions Marked on Enhanced", 
                    use_container_width=True)
        
        # Lesion mask
        st.markdown("**Lesion Detection Mask:**")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(lesion_mask, cmap='hot')
            ax.set_title('Lesion Detection Heatmap')
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            if lesion_features:
                st.markdown("**📈 Lesion Statistics:**")
                areas = [f['area'] for f in lesion_features]
                contrasts = [f['contrast'] for f in lesion_features]
                
                st.write(f"**Total Lesions:** {len(lesion_contours)}")
                st.write(f"**Average Area:** {np.mean(areas):.0f} px")
                st.write(f"**Largest Lesion:** {max(areas):.0f} px")
                st.write(f"**Average Contrast:** {np.mean(contrasts):.1f}")
                
                # Size distribution
                small = sum(1 for a in areas if a < 100)
                medium = sum(1 for a in areas if 100 <= a < 500)
                large = sum(1 for a in areas if a >= 500)
                
                st.write("**Size Distribution:**")
                st.write(f"- Small (<100px): {small}")
                st.write(f"- Medium (100-500px): {medium}")
                st.write(f"- Large (≥500px): {large}")
        
        # ================== DETAILED ANALYSIS ==================
        
        with st.expander("🔍 Detailed Lesion Analysis", expanded=False):
            if lesion_features:
                st.markdown("**Individual Lesion Characteristics:**")
                
                for i, features in enumerate(lesion_features):
                    st.write(f"**Lesion #{i+1}:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"- Area: {features['area']:.0f} pixels")
                        st.write(f"- Contrast: {features['contrast']:.1f}")
                        st.write(f"- Mean Intensity: {features['mean_intensity']:.1f}")
                    
                    with col2:
                        st.write(f"- Circularity: {features['circularity']:.3f}")
                        st.write(f"- Solidity: {features['solidity']:.3f}")
                        st.write(f"- Aspect Ratio: {features['aspect_ratio']:.2f}")
                    
                    st.markdown("---")
                
                # Medical interpretation
                st.markdown("**🩺 Medical Interpretation Notes:**")
                st.info("""
                **Lesion Characteristics:**
                - **High contrast** (>20): Strongly suggests abnormal tissue
                - **Irregular shape** (circularity < 0.7): May indicate malignancy
                - **Large size** (>500px): May require immediate attention
                - **Multiple lesions**: Could indicate metastatic disease
                
                **Note:** This analysis is for preliminary screening only.
                Always consult with a radiologist for definitive diagnosis.
                """)
            else:
                st.info("No lesions detected for detailed analysis.")
        
        # ================== DOWNLOAD SECTION ==================
        
        st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced with lesions
            enhanced_with_lesions_pil = Image.fromarray(vis_enhanced)
            buf_enhanced = io.BytesIO()
            enhanced_with_lesions_pil.save(buf_enhanced, format='PNG')
            buf_enhanced.seek(0)
            
            st.download_button(
                label="📥 Download Lesion Map",
                data=buf_enhanced,
                file_name="mri_lesion_detection.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col2:
            # Report
            report_text = f"""MRI LESION DETECTION REPORT
===============================
File: {uploaded_file.name}
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Image Size: {original_image.shape[1]} x {original_image.shape[0]}

RESULTS:
--------
Lesions Detected: {len(lesion_contours)}
Total Lesion Area: {total_area:.0f} pixels

PARAMETERS:
-----------
Sharpening Method: {sharpening_method}
Contrast Threshold: {contrast_threshold}
Min Lesion Size: {min_lesion_size} pixels
Max Lesion Size: {max_lesion_size} pixels

DETAILED FINDINGS:
------------------
"""
            
            if lesion_features:
                for i, f in enumerate(lesion_features):
                    report_text += f"\nLesion #{i+1}:"
                    report_text += f"\n  Area: {f['area']:.0f} pixels"
                    report_text += f"\n  Contrast: {f['contrast']:.1f}"
                    report_text += f"\n  Mean Intensity: {f['mean_intensity']:.1f}"
                    report_text += f"\n  Circularity: {f['circularity']:.3f}"
                    report_text += f"\n  Solidity: {f['solidity']:.3f}"
            else:
                report_text += "\nNo significant lesions detected."
            
            report_text += "\n\nMEDICAL NOTES:\n--------------"
            report_text += "\n- This report is for preliminary screening only"
            report_text += "\n- Always consult with a radiologist"
            report_text += "\n- Further imaging may be required"
            
            st.download_button(
                label="📄 Download Full Report",
                data=report_text,
                file_name="mri_lesion_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    else:
        st.error("Failed to read image.")
else:
    # Demo/placeholder
    st.info("👆 **Upload an MRI image for lesion detection**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**What We Detect:**")
        st.write("• Tumors/Masses")
        st.write("• Lesions")
        st.write("• Abnormal growths")
        st.write("• Contrast-enhancing regions")
    
    with col2:
        st.markdown("**What We Filter Out:**")
        st.write("• Small noise artifacts")
        st.write("• Blood vessels (linear)")
        st.write("• Normal tissue variations")
        st.write("• Edge artifacts")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "MRI Lesion Detection System | Focused on Mass/Lesion Detection"
    "</div>", 
    unsafe_allow_html=True
)
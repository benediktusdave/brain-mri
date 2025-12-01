import streamlit as st
import cv2
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="MRI Image Sharpening with FFT",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS yang lebih baik
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
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stSlider label {
        font-size: 0.9rem !important;
        font-weight: bold !important;
        color: #2e86ab !important;
    }
    .download-btn {
        background-color: #2e86ab !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown('<div class="main-header">🧠 MRI Image Sharpening & Analysis</div>', unsafe_allow_html=True)

# Informasi aplikasi
with st.expander("ℹ️ Tentang Aplikasi Ini", expanded=True):
    st.markdown("""
    **Aplikasi ini menggunakan FFT (Fast Fourier Transform) untuk:**
    1. **Memperjelas gambar MRI** dengan berbagai metode
    2. **Mendeteksi area abnormal** otomatis
    3. **Mengukur luas area** yang terdeteksi
    
    **Parameter Optimal untuk MRI:**
    - Cutoff Frequency: 25-40 (jangan terlalu rendah)
    - Boost Factor: 0.3-0.8 (jangan terlalu tinggi)
    - Target peningkatan: 15-40% (jangan over 100%)
    """)

# ================== FUNGSI UTAMA ==================

def calculate_sharpness_metric(image):
    """Menghitung tingkat ketajaman gambar"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def preprocess_mri(image):
    """Preprocessing untuk MRI: CLAHE + denoising"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # CLAHE untuk kontras lokal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_clahe = clahe.apply(image)
    
    # Denoising ringan
    image_denoised = cv2.medianBlur(image_clahe, 3)
    
    return image_denoised

def gentle_high_pass_filter(image, cutoff_freq=35, boost_factor=0.5):
    """High-Pass Filter yang lebih gentle untuk MRI"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Preprocessing dulu
    image_processed = preprocess_mri(image)
    
    # Normalisasi ke range 0-1
    image_float = image_processed.astype(np.float32) / 255.0
    
    # FFT
    fft = fftpack.fft2(image_float)
    fft_shifted = fftpack.fftshift(fft)
    
    # Magnitude spectrum untuk visualisasi
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
    
    rows, cols = image_float.shape
    crow, ccol = rows // 2, cols // 2
    
    # Buat filter high-pass yang lebih gentle
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Gaussian high-pass filter (lebih smooth)
    mask = 1 - np.exp(-(distance**2) / (2 * (cutoff_freq**2)))
    
    # Apply filter dengan boost factor yang lebih terkontrol
    fft_filtered = fft_shifted * (1 + boost_factor * mask)
    
    # Inverse FFT
    fft_ishifted = fftpack.ifftshift(fft_filtered)
    image_sharpened = np.real(fftpack.ifft2(fft_ishifted))
    
    # Normalize back
    image_sharpened = np.clip(image_sharpened * 255, 0, 255).astype(np.uint8)
    
    return image_sharpened, magnitude_spectrum

def detect_anomalies(image):
    """Deteksi area abnormal dengan multiple methods"""
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image.copy()
    
    # Method 1: Otsu's thresholding
    _, thresh_otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive thresholding
    thresh_adapt = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Combine results
    combined = cv2.bitwise_or(thresh_otsu, thresh_adapt)
    
    # Clean up
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    min_area = 50
    valid_contours = []
    total_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            valid_contours.append(contour)
            total_area += area
    
    # Create mask
    mask = np.zeros_like(image_gray)
    cv2.drawContours(mask, valid_contours, -1, 255, -1)
    
    is_abnormal = total_area > 100
    
    return mask, valid_contours, total_area, is_abnormal

def visualize_results(original, processed, contours, mask):
    """Visualisasi hasil dengan segmentasi"""
    # Convert to RGB for visualization
    if len(original.shape) == 2:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original
    
    if len(processed.shape) == 2:
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    else:
        processed_rgb = processed
    
    # Draw contours
    original_with_contours = original_rgb.copy()
    processed_with_contours = processed_rgb.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Only draw significant contours
            color = (255, 0, 0) if area > 200 else (255, 165, 0)
            thickness = 2 if area > 200 else 1
            
            cv2.drawContours(original_with_contours, [contour], -1, color, thickness)
            cv2.drawContours(processed_with_contours, [contour], -1, color, thickness)
    
    return original_with_contours, processed_with_contours

# ================== SIDEBAR ==================

st.sidebar.markdown('<div class="section-header">⚙️ Parameter Processing</div>', unsafe_allow_html=True)

# Pilihan metode
method = st.sidebar.radio(
    "Pilih Metode:",
    ["High-Pass Filter", "Unsharp Masking"],
    index=0
)

if method == "High-Pass Filter":
    st.sidebar.markdown("**Parameter High-Pass Filter:**")
    
    cutoff_freq = st.sidebar.slider(
        "Cutoff Frequency", 
        10, 100, 35,
        help="Rekomendasi: 25-40. Nilai rendah = sharpening kuat"
    )
    
    boost_factor = st.sidebar.slider(
        "Boost Factor", 
        0.1, 2.0, 0.5, 0.1,
        help="Rekomendasi: 0.3-0.8. Jangan lebih dari 1.0 untuk hasil natural"
    )
    
else:  # Unsharp Masking
    st.sidebar.markdown("**Parameter Unsharp Masking:**")
    
    sigma = st.sidebar.slider(
        "Blur Radius (Sigma)", 
        0.5, 3.0, 1.5, 0.1,
        help="Radius blur untuk mask"
    )
    
    strength = st.sidebar.slider(
        "Strength", 
        0.1, 1.5, 0.6, 0.1,
        help="Kekuatan sharpening. Rekomendasi: 0.4-0.8"
    )

# Parameter deteksi
st.sidebar.markdown('<div class="section-header">🔍 Deteksi Anomali</div>', unsafe_allow_html=True)

detection_enabled = st.sidebar.checkbox("Aktifkan Deteksi Otomatis", value=True)

if detection_enabled:
    min_area = st.sidebar.slider(
        "Minimum Area (pixels)",
        10, 200, 50,
        help="Area minimum untuk dianggap sebagai anomali"
    )

# ================== MAIN AREA ==================

st.markdown('<div class="section-header">📁 Upload Gambar MRI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop file gambar MRI di sini (JPG, PNG, TIFF)",
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
    key="file_uploader"
)

# Processing
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if original_image is not None:
        # Resize jika terlalu besar
        if max(original_image.shape) > 800:
            scale = 800 / max(original_image.shape)
            new_size = (int(original_image.shape[1] * scale), 
                       int(original_image.shape[0] * scale))
            original_image = cv2.resize(original_image, new_size)
        
        # Processing
        start_time = time.time()
        
        if method == "High-Pass Filter":
            processed_image, magnitude_spectrum = gentle_high_pass_filter(
                original_image, cutoff_freq, boost_factor
            )
        else:  # Unsharp Masking
            # Simple unsharp masking
            blurred = cv2.GaussianBlur(original_image, (0, 0), sigma)
            processed_image = cv2.addWeighted(
                original_image, 1.0 + strength, 
                blurred, -strength, 0
            )
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
            magnitude_spectrum = None
        
        processing_time = (time.time() - start_time) * 1000  # in ms
        
        # Anomaly detection
        if detection_enabled:
            anomaly_mask, contours, anomaly_area, is_abnormal = detect_anomalies(processed_image)
            original_vis, processed_vis = visualize_results(
                original_image, processed_image, contours, anomaly_mask
            )
        else:
            anomaly_area = 0
            is_abnormal = False
            if len(original_image.shape) == 2:
                original_vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                processed_vis = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
            else:
                original_vis = original_image
                processed_vis = processed_image
        
        # Calculate metrics
        original_sharpness = calculate_sharpness_metric(original_image)
        processed_sharpness = calculate_sharpness_metric(processed_image)
        improvement = ((processed_sharpness - original_sharpness) / original_sharpness * 100 
                      if original_sharpness > 0 else 0)
        
        # ================== DISPLAY RESULTS ==================
        
        # Status indicator
        st.markdown('<div class="section-header">📊 Hasil Analisis</div>', unsafe_allow_html=True)
        
        if detection_enabled:
            if is_abnormal:
                st.warning(f"🚨 **ANOMALI DETECTED!** - Area: {anomaly_area:.0f} pixels")
            else:
                st.success("✅ **TIDAK ADA ANOMALI** terdeteksi")
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Sharpness Awal", f"{original_sharpness:.1f}", "#666"),
            ("Sharpness Hasil", f"{processed_sharpness:.1f}", "#2e86ab"),
            ("Peningkatan", f"{improvement:+.1f}%", 
             "green" if 15 <= improvement <= 50 else "orange" if improvement > 0 else "red"),
            ("Waktu", f"{processing_time:.0f} ms", "#666")
        ]
        
        for i, (title, value, color) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.markdown(f'''
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: {color};">{value}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Quality warning
        if improvement > 100:
            st.error("⚠️ **PERINGATAN:** Peningkatan >100% kemungkinan over-sharpening! Turunkan Boost Factor.")
        elif improvement < 5:
            st.info("💡 **TIP:** Hasil masih lembut. Naikkan sedikit Boost Factor.")
        
        # ================== VISUALIZATIONS ==================
        
        st.markdown('<div class="section-header">🖼️ Visualisasi Hasil</div>', unsafe_allow_html=True)
        
        # Main comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.imshow(original_vis if detection_enabled else original_image, cmap='gray')
        ax1.set_title('MRI Original', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        ax2.imshow(processed_vis if detection_enabled else processed_image, cmap='gray')
        ax2.set_title(f'Hasil {method}', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional visualizations
        if method == "High-Pass Filter" and magnitude_spectrum is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Magnitude Spectrum (FFT)**")
                fig_spec, ax = plt.subplots(figsize=(6, 5))
                ax.imshow(magnitude_spectrum, cmap='gray')
                ax.set_title('Domain Frekuensi', fontsize=12)
                ax.axis('off')
                st.pyplot(fig_spec)
            
            with col2:
                st.markdown("**🔍 Perubahan Detail**")
                diff = cv2.absdiff(original_image, processed_image)
                diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
                fig_diff, ax = plt.subplots(figsize=(6, 5))
                ax.imshow(diff_normalized, cmap='hot')
                ax.set_title('Area yang Diperjelas', fontsize=12)
                ax.axis('off')
                st.pyplot(fig_diff)
        
        # ================== DOWNLOAD ==================
        
        st.markdown('<div class="section-header">💾 Download Hasil</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Processed image
            processed_pil = Image.fromarray(processed_image)
            buf_processed = io.BytesIO()
            processed_pil.save(buf_processed, format='PNG')
            buf_processed.seek(0)
            
            st.download_button(
                label="📥 Download Gambar Hasil",
                data=buf_processed,
                file_name=f"mri_enhanced_{method.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col2:
            # Report
            report_text = f"""LAPORAN MRI ENHANCEMENT
==========================
File: {uploaded_file.name}
Tanggal: {time.strftime("%Y-%m-%d %H:%M:%S")}
Metode: {method}

HASIL:
- Sharpness Awal: {original_sharpness:.1f}
- Sharpness Hasil: {processed_sharpness:.1f}
- Peningkatan: {improvement:+.1f}%
- Status Anomali: {'DETECTED' if is_abnormal else 'NORMAL'}
- Area Anomali: {anomaly_area:.0f} pixels

PARAMETER:
{'- Cutoff: ' + str(cutoff_freq) if method == 'High-Pass Filter' else '- Sigma: ' + str(sigma)}
{'- Boost: ' + str(boost_factor) if method == 'High-Pass Filter' else '- Strength: ' + str(strength)}
"""
            
            st.download_button(
                label="📄 Download Laporan",
                data=report_text,
                file_name="mri_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # ================== TIPS ==================
        
        with st.expander("💡 Tips untuk Hasil Optimal"):
            st.markdown("""
            **Parameter yang Direkomendasikan:**
            - **Cutoff Frequency:** 25-40
            - **Boost Factor:** 0.3-0.8
            - **Target Peningkatan:** 15-40%
            
            **Indikator Hasil Baik:**
            1. Detail lebih jelas tanpa halo effect
            2. Noise tidak meningkat drastis
            3. Gambar terlihat natural
            4. Peningkatan sharpness 15-40%
            
            **Jika hasil over-sharpened:**
            - Turunkan Boost Factor
            - Naikkan Cutoff Frequency
            - Gunakan Unsharp Masking sebagai alternatif
            """)
    
    else:
        st.error("Gagal membaca gambar. Pastikan file valid.")
else:
    # Placeholder
    st.info("👆 **Upload gambar MRI untuk memulai**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Contoh parameter baik:**")
        st.code("""
        Cutoff Frequency: 35
        Boost Factor: 0.5
        Hasil: +25% improvement
        """)
    
    with col2:
        st.markdown("**Parameter over-sharpening:**")
        st.code("""
        Cutoff Frequency: 18
        Boost Factor: 1.2
        Hasil: +237% (❌ terlalu tinggi!)
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "MRI FFT Sharpening Tool | Untuk Keperluan Riset & Edukasi"
    "</div>", 
    unsafe_allow_html=True
)
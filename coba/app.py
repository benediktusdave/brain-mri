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

# Custom CSS untuk tampilan yang lebih menarik
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown('<div class="main-header">🧠 MRI Image Sharpening with FFT</div>', unsafe_allow_html=True)

# Informasi aplikasi
with st.expander("ℹ️ Tentang Aplikasi Ini", expanded=True):
    st.markdown("""
    **Aplikasi ini menggunakan FFT (Fast Fourier Transform) untuk memperjelas gambar MRI dengan:**
    - **High-Pass Filtering**: Mempertajam detail halus dan tepi
    - **Unsharp Masking**: Meningkatkan kontras dan ketajaman
    - **Butterworth Filter**: Filter dengan transisi smooth untuk mengurangi artifact
    
    **Cara penggunaan:**
    1. Upload gambar MRI (JPEG, PNG, DICOM*)
    2. Atur parameter sharpening di sidebar
    3. Lihat hasil real-time
    4. Bandingkan sebelum dan sesudah
    *Note: Untuk DICOM butuh library tambahan (pydicom)
    """)

# Fungsi-fungsi processing
def calculate_sharpness_metric(image):
    """Menghitung tingkat ketajaman gambar menggunakan variance of Laplacian"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def high_pass_filter_sharpen(image, cutoff_freq=30, boost_factor=1.0):
    """Sharpening menggunakan High-Pass Filter di domain FFT"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalisasi image ke range 0-1
    image_float = image.astype(np.float32) / 255.0
    
    # FFT
    fft = fftpack.fft2(image_float)
    fft_shifted = fftpack.fftshift(fft)
    
    # Buat high-pass filter
    rows, cols = image_float.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create circular mask
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff_freq**2
    mask = np.ones((rows, cols))
    mask[mask_area] = 0  # Block low frequencies
    
    # Apply filter dengan boost factor
    fft_filtered = fft_shifted * (1 + boost_factor * mask)
    
    # Inverse FFT
    fft_ishifted = fftpack.ifftshift(fft_filtered)
    image_sharpened = np.real(fftpack.ifft2(fft_ishifted))
    
    # Normalize back to 0-255
    image_sharpened = np.clip(image_sharpened * 255, 0, 255).astype(np.uint8)
    
    return image_sharpened

def unsharp_mask_filter(image, sigma=1.0, strength=1.0):
    """Sharpening menggunakan Unsharp Masking"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Buat blurred version
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Calculate sharpened image
    sharpened = float(strength + 1) * image - float(strength) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened

def butterworth_high_pass_filter(image, cutoff_freq=30, order=2):
    """Sharpening menggunakan Butterworth High-Pass Filter"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image_float = image.astype(np.float32) / 255.0
    
    # FFT
    fft = fftpack.fft2(image_float)
    fft_shifted = fftpack.fftshift(fft)
    
    # Buat Butterworth filter
    rows, cols = image_float.shape
    crow, ccol = rows // 2, cols // 2
    
    # Distance matrix
    u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    d = np.sqrt(u**2 + v**2)
    
    # Butterworth transfer function
    h = 1 / (1 + (cutoff_freq / (d + 1e-6))**(2 * order))
    
    # Apply filter
    fft_filtered = fft_shifted * h
    
    # Inverse FFT
    fft_ishifted = fftpack.ifftshift(fft_filtered)
    image_sharpened = np.real(fftpack.ifft2(fft_ishifted))
    
    image_sharpened = np.clip(image_sharpened * 255, 0, 255).astype(np.uint8)
    return image_sharpened

def plot_comparison(original, processed, title1="Original", title2="Processed"):
    """Plot perbandingan side-by-side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title(title1, fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(processed, cmap='gray')
    ax2.set_title(title2, fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# Sidebar untuk kontroller
st.sidebar.markdown('<div class="section-header">⚙️ Parameter Sharpening</div>', unsafe_allow_html=True)

# Pilihan metode
method = st.sidebar.selectbox(
    "Pilih Metode Sharpening:",
    ["High-Pass Filter", "Unsharp Masking", "Butterworth Filter"],
    index=0
)

# Parameter berdasarkan metode
if method == "High-Pass Filter":
    cutoff_freq = st.sidebar.slider("Cutoff Frequency", 10, 100, 30, 
                                   help="Frekuensi cutoff untuk high-pass filter. Nilai lebih rendah = sharpening lebih kuat")
    boost_factor = st.sidebar.slider("Boost Factor", 0.1, 3.0, 1.0, 0.1,
                                    help="Faktor penguatan untuk frekuensi tinggi")
    
elif method == "Unsharp Masking":
    sigma = st.sidebar.slider("Blur Sigma", 0.1, 5.0, 1.0, 0.1,
                             help="Tingkat blur untuk mask (sigma Gaussian)")
    strength = st.sidebar.slider("Strength", 0.1, 3.0, 1.0, 0.1,
                                help="Kekuatan sharpening")
    
else:  # Butterworth Filter
    cutoff_freq = st.sidebar.slider("Cutoff Frequency", 10, 100, 30)
    order = st.sidebar.slider("Filter Order", 1, 5, 2,
                             help="Urutan filter - lebih tinggi = transisi lebih tajam")

# Upload gambar
st.markdown('<div class="section-header">📁 Upload Gambar MRI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop file gambar MRI di sini",
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
    help="Format yang didukung: JPG, JPEG, PNG, TIFF"
)

# Main processing area
if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if original_image is not None:
        # Proses gambar berdasarkan metode yang dipilih
        start_time = time.time()
        
        if method == "High-Pass Filter":
            processed_image = high_pass_filter_sharpen(original_image, cutoff_freq, boost_factor)
        elif method == "Unsharp Masking":
            processed_image = unsharp_mask_filter(original_image, sigma, strength)
        else:  # Butterworth Filter
            processed_image = butterworth_high_pass_filter(original_image, cutoff_freq, order)
        
        processing_time = time.time() - start_time
        
        # Hitung metrics
        original_sharpness = calculate_sharpness_metric(original_image)
        processed_sharpness = calculate_sharpness_metric(processed_image)
        improvement = ((processed_sharpness - original_sharpness) / original_sharpness) * 100
        
        # Tampilkan metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-box">'
                       f'<div>Sharpness Original</div>'
                       f'<div style="font-size: 1.5rem; font-weight: bold;">{original_sharpness:.2f}</div>'
                       f'</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-box">'
                       f'<div>Sharpness Hasil</div>'
                       f'<div style="font-size: 1.5rem; font-weight: bold;">{processed_sharpness:.2f}</div>'
                       f'</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-box">'
                       f'<div>Peningkatan</div>'
                       f'<div style="font-size: 1.5rem; font-weight: bold; color: {"green" if improvement > 0 else "red"};">'
                       f'{improvement:+.1f}%</div>'
                       f'</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-box">'
                       f'<div>Waktu Proses</div>'
                       f'<div style="font-size: 1.5rem; font-weight: bold;">{processing_time*1000:.1f} ms</div>'
                       f'</div>', unsafe_allow_html=True)
        
        # Tampilkan gambar
        st.markdown('<div class="section-header">🖼️ Hasil Perbandingan</div>', unsafe_allow_html=True)
        
        # Plot side-by-side
        fig = plot_comparison(original_image, processed_image, 
                             "Gambar MRI Original", f"Hasil {method}")
        st.pyplot(fig)
        
        # Difference map
        st.markdown('<div class="section-header">🔍 Peta Perubahan</div>', unsafe_allow_html=True)
        
        # Hitung difference
        diff = cv2.absdiff(original_image, processed_image)
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        # Plot difference
        fig_diff, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(diff, cmap='hot')
        ax1.set_title('Peta Perubahan (Heatmap)', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(diff_normalized, cmap='gray')
        ax2.set_title('Perubahan (Enhanced)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig_diff)
        
        # Download hasil
        st.markdown('<div class="section-header">💾 Download Hasil</div>', unsafe_allow_html=True)
        
        # Convert processed image to bytes for download
        processed_pil = Image.fromarray(processed_image)
        buf = io.BytesIO()
        processed_pil.save(buf, format='PNG')
        buf.seek(0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Gambar Hasil",
                data=buf,
                file_name=f"mri_sharpened_{method.lower().replace(' ', '_')}.png",
                mime="image/png",
                help="Download gambar yang telah di-sharpen"
            )
        
        with col2:
            # Download difference map
            diff_pil = Image.fromarray(diff_normalized)
            diff_buf = io.BytesIO()
            diff_pil.save(diff_buf, format='PNG')
            diff_buf.seek(0)
            
            st.download_button(
                label="📥 Download Peta Perubahan",
                data=diff_buf,
                file_name=f"mri_difference_map.png",
                mime="image/png",
                help="Download peta perubahan antara original dan hasil"
            )
        
        # Informasi tambahan
        with st.expander("📊 Analisis Detail"):
            st.write(f"**Metode yang digunakan:** {method}")
            if method == "High-Pass Filter":
                st.write(f"- Cutoff Frequency: {cutoff_freq}")
                st.write(f"- Boost Factor: {boost_factor}")
            elif method == "Unsharp Masking":
                st.write(f"- Sigma Blur: {sigma}")
                st.write(f"- Strength: {strength}")
            else:
                st.write(f"- Cutoff Frequency: {cutoff_freq}")
                st.write(f"- Filter Order: {order}")
            
            st.write(f"**Dimensi Gambar:** {original_image.shape[1]} x {original_image.shape[0]} piksel")
            st.write(f"**Peningkatan Ketajaman:** {improvement:+.1f}%")
            
            if improvement > 0:
                st.success("✅ Sharpening berhasil meningkatkan kualitas gambar")
            else:
                st.warning("⚠️ Sharpening tidak memberikan peningkatan. Coba ubah parameter.")
    
    else:
        st.error("❌ Gagal membaca gambar. Pastikan file yang diupload adalah gambar yang valid.")

else:
    # Placeholder ketika belum ada gambar
    st.info("👆 Silakan upload gambar MRI untuk memulai processing")
    
    # Contoh preview
    col1, col2, col3 = st.columns(3)
    
    with col2:
        st.image("https://via.placeholder.com/400x300/1f77b4/ffffff?text=Upload+MRI+Image", 
                caption="Area preview akan menampilkan gambar MRI Anda", 
                use_column_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Dibuat dengan Streamlit | MRI FFT Sharpening Tool | 2024"
    "</div>", 
    unsafe_allow_html=True
)
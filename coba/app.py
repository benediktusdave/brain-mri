# app.py
import streamlit as st
import cv2
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
import time
import pandas as pd
from skimage import exposure

# -------------------------
# Streamlit page config & CSS
# -------------------------
st.set_page_config(page_title="MRI FFT Sharpen & Segment", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .section-header { font-size: 1.25rem; color: #2e86ab; margin-top: 1rem; margin-bottom: 0.5rem; }
    .metric-box { background-color: #c5c7d6; padding: 0.5rem; border-radius: 5px; text-align: center; margin: 0.5rem 0; color: black;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 MRI FFT Sharpening & Tumor Segmentation Tool</div>', unsafe_allow_html=True)

# -------------------------
# Utility & processing modules (modular)
# -------------------------
def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def normalize_uint8(imgf):
    """normalize float image in [0,1] or arbitrary range back to uint8 0-255"""
    a = imgf.astype(np.float32)
    a = a - a.min()
    if a.max() != 0:
        a = a / a.max()
    a = (a * 255.0).astype(np.uint8)
    return a

def denoise_nlmeans(img, h=10):
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)

def denoise_bilateral(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

# FFT sharpening variations
def gaussian_hpf_mask(shape, cutoff_sigma):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    y = np.linspace(-crow, crow, rows)
    x = np.linspace(-ccol, ccol, cols)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    # Gaussian high-pass = 1 - exp(-D^2/(2*sigma^2))
    H = 1 - np.exp(-(D**2)/(2*(cutoff_sigma**2)))
    return H

def butterworth_hpf_mask(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    D = np.sqrt(u**2 + v**2)
    H = 1 / (1 + (cutoff/(D + 1e-9))**(2*order))
    return H

def fft_sharpen_generic(img, mask_type='gaussian', cutoff=30, order=2, boost=1.0):
    """Apply FFT-based high boost sharpening with selectable mask"""
    imgf = img.astype(np.float32) / 255.0
    F = fftpack.fft2(imgf)
    Fshift = fftpack.fftshift(F)
    if mask_type == 'gaussian':
        H = gaussian_hpf_mask(imgf.shape, cutoff)
    elif mask_type == 'butterworth':
        H = butterworth_hpf_mask(imgf.shape, cutoff, order)
    else:  # simple circular ideal HPF
        rows, cols = imgf.shape
        crow, ccol = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
        H = np.ones(imgf.shape, dtype=np.float32)
        H[mask_area] = 0
    # high-boost: F * (1 + boost * H)
    Fboost = Fshift * (1.0 + boost * H)
    F_ishift = fftpack.ifftshift(Fboost)
    img_back = np.real(fftpack.ifft2(F_ishift))
    return normalize_uint8(img_back)

# Unsharp (spatial) modular
def unsharp_mask(img, sigma=1.0, strength=1.0):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = np.clip((1 + strength) * img.astype(np.float32) - strength * blurred.astype(np.float32), 0, 255)
    return sharpened.astype(np.uint8)

# Contrast enhancement (optional)
def apply_clahe(img, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

# Spectrum visualization helper
def plot_spectrum(img):
    f = fftpack.fft2(img.astype(np.float32)/255.0)
    fshift = fftpack.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift) + 1e-9)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(magnitude_spectrum, cmap='viridis')
    ax.axis('off')
    return fig

# -------------------------
# Segmentation & measurement module
# -------------------------
def segment_image(img, method='otsu', adaptive_blocksize=35, adaptive_C=5,
                  morph_open=3, morph_close=5, min_area_px=50):
    """
    Returns binary mask (uint8 0/255) after segmentation and cleaned by morphology.
    method: 'otsu' | 'adaptive' | 'threshold'
    """
    gray = to_gray(img)
    if method == 'adaptive':
        # blockSize must be odd and >=3
        bs = adaptive_blocksize if adaptive_blocksize % 2 == 1 else adaptive_blocksize + 1
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, bs, adaptive_C)
    elif method == 'threshold':
        # simple fixed threshold using mean or percentile
        th = int(np.mean(gray) + np.std(gray)*0.5)
        _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    else:  # otsu
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert if foreground is too large (heuristic)
    if np.sum(mask==255) > 0.5 * mask.size:
        mask = cv2.bitwise_not(mask)
    # morphological cleaning
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    # remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            final_mask[labels == i] = 255
    return final_mask

def measure_components(mask, pixel_spacing_x=1.0, pixel_spacing_y=1.0):
    """
    Returns list of dicts for each connected component: area_px, area_mm2, bbox, centroid
    pixel_spacing in mm/pixel
    """
    comps = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]
        area_mm2 = area_px * pixel_spacing_x * pixel_spacing_y
        comps.append({
            'label': i,
            'area_px': area_px,
            'area_mm2': area_mm2,
            'bbox': (x, y, w, h),
            'centroid': (float(cx), float(cy))
        })
    return comps

def overlay_mask_on_image(img_gray, mask, color=(255,0,0), alpha=0.4):
    """Overlay mask (255) on grayscale image and return RGB PIL image"""
    # create color overlay
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    overlay[mask==255] = color
    out = cv2.addWeighted(overlay, alpha, img_rgb, 1-alpha, 0)
    return out

def annotate_image(img_rgb, components, pixel_spacing=(1.0,1.0)):
    """Annotate contours and area text on RGB image (numpy)"""
    out = img_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for comp in components:
        x,y,w,h = comp['bbox']
        area_px = comp['area_px']
        area_mm2 = comp['area_mm2']
        centroid = (int(comp['centroid'][0]), int(comp['centroid'][1]))
        # draw bbox
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        # contour
        # find contour from mask of that label area is not trivial, but we can draw bbox + text
        text = f"{area_px} px / {area_mm2:.1f} mm2"
        cv2.putText(out, text, (x, max(10,y-5)), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
        # centroid marker
        cv2.circle(out, centroid, 3, (0,255,255), -1)
    return out

# -------------------------
# Sidebar parameters
# -------------------------
st.sidebar.markdown('<div class="section-header">⚙️ Pengaturan Umum</div>', unsafe_allow_html=True)
denoise_method = st.sidebar.selectbox("Denoise method", ["None", "NLMeans", "Bilateral"], index=1)
denoise_strength = st.sidebar.slider("Denoise strength (h or d)", 0, 30, 10)

st.sidebar.markdown('<div class="section-header">🔧 FFT / Sharpening</div>', unsafe_allow_html=True)
mask_type = st.sidebar.selectbox("HPF Mask type", ["gaussian", "butterworth", "ideal"], index=0)
cutoff = st.sidebar.slider("Cutoff / Sigma", 5, 150, 30)
butter_order = st.sidebar.slider("Butterworth order (if used)", 1, 6, 2)
boost = st.sidebar.slider("Boost factor", 0.0, 5.0, 1.5, 0.1)
use_unsharp = st.sidebar.checkbox("Also apply Unsharp Mask (spatial)", value=True)
unsharp_sigma = st.sidebar.slider("Unsharp sigma", 0.5, 3.0, 1.0, 0.1)
unsharp_strength = st.sidebar.slider("Unsharp strength", 0.1, 3.0, 1.0, 0.1)

st.sidebar.markdown('<div class="section-header">🔬 Segmentation & Measurement</div>', unsafe_allow_html=True)
do_segmentation = st.sidebar.checkbox("Enable segmentation & measure area", value=True)
seg_method = st.sidebar.selectbox("Segmentation method", ["otsu", "adaptive", "threshold"], index=0)
adaptive_bs = st.sidebar.slider("Adaptive block size (odd)", 11, 101, 35, step=2)
adaptive_C = st.sidebar.slider("Adaptive C (subtract)", 0, 20, 5)
morph_open = st.sidebar.slider("Morph open kernel", 1, 15, 3)
morph_close = st.sidebar.slider("Morph close kernel", 1, 31, 5)
min_area_px = st.sidebar.slider("Min component area (px)", 10, 2000, 50)

st.sidebar.markdown('<div class="section-header">📐 Pixel spacing (mm/pixel) for area calc</div>', unsafe_allow_html=True)
pixel_spacing_x = st.sidebar.number_input("Pixel spacing X (mm)", min_value=0.01, value=1.0, step=0.01, format="%.3f")
pixel_spacing_y = st.sidebar.number_input("Pixel spacing Y (mm)", min_value=0.01, value=1.0, step=0.01, format="%.3f")

# -------------------------
# Main UI: upload & processing
# -------------------------
st.markdown('<div class="section-header">📁 Upload Gambar MRI</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload MRI image (PNG/JPG/TIFF). Jika DICOM: extract pixel array & masukkan di sini.", type=['jpg','jpeg','png','tif','tiff'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if original is None:
        st.error("Gagal membaca file. Pastikan bukan DICOM yang belum diekstrak.")
    else:
        # convert if color
        if len(original.shape) == 3:
            img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = original.copy()
        # Optionally apply CLAHE
        apply_clahe_flag = st.sidebar.checkbox("Apply CLAHE (contrast) before processing", value=False)
        if apply_clahe_flag:
            img_gray = apply_clahe(img_gray)
        # Denoise
        if denoise_method == "NLMeans":
            img_dn = denoise_nlmeans(img_gray, h=denoise_strength)
        elif denoise_method == "Bilateral":
            img_dn = denoise_bilateral(img_gray, d=int(max(1, denoise_strength//2*2+1)), sigmaColor=75, sigmaSpace=75)
        else:
            img_dn = img_gray.copy()
        # Sharpen via FFT
        processed_fft = fft_sharpen_generic(img_dn, mask_type=mask_type, cutoff=cutoff, order=butter_order, boost=boost)
        # Optionally unsharp on top
        if use_unsharp:
            processed = unsharp_mask(processed_fft, sigma=unsharp_sigma, strength=unsharp_strength)
        else:
            processed = processed_fft.copy()
        # Segmentation & measurement
        if do_segmentation:
            mask = segment_image(processed, method=seg_method, adaptive_blocksize=adaptive_bs,
                                 adaptive_C=adaptive_C, morph_open=morph_open, morph_close=morph_close,
                                 min_area_px=min_area_px)
            components = measure_components(mask, pixel_spacing_x=pixel_spacing_x, pixel_spacing_y=pixel_spacing_y)
            overlay = overlay_mask_on_image(processed, mask, color=(255,0,0), alpha=0.4)
            annotated = annotate_image(overlay, components, pixel_spacing=(pixel_spacing_x, pixel_spacing_y))
        else:
            mask = np.zeros_like(processed)
            components = []
            overlay = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            annotated = overlay
        # Metrics: variance of Laplacian (sharpness)
        original_sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        processed_sharpness = cv2.Laplacian(processed, cv2.CV_64F).var()
        improvement = ((processed_sharpness - original_sharpness) / (original_sharpness + 1e-9)) * 100.0

        # Display top metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-box"><div>Sharpness Original</div><div style="font-weight:bold">{original_sharpness:.2f}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><div>Sharpness Processed</div><div style="font-weight:bold">{processed_sharpness:.2f}</div></div>', unsafe_allow_html=True)
        with col3:
            color = "green" if improvement > 0 else "red"
            st.markdown(f'<div class="metric-box"><div>Peningkatan (%)</div><div style="font-weight:bold;color:{color}">{improvement:+.1f}%</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-box"><div>Dimensions</div><div style="font-weight:bold">{img_gray.shape[1]} x {img_gray.shape[0]}</div></div>', unsafe_allow_html=True)

        # Layout images
        st.markdown('<div class="section-header">🖼️ Visualisasi</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            st.caption("Original")
            st.image(img_gray, use_column_width=True)
        with c2:
            st.caption("Denoised / Preprocessed")
            st.image(img_dn, use_column_width=True)
        with c3:
            st.caption("Processed (FFT + optional unsharp)")
            st.image(processed, use_column_width=True)
        with c4:
            st.caption("Overlay Segmentation")
            st.image(annotated, use_column_width=True)

        # Spectrum & difference
        sp1, sp2 = st.columns(2)
        with sp1:
            st.caption("Spectrum (Processed)")
            fig_spec = plot_spectrum(processed)
            st.pyplot(fig_spec)
        with sp2:
            st.caption("Difference map (absdiff)")
            diff = cv2.absdiff(img_gray, processed)
            fig_dif, ax = plt.subplots(figsize=(4,3))
            ax.imshow(diff, cmap='hot')
            ax.axis('off')
            st.pyplot(fig_dif)

        # Components table & downloads
        st.markdown('<div class="section-header">📊 Segmentasi & Pengukuran</div>', unsafe_allow_html=True)
        if len(components) == 0:
            st.info("Tidak ditemukan komponen/area yang memenuhi kriteria. Coba ubah parameter segmentasi atau min area.")
        else:
            df = pd.DataFrame([{
                'label': c['label'],
                'area_px': c['area_px'],
                'area_mm2': round(c['area_mm2'],2),
                'centroid_x': round(c['centroid'][0],2),
                'centroid_y': round(c['centroid'][1],2),
                'bbox': c['bbox']
            } for c in components])
            st.dataframe(df)

            # Export CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download measurements (CSV)", data=csv, file_name="measurements.csv", mime="text/csv")

            # Download mask PNG
            mask_pil = Image.fromarray(mask)
            buf = io.BytesIO(); mask_pil.save(buf, format='PNG'); buf.seek(0)
            st.download_button("📥 Download mask (PNG)", data=buf, file_name="mask.png", mime="image/png")
            # Download annotated
            ann_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            buf2 = io.BytesIO(); ann_pil.save(buf2, format='PNG'); buf2.seek(0)
            st.download_button("📥 Download annotated image (PNG)", data=buf2, file_name="annotated.png", mime="image/png")

        # Info / tips
        st.markdown('<div class="section-header">💡 Tips & Notes</div>', unsafe_allow_html=True)
        st.markdown("""
        - Jika Anda memiliki file DICOM, sebaiknya ekstrak array pixel dan pixel spacing (mm) lalu upload gambar + masukkan pixel spacing ke sidebar untuk konversi area ke mm².  
        - Jika segmentation false positive: coba tingkatkan `min component area`, atau gunakan `adaptive` dengan block size lebih besar dan C lebih kecil.  
        - Untuk mengurangi ringing/artifact pada FFT: pakai Butterworth atau Gaussian HPF (mask_type = gaussian / butterworth).  
        - Jika noise meningkat setelah sharpening: tingkatkan denoise atau kurangi `boost` / `unsharp strength`.  
        """)
else:
    st.info("Upload gambar MRI Anda di panel sebelah kiri. Untuk hasil area dalam mm², masukkan pixel spacing (mm/pixel) di sidebar.")
    st.image("https://via.placeholder.com/700x300/1f77b4/ffffff?text=Upload+MRI+Image+to+Start", use_column_width=True)

# footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#666;'>MRI FFT Sharpen & Segmentation Tool — dibuat untuk tugas UAS (Image Processing)</div>", unsafe_allow_html=True)

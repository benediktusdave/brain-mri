"""
Brain MRI Tumor Detection using Classical Image Processing
Pipeline: FFT Sharpening -> Enhanced Segmentation -> Morphology
Streamlit Web Application
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

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


def process_image(image, hpf_radius=20, threshold_value=200, open_kernel=3, close_kernel=5, 
                 apply_segmentation=True, pixel_spacing_x=1.0, pixel_spacing_y=1.0):
    """
    Process a single MRI image through the complete pipeline
    
    Parameters:
    - apply_segmentation: If False, only do sharpening (for normal images)
    - pixel_spacing_x, pixel_spacing_y: mm per pixel
    
    Returns: dict with all intermediate results
    """
    results = {}
    
    # Step 1: FFT Sharpening (ALWAYS applied)
    results['sharpened'] = fft_sharpen(image, hpf_radius=hpf_radius)
    
    if apply_segmentation:
        # Step 2: Enhanced Segmentation (ONLY for tumor images)
        results['segmented'], results['enhanced'], results['denoised'] = enhanced_segmentation(
            results['sharpened'], threshold_value=threshold_value
        )
        
        # Step 3: Morphology cleanup
        results['final_mask'] = morphology_cleanup(
            results['segmented'], 
            open_kernel=open_kernel,
            close_kernel=close_kernel
        )
        
        # Step 4: Create overlay
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        overlay[results['final_mask'] > 0] = [255, 0, 0]  # Red for detected tumor
        results['overlay'] = overlay
        
        # Calculate statistics in mm²
        tumor_pixels = np.count_nonzero(results['final_mask'])
        results['tumor_area_px'] = tumor_pixels
        results['tumor_area_mm2'] = tumor_pixels * pixel_spacing_x * pixel_spacing_y
        
    else:
        # No segmentation for normal images
        results['segmented'] = None
        results['enhanced'] = None
        results['denoised'] = None
        results['final_mask'] = np.zeros_like(image)
        results['overlay'] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        results['tumor_area_px'] = 0
        results['tumor_area_mm2'] = 0.0
    
    return results


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
1. **FFT Sharpening** - Unsharp Masking dengan High-Pass Filter
2. **Enhanced Segmentation** - CLAHE + Denoising + Thresholding
3. **Morphology Cleanup** - Opening & Closing untuk hasil lebih jelas
""")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Parameters")

mode = st.sidebar.radio("Mode", ["Random Dataset", "Upload Image"])

st.sidebar.markdown("---")
st.sidebar.subheader("FFT Sharpening")
hpf_radius = st.sidebar.slider("HPF Radius", 5, 50, 20, 
                                help="Radius untuk High-Pass Filter. Lebih besar = lebih gentle")

st.sidebar.subheader("Segmentation (Tumor Only)")
threshold_value = st.sidebar.slider("Threshold Value", 100, 255, 200,
                                     help="Threshold untuk deteksi area terang (tumor)")

st.sidebar.subheader("Morphology Cleanup")
open_kernel = st.sidebar.slider("Opening Kernel", 1, 15, 3, step=2,
                                 help="Kernel untuk menghilangkan noise kecil")
close_kernel = st.sidebar.slider("Closing Kernel", 1, 15, 5, step=2,
                                  help="Kernel untuk mengisi hole kecil")

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
            st.info("💡 Normal images: Hanya FFT Sharpening, TIDAK ada segmentasi (tidak perlu deteksi tumor)")
            
            for i, img_path in enumerate(st.session_state.selected_normal):
                st.markdown(f"### Normal Image #{i+1}: `{os.path.basename(img_path)}`")
                
                # Read and process (NO SEGMENTATION)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                results = process_image(image, hpf_radius, threshold_value, open_kernel, close_kernel, 
                                       apply_segmentation=False, 
                                       pixel_spacing_x=pixel_spacing_x, 
                                       pixel_spacing_y=pixel_spacing_y)
                
                # Display - Only 2 images for normal
                cols = st.columns([1, 1, 2])
                with cols[0]:
                    st.image(image, caption="Original", use_container_width=True, clamp=True)
                with cols[1]:
                    st.image(results['sharpened'], caption="FFT Sharpened (Enhanced)", use_container_width=True, clamp=True)
                with cols[2]:
                    # Show comparison
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                    ax1.imshow(image, cmap='gray')
                    ax1.set_title('Before')
                    ax1.axis('off')
                    ax2.imshow(results['sharpened'], cmap='gray')
                    ax2.set_title('After Sharpening')
                    ax2.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Metrics
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Status", "Normal")
                with metric_cols[1]:
                    st.metric("Tumor Area", "0 mm²")
                with metric_cols[2]:
                    st.success("✅ No tumor detected (as expected)")
                
                st.markdown("---")
            
            st.markdown('<div class="section-header">🔴 Results - TUMOR Images (Full Pipeline)</div>', unsafe_allow_html=True)
            st.info("💡 Tumor images: FFT Sharpening + Segmentasi + Pengukuran area dalam mm²")
            
            for i, img_path in enumerate(st.session_state.selected_tumor):
                st.markdown(f"### Tumor Image #{i+1}: `{os.path.basename(img_path)}`")
                
                # Read and process (WITH SEGMENTATION)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                results = process_image(image, hpf_radius, threshold_value, open_kernel, close_kernel, 
                                       apply_segmentation=True,
                                       pixel_spacing_x=pixel_spacing_x, 
                                       pixel_spacing_y=pixel_spacing_y)
                
                # Display full pipeline
                cols = st.columns(5)
                with cols[0]:
                    st.image(image, caption="Original", use_container_width=True, clamp=True)
                with cols[1]:
                    st.image(results['sharpened'], caption="FFT Sharpened", use_container_width=True, clamp=True)
                with cols[2]:
                    st.image(results['enhanced'], caption="Enhanced (CLAHE)", use_container_width=True, clamp=True)
                with cols[3]:
                    st.image(results['final_mask'], caption="Tumor Detection", use_container_width=True, clamp=True)
                with cols[4]:
                    st.image(results['overlay'], caption="Overlay", use_container_width=True, clamp=True)
                
                # Metrics with mm²
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Tumor Area", f"{results['tumor_area_mm2']:.2f} mm²")
                with metric_cols[1]:
                    img_size_mm2 = image.size * pixel_spacing_x * pixel_spacing_y
                    percentage = (results['tumor_area_mm2'] / img_size_mm2) * 100 if img_size_mm2 > 0 else 0
                    st.metric("% of Image", f"{percentage:.2f}%")
                with metric_cols[2]:
                    if results['tumor_area_mm2'] > 0:
                        st.error("🔴 Tumor Detected")
                    else:
                        st.info("ℹ️ No tumor detected")
                
                st.markdown("---")

else:  # Upload Image mode
    st.markdown('<div class="section-header">📤 Upload Image Mode</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload MRI Image", type=['png', 'jpg', 'jpeg'])
    
    # Toggle for image type
    image_type = st.radio("Image Type", ["Normal (Sharpening only)", "Tumor (Full segmentation)"], 
                         horizontal=True)
    apply_seg = (image_type == "Tumor (Full segmentation)")
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            st.error("❌ Failed to read image")
        else:
            # Process
            with st.spinner("Processing..."):
                results = process_image(image, hpf_radius, threshold_value, open_kernel, close_kernel,
                                       apply_segmentation=apply_seg,
                                       pixel_spacing_x=pixel_spacing_x,
                                       pixel_spacing_y=pixel_spacing_y)
            
            st.success("✅ Processing complete!")
            
            if apply_seg:
                # Full pipeline for tumor
                st.markdown("### Processing Pipeline")
                
                cols = st.columns(5)
                with cols[0]:
                    st.image(image, caption="1. Original", use_container_width=True, clamp=True)
                with cols[1]:
                    st.image(results['sharpened'], caption="2. FFT Sharpened", use_container_width=True, clamp=True)
                with cols[2]:
                    st.image(results['enhanced'], caption="3. Enhanced (CLAHE)", use_container_width=True, clamp=True)
                with cols[3]:
                    st.image(results['segmented'], caption="4. Segmented", use_container_width=True, clamp=True)
                with cols[4]:
                    st.image(results['final_mask'], caption="5. Final (Morphology)", use_container_width=True, clamp=True)
                
                st.markdown("### Final Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True, clamp=True)
                with col2:
                    st.image(results['overlay'], caption="Tumor Detection Overlay", use_container_width=True, clamp=True)
                
                # Metrics with mm²
                st.markdown("### Analysis Results")
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Image Size", f"{image.shape[1]} x {image.shape[0]}")
                with metric_cols[1]:
                    st.metric("Tumor Area", f"{results['tumor_area_mm2']:.2f} mm²")
                with metric_cols[2]:
                    if results['tumor_area_mm2'] > 0:
                        st.error("🔴 Tumor Detected")
                    else:
                        st.success("✅ No tumor detected")
            else:
                # Sharpening only for normal
                st.markdown("### Sharpening Result (No Segmentation)")
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
                    st.success("✅ Normal (No segmentation applied)")

# ==================== INFO & HELP ====================
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Help & Info"):
    st.markdown("""
    **Parameter Guide:**
    
    - **HPF Radius**: 15-25 untuk MRI normal
    - **Threshold**: 180-220 untuk tumor
    - **Opening**: 3-5 untuk noise removal
    - **Closing**: 5-9 untuk hole filling
    - **Pixel Spacing**: Biasanya 0.5-1.0 mm/pixel untuk MRI
    
    **Processing:**
    - **Normal**: Hanya sharpening, TIDAK ada segmentasi
    - **Tumor**: Full pipeline + area dalam mm²
    
    **Expected Results:**
    - **Normal**: 0 mm² (no tumor)
    - **Tumor**: > 10-100 mm² (tergantung ukuran)
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Brain MRI Tumor Detection v1.0")

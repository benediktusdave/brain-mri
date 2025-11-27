import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
import pandas as pd
import matplotlib.pyplot as plt

from modules.preprocessing import to_gray, denoise_nlmeans, denoise_bilateral, apply_clahe
from modules.fft_sharpen import fft_sharpen_generic
from modules.segmentation import segment_image, measure_components, overlay_mask_on_image, annotate_image
from modules.utils import plot_spectrum


# Page config
st.set_page_config(page_title="MRI FFT Sharpen & Seg", layout='wide')


# CSS small
st.markdown("""
<style>
.section-header { font-size: 1.1rem; color: #2e86ab; }
.metric-box { background-color: #e8f4f8; padding: 0.4rem; border-radius:5px; }
</style>
""", unsafe_allow_html=True)


st.title('🧠 MRI FFT Sharpen & Segmentation (Modular)')


# Sidebar: mode specific UI (show only relevant parameters)
st.sidebar.header('1) Preprocessing')
denoise_method = st.sidebar.selectbox('Denoise method', ['None','NLMeans','Bilateral'])
denoise_strength = st.sidebar.slider('Denoise strength (h or d param)', 0, 30, 10)


st.sidebar.header('2) FFT Sharpening')
mask_type = st.sidebar.selectbox('HPF Mask type', ['gaussian','butterworth','ideal'])
cutoff = st.sidebar.slider('Cutoff / Sigma', 5, 150, 30)
butter_order = st.sidebar.slider('Butterworth order (if used)', 1, 6, 2)
boost = st.sidebar.slider('Boost factor', 0.0, 5.0, 1.5, 0.1)


st.sidebar.header('3) Optional Enhancements')
use_unsharp = st.sidebar.checkbox('Also apply Unsharp Mask (spatial)', True)
unsharp_sigma = st.sidebar.slider('Unsharp sigma', 0.5, 3.0, 1.0, 0.1)
unsharp_strength = st.sidebar.slider('Unsharp strength', 0.1, 3.0, 1.0, 0.1)
use_clahe = st.sidebar.checkbox('Apply CLAHE for display (will not affect segmentation)', value=False)


st.sidebar.header('4) Segmentation & Measurement')
do_seg = st.sidebar.checkbox('Enable segmentation & measurement', True)
seg_method = st.sidebar.selectbox('Segmentation method', ['otsu','adaptive','threshold'])
adaptive_bs = st.sidebar.slider('Adaptive block size (odd)', 11, 101, 35, step=2)
adaptive_C = st.sidebar.slider('Adaptive C', 0, 20, 5)
morph_open = st.sidebar.slider('Morph open kernel', 1, 15, 3)
morph_close = st.sidebar.slider('Morph close kernel', 1, 31, 5)
min_area_px = st.sidebar.slider('Min component area (px)', 10, 2000, 50)


st.sidebar.header('5) Pixel spacing for area calc (mm/pixel)')
psx = st.sidebar.number_input('Pixel spacing X (mm)', min_value=0.01, value=1.0, format='%.3f')
psy = st.sidebar.number_input('Pixel spacing Y (mm)', min_value=0.01, value=1.0, format='%.3f')


# Helpful hints (contextual) - dynamic based on param selection
st.sidebar.markdown('---')
st.sidebar.subheader('Hints (quick)')
if mask_type == 'gaussian':
    st.sidebar.write('Gaussian HPF: smooth transition, less ringing. Lower sigma = stronger sharpening on low freqs.')
elif mask_type == 'butterworth':
    st.sidebar.write('Butterworth: smoother than ideal; order increases sharpness of transition (higher order -> more ringing).')
else:
    st.sidebar.write('Ideal HPF: hard cutoff -> potential ringing artifacts.')

st.sidebar.write('Boost factor >1 increases high-frequency magnitude (sharper edges).')
st.sidebar.write('If segmentation fails after display adjustments, lower boost or enable denoising.')

# File upload
st.markdown('### Upload MRI (PNG/JPG/TIFF)')
uploaded = st.file_uploader('Upload file', type=['png','jpg','jpeg','tif','tiff'])


if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if original is None:
        st.error('Could not read image. If DICOM, extract pixel array first.')
    else:
        img_gray = to_gray(original)
        # preprocessing
        if denoise_method == 'NLMeans':
            img_dn = denoise_nlmeans(img_gray, h=denoise_strength)
        elif denoise_method == 'Bilateral':
            d = int(max(1, denoise_strength//2*2+1))
            img_dn = denoise_bilateral(img_gray, d=d)
        else:
            img_dn = img_gray.copy()


        # FFT sharpen (this is primary output like your second picture)
        processed_fft = fft_sharpen_generic(img_dn, mask_type=mask_type, cutoff=cutoff, order=butter_order, boost=boost)


        # segmentation SHOULD be performed on processed_fft (before CLAHE)
        if do_seg:
            mask = segment_image(processed_fft, method=seg_method, adaptive_blocksize=adaptive_bs,
                                adaptive_C=adaptive_C, morph_open=morph_open, morph_close=morph_close,
                                min_area_px=min_area_px)
            components = measure_components(mask, pixel_spacing_x=psx, pixel_spacing_y=psy)
            overlay = overlay_mask_on_image(processed_fft, mask, color=(255,0,0), alpha=0.4)
            annotated = annotate_image(overlay, components)
        else:
            mask = np.zeros_like(processed_fft)
            components = []
            annotated = cv2.cvtColor(processed_fft, cv2.COLOR_GRAY2RGB)

        # apply unsharp optionally
        if use_unsharp:
            blurred = cv2.GaussianBlur(processed_fft, (0,0), unsharp_sigma)
            processed_unsharp = np.clip((1+unsharp_strength)*processed_fft.astype(np.float32) - unsharp_strength*blurred.astype(np.float32), 0, 255).astype(np.uint8)
        else:
            processed_unsharp = processed_fft.copy()


        # CLAHE only for display (do not use for segmentation) - if user checked
        if use_clahe:
            processed_display = apply_clahe(processed_unsharp)
        else:
            processed_display = processed_unsharp.copy()


        # metrics
        orig_sharp = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        proc_sharp = cv2.Laplacian(processed_unsharp, cv2.CV_64F).var()
        improvement = ((proc_sharp - orig_sharp)/(orig_sharp+1e-9))*100.0


        # layout
        c1,c2,c3,c4 = st.columns([1,1,1,1])
        c1.metric('Original shape', f'{img_gray.shape[1]}x{img_gray.shape[0]}')
        c2.metric('Sharpness original', f'{orig_sharp:.2f}')
        c3.metric('Sharpness processed', f'{proc_sharp:.2f}')
        c4.metric('Improvement %', f'{improvement:+.1f}%')


        # show images similar to your second picture: Original | Processed (FFT HPF)
        st.markdown('## Hasil Perbandingan')
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].imshow(img_gray, cmap='gray'); ax[0].set_title('Gambar MRI Original'); ax[0].axis('off')
        ax[1].imshow(processed_fft, cmap='gray'); ax[1].set_title('Hasil High-Pass Filter'); ax[1].axis('off')
        st.pyplot(fig)


        # difference map and spectrum and annotated overlay
        st.markdown('## Peta Perubahan, Spektrum & Segmen')
        d1, d2, d3 = st.columns([1,1,1])
        diff = cv2.absdiff(img_gray, processed_unsharp)
        with d1:
            st.caption('Difference (absdiff)')
            figd, axd = plt.subplots(figsize=(4,3)); axd.imshow(diff, cmap='hot'); axd.axis('off'); st.pyplot(figd)
        with d2:
            st.caption('Spectrum processed')
            st.pyplot(plot_spectrum(processed_unsharp))
        with d3:
            st.caption('Segmentation overlay & measurements')
            st.image(annotated, use_column_width=True)
        if len(components) == 0:
            st.info('No components found with current segmentation parameters.')
        else:
            df = pd.DataFrame([{'label':c['label'],'area_px':c['area_px'],'area_mm2':round(c['area_mm2'],2),'centroid_x':round(c['centroid'][0],2),'centroid_y':round(c['centroid'][1],2)} for c in components])
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download measurements CSV', csv, file_name='measurements.csv', mime='text/csv')
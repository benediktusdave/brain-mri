# Top-hat Filtering untuk Brain Tumor Detection

## Apa itu Top-hat Filtering?

Top-hat filtering adalah operasi morfologi yang digunakan untuk **meningkatkan kontras struktur kecil yang terang (bright) pada background gelap (dark)**.

### Formula
```
Top-hat Transform = Original Image - Opening(Original Image)
```

Where:
- **Opening** = Erosion followed by Dilation

## Kenapa Digunakan untuk Tumor Detection?

1. **Tumor MRI biasanya muncul sebagai area terang** pada background otak yang lebih gelap
2. **Top-hat filtering mengisolasi struktur kecil yang terang** dengan cara:
   - Opening menghilangkan small bright regions
   - Subtract dari original → hanya bright regions yang tersisa
3. **Meningkatkan akurasi Watershed segmentation** dengan membuat tumor lebih jelas

## Pipeline di Aplikasi Ini

Untuk **Watershed Method**, top-hat filtering digunakan sebagai preprocessing:

```
Input Image 
  ↓
FFT Sharpening (enhance edges)
  ↓
Top-hat Filtering (isolate bright tumor regions) ← ADDED
  ↓
CLAHE (contrast enhancement)
  ↓
Denoising (reduce noise)
  ↓
Watershed Segmentation (multi-region detection)
  ↓
Colored Visualization
```

## Parameter

### `tophat_kernel` (kernel size)
- **Default:** 15
- **Range:** 5-31 (odd numbers)
- **Effect:**
  - **Smaller kernel (5-11):** Detect smaller bright structures, more sensitive to small tumors
  - **Medium kernel (13-19):** Balanced, detect medium-sized tumors
  - **Larger kernel (21-31):** Only detect larger bright structures, ignore small bright spots

**Tips:** 
- Jika tumor detection terlalu sensitif (banyak false positives), **tingkatkan kernel size**
- Jika tumor kecil tidak terdeteksi, **turunkan kernel size**

## Hasil yang Diharapkan

### Sebelum Top-hat Filtering
- Tumor kurang jelas
- Background noise masih ada
- Watershed bisa salah detect area non-tumor

### Setelah Top-hat Filtering
- ✅ Tumor regions lebih terang dan jelas
- ✅ Background menjadi lebih uniform (gelap)
- ✅ Watershed lebih akurat dalam mendeteksi tumor boundaries
- ✅ Reduced false positives

## Research Reference

Implementasi ini mengikuti metodologi dari paper:
**"Brain Tumor Detection using Image Processing"** by Mlachahe Said Salimo
- Published: March 2024
- Medium: https://medium.com/wanabilini/brain-tumor-detection-using-image-processing-a26b1c927d5d

Pipeline yang digunakan dalam paper:
1. Anisotropic Diffusion Filter (ADF)
2. Skull Stripping
3. **Top-hat Filtering** ← Focus of this implementation
4. Histogram Equalization (HE)
5. Binarization
6. Watershed Segmentation
7. Morphological Operations

## Kode Implementation

```python
def tophat_filtering(image, kernel_size=15):
    """
    Apply Top-hat filtering to enhance bright structures (tumors) on dark background
    
    Top-hat transform = Original - Opening
    This highlights small bright regions that are smaller than the structuring element
    """
    # Create structuring element (disk/ellipse shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply morphological opening
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Top-hat transform: Original - Opening
    tophat = cv2.subtract(image, opened)
    
    return tophat, opened
```

## Visualisasi di Streamlit

Aplikasi menampilkan 5 tahapan processing untuk Watershed method:
1. **Original** - Input MRI image
2. **FFT Sharpened** - Edge enhancement
3. **🎩 Top-hat Filtered** - Isolated bright tumor regions (NEW!)
4. **Enhanced (CLAHE)** - Improved contrast
5. **🌈 Watershed Result** - Multi-region colored segmentation

---

**Last Updated:** December 1, 2025
**Author:** Brain MRI Processing Pipeline

# Brain MRI Tumor Detection - Classical Image Processing

Aplikasi web (Streamlit) untuk deteksi tumor pada gambar MRI otak menggunakan **Classical Image Processing** (tanpa AI/ML).

## 🎯 Key Features

- ✅ **Automatic Tumor Detection** - Upload gambar, sistem otomatis deteksi tumor/normal
- ✅ **Smart Pipeline** - Auto-detect apakah perlu segmentasi atau hanya sharpening
- ✅ **Watershed Segmentation** - Multi-region tumor detection dengan geometric filtering
- ✅ **Color-Coded Visualization** - Setiap region tumor ditandai dengan warna berbeda (tanpa label angka)
- ✅ **Area & Radius Measurement** - Perhitungan dalam mm² dan mm (dengan pixel spacing)
- ✅ **Interactive Parameters** via sidebar controls
- ✅ **Two Modes:** Random Dataset atau Upload Image

## 🔬 Processing Pipeline

### 🟢 Normal Images (Auto-detected)
**Quick detection:** Brightness analysis (< 1% bright pixels)
1. **FFT Sharpening Only** - Enhanced visualization

### 🔴 Tumor Images (Auto-detected)
**Quick detection:** Brightness analysis (> 1% bright pixels)

**Watershed Method (Multi-region):**
1. **FFT Sharpening** - Unsharp Masking dengan High-Pass Filter
2. **Skull Stripping** - Remove tengkorak, fokus ke otak
3. **Top-hat Filtering** - Enhance bright structures (tumor)
4. **Histogram Equalization** - Enhance contrast
5. **Hybrid Binarization** - Brightness threshold + Top-hat texture
6. **Aggressive Morphology** - Multiple dilation & erosion
7. **Watershed Segmentation** - Separate overlapping regions
8. **Geometric Filtering** - Filter by Solidity (> 0.6) untuk buang false positive
9. **Color Visualization** - Each tumor marked with distinct colors

**Research Reference:** [Brain Tumor Detection using Image Processing](https://medium.com/wanabilini/brain-tumor-detection-using-image-processing-a26b1c927d5d) by Mlachahe Said Salimo

## Setup

### 1. Aktifkan Virtual Environment

```powershell
# Windows PowerShell
cd "d:\a_kuliah\SEMESTER 5\Image Processing\uas\brain_sharpening"
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run Streamlit App

```powershell
streamlit run app.py
```

Browser akan otomatis membuka di `http://localhost:8501`

## 🎮 How to Use

### Mode 1: 🎲 Random Dataset
- Pilih "Random Dataset" di sidebar
- Sistem load random Normal + Tumor images
- View side-by-side comparison
- See complete processing pipeline

### Mode 2: 📤 Upload Image
- Pilih "Upload Image" di sidebar
- Upload file MRI (PNG, JPG, JPEG)
- **Sistem otomatis mendeteksi:**
  - 🟢 **Normal** (< 1% bright pixels) → Sharpening only
  - 🔴 **Tumor** (> 1% bright pixels) → Full watershed segmentation
- No need to manually select image type!

## 📊 Output Information

### For Tumor Images:
- **Processing Pipeline** - 6 tahap visualisasi
- **Final Result** - Original vs Tumor Detection
- **Metrics:**
  - Total Tumor Area (mm²)
  - Number of Regions
  - Percentage of Image
- **Colored Table** - Individual tumor regions dengan background color yang sama dengan gambar

### For Normal Images:
- Original vs FFT Sharpened
- Simple metrics

## 🎨 Color-Coded Regions

Setiap region tumor ditandai dengan warna berbeda:
- 🟢 Lime Green
- 🟣 Magenta
- 🔵 Cyan
- 🟠 Orange
- 🔴 Red
- 🔵 Blue
- 🟡 Yellow
- 🟣 Purple
- 🟢 Spring Green
- 🩷 Deep Pink

**Tabel menampilkan warna yang sama dengan region di gambar** untuk memudahkan identifikasi.

## 📏 Area Calculation (Pixel → mm²)

### Formula:
```python
# Area dalam pixel
area_pixels = cv2.contourArea(contour)

# Konversi ke mm² menggunakan pixel spacing
area_mm2 = area_pixels × pixel_spacing_x × pixel_spacing_y

# Hitung radius (anggap tumor = lingkaran)
radius_mm = sqrt(area_mm2 / π)
diameter_mm = 2 × radius_mm
```

### Pixel Spacing Default:
- `pixel_spacing_x = 1.0 mm/pixel`
- `pixel_spacing_y = 1.0 mm/pixel`

**Untuk MRI DICOM:** Pixel spacing biasanya ada di metadata (0.4-0.9 mm/pixel)

### Contoh Perhitungan:
```
Area: 500 pixels
Pixel spacing: 1.0 mm/pixel

Area (mm²) = 500 × 1.0 × 1.0 = 500 mm²
Radius = sqrt(500 / π) = 12.62 mm
Diameter = 25.24 mm
```

## Struktur Dataset

Dataset harus berada di lokasi relatif:

```
../dataset/Brain MRI Images/Train/
├── Normal/     # Gambar MRI otak sehat
└── Tumor/      # Gambar MRI otak dengan tumor
```

## ⚙️ Adjustable Parameters (Sidebar)

### FFT Sharpening
```python
HPF_RADIUS = 20           # 5-50: Radius High-Pass Filter untuk frequency domain
```

### Watershed Segmentation (Tumor Only)
```python
BRIGHTNESS_THRESHOLD = 180    # 100-255: Threshold untuk deteksi area terang
TOPHAT_KERNEL_SIZE = 50      # 5-150: Kernel size untuk top-hat filtering
WATERSHED_SENSITIVITY = 0.4   # 0.1-1.0: Distance transform threshold
MIN_TUMOR_AREA = 300         # 100-2000: Minimum area (pixels) untuk filter noise
```

### Pixel Spacing (for mm² calculation)
```python
PIXEL_SPACING_X = 1.0        # mm per pixel (horizontal)
PIXEL_SPACING_Y = 1.0        # mm per pixel (vertical)
```

## 🧪 Technical Details

### Auto-Detection Algorithm
```python
def quick_tumor_detection(image, threshold=180):
    # Hitung % pixel terang
    _, bright_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    bright_ratio = (bright_pixels / total_pixels) * 100
    
    # Kriteria: > 1% = Tumor
    is_tumor = bright_ratio > 1.0
    return is_tumor, bright_ratio
```

### Geometric Filtering (Anti False Positive)
```python
# Filter berdasarkan Solidity
solidity = contour_area / convex_hull_area

# Kriteria: Solidity > 0.6
# Tumor = padat (high solidity)
# Lipatan otak = berongga (low solidity)
```

### Watershed Markers
- **Foreground markers:** Distance transform + threshold
- **Background markers:** Dilation dari foreground
- **Unknown region:** Watershed decision boundary

## 📈 Expected Results

### Normal Images:
- **Bright pixels:** < 0.5-1.0%
- **Action:** FFT Sharpening only
- **Tumor area:** 0 mm²

### Tumor Images:
- **Bright pixels:** > 1-5%
- **Action:** Full watershed segmentation
- **Tumor area:** Varies (hundreds to thousands of mm²)
- **Regions:** Multiple colored regions with geometric filtering

## 🎨 Visualization Examples

### Processing Steps (Tumor):
1. **Original** - Raw MRI grayscale
2. **FFT Sharpened** - Enhanced edges & details
3. **Brain Only** - Skull stripped
4. **Top-hat** - Bright structures enhanced
5. **Hybrid Binary** - Combined threshold masks
6. **Watershed + Filter** - Color-coded tumor regions

### Final Output:
- Each tumor region = **Unique color**
- Small circle at centroid
- Black boundaries between regions
- No text labels (cleaner visualization)

## 📋 Table Output

| Region Color | Area (mm²) | Area (px) | Centroid |
|-------------|-----------|----------|----------|
| Lime Green  | 1462.00   | 1462     | (38, 164)|
| Magenta     | 1428.00   | 1428     | (98, 69) |
| Cyan        | 1282.00   | 1282     | (206, 140)|

**Note:** Setiap baris memiliki background color yang sama dengan region di gambar!

## 🚀 Quick Start

```powershell
# 1. Activate venv
.\venv\Scripts\Activate.ps1

# 2. Run app
streamlit run app.py

# 3. Upload MRI image or use Random Dataset
# 4. System auto-detects tumor/normal
# 5. View results!
```

## 🛠️ Troubleshooting

### Issue: "No module named 'cv2'"
```powershell
pip install opencv-python
```

### Issue: "Dataset folder not found"
Pastikan struktur folder:
```
uas/
├── brain_sharpening/
│   └── app.py
└── dataset/
    └── Brain MRI Images/Train/Normal & Tumor/
```

### Issue: "Too many regions detected"
Adjust parameters:
- Increase `MIN_TUMOR_AREA` (filter small noise)
- Decrease `WATERSHED_SENSITIVITY` (less aggressive splitting)
- Increase `BRIGHTNESS_THRESHOLD` (stricter detection)

## 📚 References

1. [Brain Tumor Detection using Image Processing](https://medium.com/wanabilini/brain-tumor-detection-using-image-processing-a26b1c927d5d)
2. OpenCV Documentation - Watershed Algorithm
3. Scikit-image Morphology Operations

## 🔄 Updates

### Latest Changes:
- ✅ **Auto-detection** - No manual tumor/normal selection
- ✅ **Color-coded regions** - No text labels, cleaner visualization
- ✅ **Geometric filtering** - Solidity > 0.6 (reduce false positives)
- ✅ **Improved color scheme** - High contrast colors (not white)
- ✅ **Table styling** - Background colors match image regions

## Deactivate venv

```powershell
deactivate
```

## 👨‍💻 Author

Image Processing Project - Brain MRI Tumor Detection
Classical Computer Vision Approach (No AI/ML)

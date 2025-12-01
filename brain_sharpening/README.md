# Brain MRI Tumor Detection - Classical Image Processing

Aplikasi web (Streamlit) untuk deteksi tumor pada gambar MRI otak menggunakan **Classical Image Processing** (tanpa AI/ML).

## Pipeline

1. **FFT Sharpening** - Unsharp Masking dengan High-Pass Filter
2. **Enhanced Segmentation** - CLAHE + Denoising + Thresholding untuk hasil lebih jelas
3. **Morphology Cleanup** - Opening & Closing untuk menghilangkan noise dan fill holes

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

## Features

### 🎲 Random Dataset Mode
- Load random images dari folder Normal dan Tumor
- Compare hasil detection secara side-by-side
- Lihat semua step processing pipeline

### 📤 Upload Image Mode
- Upload gambar MRI sendiri
- Real-time processing dengan adjustable parameters
- Visualisasi lengkap setiap tahap

## Struktur Dataset

Dataset harus berada di lokasi relatif:

```
../dataset/Brain MRI Images/Train/
├── Normal/     # Gambar MRI otak sehat
└── Tumor/      # Gambar MRI otak dengan tumor
```

## Parameters (Adjustable via Sidebar)

```python
HPF_RADIUS = 20           # 5-50: Radius High-Pass Filter
THRESHOLD_VALUE = 200     # 100-255: Threshold untuk segmentasi
OPEN_KERNEL = 3          # 1-15: Kernel untuk remove noise
CLOSE_KERNEL = 5         # 1-15: Kernel untuk fill holes
```

## Enhanced Segmentation

Dibanding versi sebelumnya, sekarang menggunakan:
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - Meningkatkan kontras lokal
2. **Median Blur** - Mengurangi noise sebelum thresholding
3. **Advanced Morphology** - Opening + Closing dengan kernel terpisah

Hasil segmentasi **lebih jelas** dan **lebih akurat**!

## Expected Results

- **Normal images**: Tumor Detection < 1% (mostly black)
- **Tumor images**: Tumor Detection > 1-5% (white tumor regions visible)

## Screenshot

Aplikasi menampilkan:
- Original image
- FFT Sharpened
- Enhanced (CLAHE)
- Tumor Detection mask
- Overlay visualization
- Metrics: Area & Percentage

## Deactivate venv

```powershell
deactivate
```

## Author

[Your Name]

# =========================

# File: README.md

# =========================

# MRI FFT Sharpen & Segmentation

#

# Instructions:

# 1. Create project folder and place files as shown in layout.

# 2. (Optional) create virtual environment: python -m venv venv && source venv/bin/activate

# 3. pip install -r requirements.txt

# 4. Run: streamlit run main.py

#

# Notes:

# - Segmentation is performed on processed FFT image before CLAHE (so CLAHE will not break segmentation).

# - You can try mask_type = 'ideal' to reproduce the hard-cutoff output similar to original example (may show ringing).

# - Provide pixel spacing (mm/pixel) manually if you want area in mm^2.

# - To add additional algorithms: implement in modules/ and call from main.py.

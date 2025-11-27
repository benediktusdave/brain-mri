from scipy import fftpack
import numpy as np
from .preprocessing import normalize_uint8

def gaussian_hpf_mask(shape, cutoff_sigma):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    y = np.linspace(-crow, crow, rows)
    x = np.linspace(-ccol, ccol, cols)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = 1 - np.exp(-(D**2)/(2*(cutoff_sigma**2)))
    return H


def butterworth_hpf_mask(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    D = np.sqrt(u**2 + v**2)
    H = 1 / (1 + (cutoff/(D + 1e-9))**(2*order))
    return H


def ideal_hpf_mask(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
    H = np.ones(shape, dtype=np.float32)
    H[mask_area] = 0
    return H




def fft_sharpen_generic(img: np.ndarray, mask_type: str = 'gaussian', cutoff: float = 30.0,
order: int = 2, boost: float = 1.0) -> np.ndarray:
    imgf = img.astype(np.float32) / 255.0
    F = fftpack.fft2(imgf)
    Fshift = fftpack.fftshift(F)
    if mask_type == 'gaussian':
        H = gaussian_hpf_mask(imgf.shape, cutoff)
    elif mask_type == 'butterworth':
        H = butterworth_hpf_mask(imgf.shape, cutoff, order)
    else:
        H = ideal_hpf_mask(imgf.shape, cutoff)
    Fboost = Fshift * (1.0 + boost * H)
    F_ishift = fftpack.ifftshift(Fboost)
    img_back = np.real(fftpack.ifft2(F_ishift))
    return normalize_uint8(img_back)
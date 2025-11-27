import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def plot_spectrum(img: np.ndarray):
    f = fftpack.fft2(img.astype(np.float32)/255.0)
    fshift = fftpack.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift) + 1e-9)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(magnitude_spectrum, cmap='viridis')
    ax.axis('off')
    return fig


def normalize_uint8(imgf: np.ndarray) -> np.ndarray:
    a = imgf.astype(np.float32)
    a = a - a.min()
    if a.max() != 0:
        a = a / a.max()
    return (a * 255.0).astype(np.uint8)
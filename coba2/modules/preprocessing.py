from typing import Tuple
import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def normalize_uint8(imgf: np.ndarray) -> np.ndarray:
    a = imgf.astype(np.float32)
    a = a - a.min()
    if a.max() != 0:
        a = a / a.max()
    return (a * 255.0).astype(np.uint8)

def denoise_nlmeans(img: np.ndarray, h: int = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)

def denoise_bilateral(img: np.ndarray, d: int = 9, sigmaColor: int = 75, sigmaSpace: int = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def apply_clahe(img: np.ndarray, clipLimit: float = 2.0, tileGridSize: Tuple[int,int] = (8,8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)
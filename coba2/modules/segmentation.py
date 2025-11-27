import cv2
import numpy as np
from typing import List, Dict, Tuple

def segment_image(img: np.ndarray, method: str = 'otsu', adaptive_blocksize: int = 35, adaptive_C: int = 5,
morph_open: int = 3, morph_close: int = 5, min_area_px: int = 50) -> np.ndarray:
    gray = img.copy()
    if method == 'adaptive':
        bs = adaptive_blocksize if adaptive_blocksize % 2 == 1 else adaptive_blocksize + 1
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, bs, adaptive_C)
    elif method == 'threshold':
        th = int(np.mean(gray) + np.std(gray)*0.5)
        _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # invert if foreground too large
    if np.sum(mask==255) > 0.5 * mask.size:
        mask = cv2.bitwise_not(mask)
    
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

def measure_components(mask: np.ndarray, pixel_spacing_x: float = 1.0, pixel_spacing_y: float = 1.0) -> List[Dict]:
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


def overlay_mask_on_image(img_gray: np.ndarray, mask: np.ndarray, color: Tuple[int,int,int]=(255,0,0), alpha: float=0.4) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    overlay[mask==255] = color
    out = cv2.addWeighted(overlay, alpha, img_rgb, 1-alpha, 0)
    return out


def annotate_image(img_rgb: np.ndarray, components: List[Dict]) -> np.ndarray:
    out = img_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for comp in components:
        x,y,w,h = comp['bbox']
        area_px = comp['area_px']
        area_mm2 = comp['area_mm2']
        centroid = (int(comp['centroid'][0]), int(comp['centroid'][1]))
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        text = f"{area_px} px / {area_mm2:.1f} mm2"
        cv2.putText(out, text, (x, max(10,y-5)), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.circle(out, centroid, 3, (0,255,255), -1)
    return out
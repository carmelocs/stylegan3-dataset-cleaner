"""
quality.py
-----------
Image quality checks: sharpness, brightness, saturation, exposure.
"""

import cv2
import numpy as np


def variance_of_laplacian(img_bgr):
    """Sharpness metric: variance of Laplacian of grayscale."""
    return cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def brightness(img_bgr):
    """Mean brightness of grayscale image."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean()


def saturation(img_bgr):
    """Mean saturation in HSV."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv[..., 1].mean() / 255.0


def passes_quality(img_bgr, min_sharp=120, min_brightness=90, max_brightness=180, min_sat=0.12, max_sat=0.55):
    """Return True if image passes all thresholds."""
    sharp = variance_of_laplacian(img_bgr)
    bright = brightness(img_bgr)
    sat = saturation(img_bgr)

    if sharp < min_sharp:
        return False, "low_sharpness"
    if not (min_brightness <= bright <= max_brightness):
        return False, "bad_exposure"
    if not (min_sat <= sat <= max_sat):
        return False, "bad_saturation"

    return True, None

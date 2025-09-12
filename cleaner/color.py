"""
color.py
---------
Color normalization: match LAB mean/std to a reference image.
"""

import cv2
import numpy as np


def lab_match(src_bgr, ref_bgr):
    """Normalize LAB mean/std of src to match reference image."""
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src_lab.copy()

    for c in range(3):
        s_mu, s_std = src_lab[..., c].mean(), src_lab[..., c].std() + 1e-6
        r_mu, r_std = ref_lab[..., c].mean(), ref_lab[..., c].std() + 1e-6
        out[..., c] = (out[..., c] - s_mu) / s_std * r_std + r_mu

    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

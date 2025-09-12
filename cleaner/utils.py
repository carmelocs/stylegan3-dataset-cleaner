"""
utils.py
---------
General utilities: I/O, saving, manifests, logging.
"""

import os
import cv2
import hashlib
from pathlib import Path


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_png(path, img_bgr):
    """Save image as compressed PNG."""
    cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def unique_filename(src_path):
    """Deterministic filename from md5 of path string."""
    md5 = hashlib.md5(str(src_path).encode()).hexdigest()[:10]
    return f"{md5}.png"

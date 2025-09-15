"""
pipeline.py
------------
Main orchestration of dataset cleaning:
1. Load → align → quality check
2. Color normalization
3. Deduplication
4. Save outputs and manifest
"""

import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .align import FaceAligner
from .quality import passes_quality
from .color import lab_match
from .dedupe import phash_dedupe, dedupe_embeddings
from .utils import ensure_dir, save_png, unique_filename


class CleanerPipeline:
    def __init__(self, input_dir, out_dir, min_conf, ref_image=None, thresholds=None):
        self.input_dir = Path(input_dir)
        self.out_dir = Path(out_dir)
        self.ref_image = cv2.imread(ref_image) if ref_image else None
        if self.ref_image is not None:
            self.ref_image = cv2.resize(self.ref_image, (512, 512))
        self.min_conf = min_conf
        self.thresholds = thresholds or {}
        ensure_dir(self.out_dir / "images")

    def run(self):
        aligner = FaceAligner(min_conf=self.min_conf)
        img_paths = sorted([p for p in self.input_dir.rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
        manifest, out_paths = [], []

        for p in tqdm(img_paths, desc="Processing"):
            img = cv2.imread(str(p))
            if img is None:
                manifest.append({"path": str(p), "status": "fail", "reason": "read_fail"})
                continue

            aligned, meta = aligner.align(img)
            if aligned is None:
                manifest.append({"path": str(p), "status": "fail", "reason": meta.get("reason", "align_fail")})
                continue

            ok, reason = passes_quality(aligned, **self.thresholds)
            if not ok:
                manifest.append({"path": str(p), "status": "fail", "reason": reason})
                continue

            if self.ref_image is not None:
                aligned = lab_match(aligned, self.ref_image)

            out_name = unique_filename(p)
            out_path = self.out_dir / "images" / out_name
            save_png(out_path, aligned)
            out_paths.append(str(out_path))
            manifest.append({"path": str(p), "status": "ok", "out": str(out_path)})

        pd.DataFrame(manifest).to_csv(self.out_dir / "manifest_pre_dedupe.csv", index=False)

        # Deduplication (fallback to pHash)
        if out_paths:
            keep_idx = phash_dedupe(out_paths)
            final_paths = [out_paths[i] for i in keep_idx]
            pd.DataFrame({"out_path": final_paths}).to_csv(self.out_dir / "manifest_final.csv", index=False)
        else:
            print("[WARN] No images passed filtering.")

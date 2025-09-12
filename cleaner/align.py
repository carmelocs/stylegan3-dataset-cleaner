"""
align.py
---------
Face detection + alignment utilities for dataset cleaning.

Two backends are supported:
1. InsightFace (preferred, accurate, needs CUDA for speed).
2. Mediapipe (fallback, CPU only).
"""

import cv2
import numpy as np

# Try to load insightface
HAS_INSIGHTFACE = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except Exception:
    HAS_INSIGHTFACE = False

# Mediapipe fallback
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


class FaceAligner:
    """Detects a single face and aligns it to a canonical 512×512 crop."""

    def __init__(self, min_conf=0.9, output_size=512, scale=1.3):
        self.min_conf = min_conf
        self.output_size = output_size
        self.scale = scale
        self.backend = None

        if HAS_INSIGHTFACE:
            try:
                self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                self.backend = "insightface"
                print("[INFO] Using InsightFace for alignment")
            except Exception as e:
                print("[WARN] InsightFace init failed:", e)

        if self.backend is None and HAS_MEDIAPIPE:
            self.backend = "mediapipe"
            self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
            self.mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
            print("[INFO] Falling back to Mediapipe for alignment")

        if self.backend is None:
            raise RuntimeError("No alignment backend found (install insightface or mediapipe).")

    def align(self, img_bgr):
        """Align one face, return aligned 512×512 image and metadata dict."""
        if self.backend == "insightface":
            return self._align_insightface(img_bgr)
        elif self.backend == "mediapipe":
            return self._align_mediapipe(img_bgr)

    def _align_insightface(self, img_bgr):
        faces = self.app.get(img_bgr)
        faces = [f for f in faces if f.det_score >= self.min_conf]
        if len(faces) != 1:
            return None, {"reason": "no_single_face", "count": len(faces)}

        f = faces[0]
        try:
            aligned = insightface.utils.face_align.norm_crop(
                img_bgr, f.kps, image_size=self.output_size, mode="arcface"
            )
            meta = {
                "det_conf": float(f.det_score),
                "bbox": [int(x) for x in f.bbox.tolist()],
                "landmarks": f.kps.tolist(),
            }
            if hasattr(f, "pose") and f.pose is not None:
                meta.update({"yaw": float(f.pose[0]), "pitch": float(f.pose[1]), "roll": float(f.pose[2])})
            return aligned, meta
        except Exception as e:
            return None, {"reason": "align_failed", "error": str(e)}

    def _align_mediapipe(self, img_bgr):
        """Simplified alignment using eye/nose positions (less accurate)."""
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(img_rgb)
        if not res.detections or len(res.detections) != 1:
            return None, {"reason": "no_single_face"}

        mesh = self.mp_mesh.process(img_rgb)
        if not mesh.multi_face_landmarks:
            return None, {"reason": "no_landmarks"}

        lms = mesh.multi_face_landmarks[0]
        coords = [(int(pt.x * w), int(pt.y * h)) for pt in lms.landmark]

        left_eye, right_eye, nose = coords[33], coords[263], coords[1]
        src = np.array([left_eye, right_eye, nose], dtype=np.float32)

        # Target landmark positions in 512×512
        target = np.array(
            [(self.output_size * 0.35, self.output_size * 0.38),
             (self.output_size * 0.65, self.output_size * 0.38),
             (self.output_size * 0.5, self.output_size * 0.58)],
            dtype=np.float32,
        )
        M = cv2.getAffineTransform(src, target)
        aligned = cv2.warpAffine(img_bgr, M, (self.output_size, self.output_size), borderMode=cv2.BORDER_REFLECT)
        return aligned, {"backend": "mediapipe", "det_conf": 1.0}

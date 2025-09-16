"""
align.py
---------
人脸检测 + 对齐模块
支持：
- InsightFace (首选)
- Mediapipe (回退)

功能：
- 保证人脸居中对齐
- 在裁剪时可保留一定比例的背景（face_scale 参数控制）
"""

import cv2
import numpy as np

# InsightFace
HAS_INSIGHTFACE = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except Exception:
    HAS_INSIGHTFACE = False

# Mediapipe
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


class FaceAligner:
    def __init__(self, min_conf=0.9, output_size=512, face_scale=1.3, keep_input_size=False):
        """
        参数:
        - min_conf: 人脸检测的最小置信度
        - output_size: 输出尺寸（int 或 None）。如果 keep_input_size=True，则忽略此参数
        - face_scale: 背景保留比例，越大背景越多
        - keep_input_size: 是否保留原始分辨率（True 时输出与输入同尺寸）
        """
        self.min_conf = min_conf
        self.output_size = output_size
        self.face_scale = face_scale
        self.keep_input_size = keep_input_size
        self.backend = None

        if HAS_INSIGHTFACE:
            try:
                self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                self.backend = "insightface"
                print("[INFO] Using InsightFace for face alignment")
            except Exception as e:
                print("[WARN] InsightFace init failed:", e)

        if self.backend is None and HAS_MEDIAPIPE:
            self.backend = "mediapipe"
            self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
            self.mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
            print("[INFO] Falling back to Mediapipe")

        if self.backend is None:
            raise RuntimeError("No backend available: install insightface or mediapipe.")

    def align(self, img_bgr):
        if self.backend == "insightface":
            return self._align_insightface(img_bgr)
        else:
            return self._align_mediapipe(img_bgr)

    def _align_insightface(self, img_bgr):
        faces = self.app.get(img_bgr)
        faces = [f for f in faces if f.det_score >= self.min_conf]
        if len(faces) != 1:
            return None, {"reason": "no_single_face", "count": len(faces)}

        f = faces[0]
        bbox = f.bbox.astype(int)  # (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox

        # 扩展 bbox 保留背景
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        new_w, new_h = int(w * self.face_scale), int(h * self.face_scale)
        x1, y1 = cx - new_w // 2, cy - new_h // 2
        x2, y2 = cx + new_w // 2, cy + new_h // 2

        # 保证不超出边界
        h_img, w_img = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        crop = img_bgr[y1:y2, x1:x2]

        if self.keep_input_size:
            aligned = cv2.resize(crop, (w_img, h_img), interpolation=cv2.INTER_AREA)
        else:
            aligned = cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)

        meta = {
            "det_conf": float(f.det_score),
            "bbox": bbox.tolist(),
            "expanded_bbox": [x1, y1, x2, y2],
        }
        return aligned, meta

    def _align_mediapipe(self, img_bgr):
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(img_rgb)
        if not res.detections or len(res.detections) != 1:
            return None, {"reason": "no_single_face"}

        det = res.detections[0]
        box = det.location_data.relative_bounding_box
        x1, y1 = int(box.xmin * w), int(box.ymin * h)
        x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)

        # 扩展 bbox 保留背景
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        new_w, new_h = int(bw * self.face_scale), int(bh * self.face_scale)
        x1, y1 = cx - new_w // 2, cy - new_h // 2
        x2, y2 = cx + new_w // 2, cy + new_h // 2
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img_bgr[y1:y2, x1:x2]
        if self.keep_input_size:
            aligned = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        else:
            aligned = cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)

        meta = {"det_conf": 1.0, "expanded_bbox": [x1, y1, x2, y2]}
        return aligned, meta

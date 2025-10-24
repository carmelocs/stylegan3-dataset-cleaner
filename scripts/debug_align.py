"""
debug_align.py
---------------
可视化调试人脸检测与对齐 (含背景保留比例)
"""

import sys
import os
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
import glob
import argparse
from cleaner.align import FaceAligner

def draw_bbox(img, bbox, color=(0,255,0), label="bbox"):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main(args):
    # 初始化对齐器
    aligner = FaceAligner(
        min_conf=args.min_conf,
        output_size=args.output_size,
        face_scale=args.face_scale,
        keep_input_size=args.keep_input_size
    )

    # 加载图片
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.*")))
    if not img_paths:
        print("[ERR] No images found in", args.input_dir)
        return

    idx = 0
    while True:
        img_path = img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print("[WARN] failed to load", img_path)
            idx = (idx + 1) % len(img_paths)
            continue

        aligned, meta = aligner.align(img)
        vis = img.copy()

        if aligned is not None and "expanded_bbox" in meta:
            # 画原始bbox
            if "bbox" in meta:
                draw_bbox(vis, meta["bbox"], color=(0,255,0), label="face")
            # 画扩展bbox
            draw_bbox(vis, meta["expanded_bbox"], color=(0,0,255), label="expanded")

            # 拼接图像（原图 + 裁剪图）
            crop_resized = cv2.resize(aligned, (vis.shape[1]//2, vis.shape[0]//2))
            vis_top = cv2.resize(vis, (vis.shape[1]//2, vis.shape[0]//2))
            combined = cv2.hconcat([vis_top, crop_resized])

            cv2.imshow("align_debug", combined)
        else:
            cv2.imshow("align_debug", vis)
            print("[WARN] No face detected in", img_path)

        key = cv2.waitKey(0)
        if key == ord('q'):  # q退出
            break
        elif key == 83 or key == ord('d'):  # → or d
            idx = (idx + 1) % len(img_paths)
        elif key == 81 or key == ord('a'):  # ← or a
            idx = (idx - 1) % len(img_paths)
        else:
            idx = (idx + 1) % len(img_paths)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    ap.add_argument("--min_conf", type=float, default=0.8, help="人脸检测最小置信度")
    ap.add_argument("--output_size", type=int, default=512, help="输出尺寸 (ignored if --keep_input_size)")
    ap.add_argument("--face_scale", type=float, default=1.3, help="背景保留比例，越大背景越多")
    ap.add_argument("--keep_input_size", action="store_true", help="是否保持原始分辨率")
    args = ap.parse_args()
    main(args)

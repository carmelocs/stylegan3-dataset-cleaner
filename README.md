# StyleGAN3 Dataset Cleaner

A modular Python pipeline for cleaning and aligning face/selfie datasets before training with [StyleGAN3](https://github.com/NVlabs/stylegan3).

This tool:
- Detects & aligns faces to **512×512** crops (InsightFace preferred, Mediapipe fallback).
- Filters low-quality images (blur, bad exposure, oversaturation).
- Normalizes colors to a reference image (LAB mean/variance).
- Removes duplicates (ArcFace embeddings + FAISS if available, pHash fallback).
- Outputs a cleaned dataset ready for StyleGAN3 (`datasets/<name>-512x512.zip`).

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stylegan3-dataset-cleaner.git
cd stylegan3-dataset-cleaner
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the pipeline on a folder of raw images:
```bash
python scripts/run_cleaner.py \
  --input_dir ./raw_images \
  --out_dir ./cleaned_dataset \
  --ref_image ./examples/reference.jpg
```

## Options

| Flag               | Description                                   | Default  |
| ------------------ | --------------------------------------------- | -------- |
| `--input_dir`      | Folder with raw input images                  | required |
| `--out_dir`        | Output folder for cleaned images & manifests  | required |
| `--ref_image`      | Optional neutral reference for color matching | None     |
| `--min_sharpness`  | Laplacian variance threshold for blur         | 120      |
| `--min_brightness` | Min allowed brightness (grayscale mean)       | 90       |
| `--max_brightness` | Max allowed brightness (grayscale mean)       | 180      |
| `--min_saturation` | Min allowed saturation (HSV)                  | 0.12     |
| `--max_saturation` | Max allowed saturation (HSV)                  | 0.55     |

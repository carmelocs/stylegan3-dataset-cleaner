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
git clone https://github.com/carmelocs/stylegan3-dataset-cleaner.git
cd stylegan3-dataset-cleaner
```

<!-- ### 2. Create and activate virtual environment
```bash
conda create -n sg3cleaner python=3.10 -y
conda activate sg3cleaner
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
``` -->

### 2. Build and activate
```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate sg3cleaner
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

## 📂 Outputs

`cleaned_dataset/images/` → aligned 512×512 PNGs

`cleaned_dataset/manifest_pre_dedupe.csv` → before deduplication

`cleaned_dataset/manifest_final.csv` → after deduplication

`datasets/<inputname>-512x512.zip` → ready-to-train StyleGAN3 dataset (if you enable zipping in the pipeline)


## 🧩 Project Structure
```bash
cleaner/
├─ align.py      # face detection & alignment
├─ quality.py    # image quality checks
├─ color.py      # color normalization
├─ dedupe.py     # duplicate removal
├─ utils.py      # helpers
├─ pipeline.py   # orchestrates cleaning
scripts/
└─ run_cleaner.py # CLI entrypoint
examples/
└─ reference.jpg  # optional neutral reference
```

## 🛠️ Notes
```bash
For best results, install InsightFace with GPU support (onnxruntime-gpu) and FAISS (faiss-gpu).

Mediapipe fallback works but is less precise for alignment.

Tune thresholds depending on your dataset quality.
```

## 📜 License
MIT License
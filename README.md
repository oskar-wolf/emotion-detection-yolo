# Emotion Detection with YOLOv8

A simple, end-to-end pipeline for detecting a person’s emotional state in images using Ultralytics YOLOv8. Covers raw data audit, preprocessing, training, detailed evaluation, and an inference demo—organized into Jupyter notebooks.

---

## 📁 Repository Structure

```text
emotion-detection-yolo/
├── data/
│   ├── raw/         # Original train/valid/test splits (images + YOLO .txt labels)
│   ├── interim/     # 640×640 face crops + labels after data audit
│   └── processed/   # Final model-ready images + labels
│
├── demo_images/     # Your 9 selfies for the inference demo
│
├── models/
│   └── emotion_detector/
│       └── weights/ # best.pt & last.pt checkpoints
│
├── runs/
│   ├── detect/          # validation outputs & metrics
│   └── inference_demo/  # saved inference visuals & JSON
│
├── notebooks/        # Phase-by-phase Jupyter notebooks
│   ├── 1_data_audit.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_train_and_val.ipynb
│   ├── 4_evaluate.ipynb
│   └── 5_inference_demo.ipynb
│
├── data.yaml         # YOLOv8 dataset config (train/val/test paths + class names)
├── environment.yml   # Conda environment spec (yolo38)
├── LICENSE           # MIT
└── README.md         # this file

```markdown
# Quickstart

## Clone & enter the repo

```bash
git clone https://github.com/oskar-wolf/emotion-detection-yolo.git
cd emotion-detection-yolo
```

## Create & activate the Conda environment

```bash
conda env create -f environment.yml
conda activate yolo38
```

## Prepare your data

- Download the “9 Facial Expressions for YOLO” dataset from Kaggle or Roboflow.
- Place `train/`, `valid/`, and `test/` folders under `data/raw/`.

## Run the notebooks

Launch Jupyter in the project root:

```bash
jupyter notebook
```

Then execute, in order:

```text
notebooks/1_data_audit.ipynb
notebooks/2_preprocessing.ipynb
notebooks/3_train_and_val.ipynb
notebooks/4_evaluate.ipynb
notebooks/5_inference_demo.ipynb
```

---

## 🗂️ Phase Overviews

### 1. Data Audit

- Detect & crop the primary face in each image using OpenCV Haar cascades.
- Resize to 640×640 and save under `data/interim/<split>/images`.
- Copy matching YOLO labels to `data/interim/<split>/labels`.
- Summarize kept images per split.

### 2. Preprocessing

- Load interim face crops.
- Normalize pixel values to `[0,1]` and convert grayscale → RGB.
- Save to `data/processed/<split>/images` and copy labels.

### 3. Training & Validation

- Fine-tune YOLOv8n on processed `train/` + validate on `valid/`.
- Monitor box/cls/DFL losses, mAP@0.5, mAP@0.5–0.95.
- Save best checkpoint to `models/emotion_detector/weights/best.pt`.

### 4. Detailed Evaluation

- Run `model.val()` on the held-out test split.
- Generate overall & per-class mAP, confusion matrix, PR/F1 curves.

### 5. Inference Demo

- Place your own nine selfies in `demo_images/`.
- Run inference, display top-2 predictions per image.
- Save outputs to `runs/inference_demo/`.

---

## ⚙️ Configuration

- **`data.yaml`** defines dataset paths and class names.
- **`environment.yml`** pins Python, PyTorch, OpenCV, Ultralytics YOLOv8, and other dependencies.

To tweak training hyperparameters, edit `notebooks/3_train_and_val.ipynb` or supply a custom `hyp_custom.yaml`.
```

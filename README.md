# Emotion Detection with YOLOv8

A simple, end-to-end pipeline for detecting a person’s emotional state in images using Ultralytics YOLOv8. Covers raw data audit, preprocessing, training, detailed evaluation, and an inference demo—organized into Jupyter notebooks.

---

## 📁 Repository Structure
emotion-detection-yolo/
├── data/
│ ├── raw/ # Original train/valid/test splits with images & labels
│ ├── interim/ # Face-cropped 640×640 images & labels after audit
│ └── processed/ # Final, model-ready images & labels
│
├── demo_images/ # Your own 9 example selfies for inference demo
│
├── models/
│ └── emotion_detector/
│ └── weights/ # best.pt & last.pt checkpoints from training
│
├── runs/
│ ├── detect/ # detection-task validation outputs
│ └── inference_demo/ # saved demo inference images & JSON
│
├── notebooks/ # Phase-by-phase Jupyter notebooks
│ ├── 1_data_audit.ipynb
│ ├── 2_preprocessing.ipynb
│ ├── 3_train_and_val.ipynb
│ ├── 4_evaluate.ipynb
│ └── 5_inference_demo.ipynb
│
├── data.yaml # YOLOv8 dataset config (train/val/test paths + class names)
├── requirements.txt # pip install -r requirements.txt
├── yolov8n.pt # pretrained YOLOv8n backbone
├── LICENSE # MIT
└── README.md # this file


---

## 🚀 Quickstart

1. **Clone & install**  
   ```bash
   git clone https://github.com/yourname/emotion-detection-yolo.git
   cd emotion-detection-yolo
   pip install -r requirements.txt

Prepare your data

Download the “8 Facial Expressions for YOLO” dataset from Kaggle/Roboflow

Place the train/, valid/, and test/ folders under data/raw/

Run the notebooks
Launch Jupyter in the project root:

Then execute, in order:

notebooks/1_data_audit.ipynb

notebooks/2_preprocessing.ipynb

notebooks/3_train_and_val.ipynb

notebooks/4_evaluate.ipynb

notebooks/5_inference_demo.ipynb

🗂️ Phase Overviews
1. Data Audit
Detect & crop the primary face in each image using OpenCV Haar cascades

Resize crops to 640×640 and save under data/interim/<split>/images

Copy matching labels to data/interim/<split>/labels

Summarize how many images per split were kept

2. Preprocessing
Load interim crops

Normalize pixels to [0,1] and convert any grayscale → RGB

Save final images under data/processed/<split>/images and copy labels

3. Training & Validation
Train a YOLOv8n model on the processed train/ and valid/ splits

Monitor box loss, classification loss, mAP@0.5 and mAP@0.5:0.95

Save best checkpoint to models/emotion_detector/weights/best.pt

4. Detailed Evaluation
Run model.val() on the held-out test split

Print overall and per-class mAP, confusion matrix, precision/recall/F1

5. Inference Demo
Place your own 9 selfies (one per emotion) in demo_images/

Run inference, display the top-2 predictions per image, and save to runs/inference_demo/

📊 Results
AP50: ~72 %

mAP50–95: ~52 %

Single-label accuracy (per-image): ~80 %

Per-class F1-score range:

Happy: ~0.94

Neutral: ~0.65

⚙️ Configuration
data.yaml defines your dataset paths and class names.

Adjust training hyperparameters directly in notebooks/3_train_and_val.ipynb or supply a custom hyp_custom.yaml to model.train().

📜 License
This project is released under the MIT License.
See LICENSE for details.
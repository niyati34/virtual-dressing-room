# 👗 Virtual Dressing Room System Using Computer Vision and Deep Learning

This repository implements a high-resolution **2D virtual try-on system** that allows users to try garments virtually using just two input images — a photo of a person and a clothing item. The pipeline is built upon the VITON-HD architecture with enhanced human parsing using the CIHP_PGN model.

---

## ✨ Project Highlights

- Realistic 2D garment fitting using **pose estimation**, **semantic segmentation**, and **image warping**.
- Uses **OpenPose** for skeletal landmark detection.
- Employs a **PGN-based segmentation model** for precise body part parsing.
- Supports **high-resolution try-on (1024x768)** using ALIAS and SPADE-based deep image synthesis.
- No 3D data or multi-view input required — works with just 2 front-view images.

---

## 📂 Project Structure

├── checkpoints/ # Downloaded model checkpoints (place here)
├── inputset/
│ └── test/
│ ├── cloth/
│ ├── cloth-mask/
│ ├── image/
│ ├── image-parse/
│ ├── openpose-img/
│ └── openpose-json/
├── results/ # Output results after running test
├── input.py # Preprocessing script for a single pair (image + cloth)
├── test.py # Inference script (try-on)
├── requirements.txt # Python dependencies (pip)
├── environment.yml # Conda environment (recommended)
└── README.md



## ⚙️ Setup Instructions

1. **Clone the repository:**


```bash
git clone https://github.com/yourusername/virtual-dressing-room.git
cd virtual-dressing-room

2. **Install dependencies:**

Using Conda (recommended):

conda env create -f environment.yml
conda activate viton-env

3. **Download Checkpoints:**

Download the following models and place them in the checkpoints/ directory:

CIHP_PGN for human parsing **(checkpoints/CIHP_PGN directory)**

GMM, SegGenerator, and ALIASGenerator weights from the official VITON-HD release.place it in folder /checkpoints

OpenPose for pose detection and keypoints **(openpose\models\pose\coco)**
🧪 Inference on Custom Inputs 

🖼 1. Prepare Your Inputs
You’ll need two front-facing images:

A person image (e.g., person.jpg)

A cloth image (e.g., cloth.png)

Place them anywhere on your system.

🔁 2. Run Preprocessing (input.py)

python input.py

When prompted, enter the full paths to your image and cloth:

👤 Enter the full path to the person image: D:/path/to/person.jpg
👕 Enter the full path to the cloth image: D:/path/to/cloth.png

This step will:

Copy and format the input images

Run OpenPose for pose keypoints

Generate the cloth mask

Generate the person segmentation using PGN

Write the required test_pairs.txt

🎨 3. Run Virtual Try-On (test.py)

python test.py \
  --name test_output \
  --dataset_dir ./inputset \
  --dataset_mode test \
  --dataset_list test_pairs.txt \
  --checkpoint_dir ./checkpoints \
  --save_dir ./results
Output images will be saved to the results/ folder.

📖 Methodology (Summary)
This project is based on VITON-HD with key improvements:

PGN Segmentation: Accurate parsing of body regions (head, torso, arms).

OpenPose: Skeleton-based pose detection for warping alignment.

TPS Warping: Flexible garment transformation using Geometric Matching Module (GMM).

ALIAS + SPADE Generator: Photorealistic final image synthesis using adversarial training with multiple loss functions (L1, VGG, GAN).

For complete details, please refer to the attached research paper in the repository.

🎓 Authors
Niyati Raiyani


📝 License
MIT License (or specify your chosen license).

📌 Acknowledgment
Special thanks to the authors of VITON-HD and the PGN parsing model. 









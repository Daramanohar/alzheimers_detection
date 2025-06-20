Alzheimer’s Disease Detection from Brain MRI Scans
A deep learning project for classifying Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) cases using MRI scans

Overview :

This project leverages transfer learning with state-of-the-art CNN architectures (VGG16, VGG19, InceptionV3) to classify brain MRI scans into three categories: Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN). The workflow includes data preprocessing, augmentation, model training, and evaluation

Features :

Data Preprocessing: Segmentation, skull stripping (placeholder), spatial normalization, and augmentation.

Model Architectures: VGG16, VGG19, and InceptionV3 with custom classification heads.

Transfer Learning: Pre-trained weights from ImageNet, fine-tuning for medical images.

Class Imbalance Handling: Automatic calculation and application of class weights.

Training and Evaluation: Early stopping, learning rate reduction, model checkpointing, and performance visualization

Requirements :

Python 3.x

TensorFlow 2.x

Keras

OpenCV

NumPy, Pandas, Matplotlib, Seaborn

Albumentations

scikit-learn

Pillow

ReportLab (for optional report generation)

Install requirements using: 
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow keras tqdm albumentations pillow reportlab

project structure :

alzheimers_classification/
│
├── data/
│   ├── raw/
│   │   └── Axial/
│   │       ├── AD/
│   │       ├── CMI/  # (Note: Should be MCI for Mild Cognitive Impairment)
│   │       └── CN/
│   ├── processed/
│   │   ├── AD/
│   │   ├── CMI/
│   │   └── CN/
│   ├── train/
│   │   ├── AD/
│   │   ├── CMI/
│   │   └── CN/
│   └── val/
│       ├── AD/
│       ├── CMI/
│       └── CN/
├── models/
│   ├── VGG16/
│   ├── VGG19/
│   └── InceptionV3/
└── alzheimers.ipynb

Usage :
Data Preprocessing:

Segments brain regions using Otsu’s thresholding.

Applies spatial normalization and augmentation.

Model Training:

Uses pre-trained models with custom dense layers.

Implements early stopping, learning rate reduction, and model checkpointing.

Evaluation:

Visualizes training history (accuracy, loss).

Generates confusion matrix and classification report.

Results
Training and validation accuracy/loss plots

Confusion matrix for each model

Classification report (precision, recall, F1-score)

References
VGG16, VGG19, InceptionV3: Original papers and Keras documentation.

Albumentations: For image augmentation.

OpenCV: For image processing.

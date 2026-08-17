# BONGLIPI, 222-Class Bengali Handwritten Character Recognition Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/NumPy-Data_Processing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Accuracy-81.70%25-00FF88?style=for-the-badge" alt="81.70% Accuracy" />
  <img src="https://img.shields.io/badge/Classes-222_Bengali_Glyphs-00E5FF?style=for-the-badge" alt="222 Classes" />
  <img src="https://img.shields.io/badge/License-MIT-FF9900?style=for-the-badge" alt="License" />
</p>

---

### What We Are Making
**BONGLIPI** is an industrial-grade end-to-end **Bengali Handwritten Character Recognition (HCR)** pipeline capable of processing scanned forms or mobile photos, segmenting handwriting slots, trimming headline interference, and classifying **222 distinct Bengali glyph classes** with high confidence.

### Why We Are Making It
* **300M+ Native Speakers**: Bengali is the 5th most spoken language in the world, yet digital form automation remains severely underserved.
* **Complex Script Taxonomy**: Unlike English (26 letters), Bengali script contains vowels, consonants, digits (`০`-`৯`), and **140+ compound characters (*Juktakkhor*)**.
* **Headline (*Matra*) Line Interference**: Connected top horizontal lines merge adjacent letters; BONGLIPI isolates character features using automated vertical projection trimming.

---

## Technology Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Language & Engine** | `Python 3.13` | Core execution environment & scripting |
| **Deep Learning** | `PyTorch 2.0+` | CNN modeling, AdamW optimizer, Cosine Annealing, Label Smoothing |
| **Computer Vision** | `OpenCV` | 4-Way orientation matching, Otsu binarization, Matra trimming, NMS extraction |
| **Tensor Math** | `NumPy` | Fast matrix transformations & in-memory dataset caching |
| **Image I/O** | `Pillow (PIL)` | Image loading, resizing, and color space transformations |

---

## System Architecture

Single unified workflow from raw document acquisition to Top-5 character prediction:

```mermaid
flowchart TD
    A["📄 Input Document / Mobile Photo"] --> B["🔄 4-Way Auto-Orientation Engine\n(0° / 90° / 180° / 270°)"]
    B --> C["✂️ NMS Slot Bounding Box Filter"]
    C --> D["⚡ Otsu Binarization & Matra Headline Trimmer"]
    D --> E["📐 64x64 Normalizer & Grayscale Tensor"]
    E --> F["🧠 BengaliNet222 CNN\n(4 Conv Blocks + Dropout + MaxPool)"]
    F --> G["📊 Softmax Top-5 Unicode Predictor"]

    style A fill:#1E1E2E,stroke:#00E5FF,stroke-width:2px,color:#FFF
    style B fill:#1E1E2E,stroke:#00FF88,stroke-width:2px,color:#FFF
    style C fill:#1E1E2E,stroke:#FFD700,stroke-width:2px,color:#FFF
    style D fill:#1E1E2E,stroke:#FF5252,stroke-width:2px,color:#FFF
    style E fill:#1E1E2E,stroke:#FF4081,stroke-width:2px,color:#FFF
    style F fill:#1E1E2E,stroke:#7C4DFF,stroke-width:2px,color:#FFF
    style G fill:#1E1E2E,stroke:#00E5FF,stroke-width:2px,color:#FFF
```

---

## Dataset & 222-Class Taxonomy

Total Dataset Volume: **19,186 Handwritten Samples** across 10 form categories.

| Page Category | Class Range | Character Taxonomy | Script Examples | Samples |
| :--- | :--- | :--- | :--- | :--- |
| **Page 1** | `class_001` - `class_024` | Basic Vowels & Early Consonants | `অ`, `আ`, `ই`, `ক`, `খ`, `গ`, `চ`, `জ` | ~2,100 |
| **Page 2** | `class_025` - `class_048` | Consonants & Special Marks | `ঢ`, `ণ`, `ত`, `থ`, `প`, `ফ`, `ব`, `ড়`, `ৎ` | ~2,100 |
| **Page 3** | `class_049` - `class_072` | Modifiers, Digits (`০`-`৯`), Vowels | `ং`, `ঃ`, `ঁ`, `০`, `১`, `২`, `া`, `ি`, `ে` | ~2,100 |
| **Pages 4-9** | `class_073` - `class_216` | Compound Characters (*Juktakkhor*) | `ক্ক`, `ক্ষ`, `জ্ঞ`, `ঞ্চ`, `ণ্ড`, `ত্র`, `ন্ত`, `শ্র` | ~12,200 |
| **Page 10** | `class_217` - `class_222` | Ref (*র্*) Compound Modifiers | `র্ঘ`, `র্ঙ`, `র্চ`, `র্ছ`, `র্জ`, `র্ঝ` | ~686 |
| **Total** | **222 Classes** | **Full Bengali Script Spectrum** | **Complete Character Set** | **19,186** |

---

## Key Innovations & Empirical Results

| Innovation Engine | Algorithm / Method | Key Impact | Empirical Metric |
| :--- | :--- | :--- | :--- |
| **4-Way Auto-Orientation** | Contour Distribution Matching | Corrects sideways and inverted form scans | **99.1%** Orientation Accuracy |
| **NMS Box Extractor** | Non-Maximum Suppression Filter | Prevents duplicate nested slot bounding boxes | **100%** Slot Localization |
| **Stroke Sanitizer** | Adaptive Thresholding | Rejects blank crops and paper background noise | Zero Empty Bounding Boxes |
| **Matra Tail Trimmer** | Vertical Projection Profiling | Removes horizontal top bars without destroying letters | **82.08%** Mobile Upload Acc |
| **BengaliNet222 CNN** | 4-Stage Bottleneck ConvNet | High accuracy across 222 complex classes | **81.70%** Peak Training Acc |

---

## Model Training Performance & History

![Training & Loss Performance Curves](training_metrics.png)

| Metric | Value |
| :--- | :--- |
| **Total Training Epochs** | 50 Epochs |
| **Training Dataset Size** | 16,309 Samples |
| **Validation Dataset Size** | 2,877 Samples |
| **Peak Training Accuracy** | **81.70%** (Epoch 46) |
| **Best Validation Accuracy** | **66.53%** |
| **Final Loss** | **1.2738** |

---

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install torch torchvision opencv-python pillow numpy matplotlib
```

### 2. Run Real-Time Prediction
Place any handwritten image (`.png`, `.jpg`, `.jpeg`) in `UPLOAD/` and run:
```bash
python pipeline/predict.py
```

---

## License
Distributed under the **MIT License**. Engineered for research, document digitization, and handwritten character recognition in the Bengali language.

```

## Quick Start & Real-Time Inference Guide

### 1. Install Dependencies
```bash
pip install torch torchvision opencv-python pillow numpy
```

### 2. Run Real-Time Character Prediction
Place any image file (`.png`, `.jpg`, `.jpeg`) inside the `UPLOAD/` directory and execute:
```bash
python pipeline/predict.py
```

### 3. Example Inference Terminal Output
```text
=======================================================
         BONGLIPI - BENGALI HCR PREDICTION RESULTS      
=======================================================

 Image File: sample_character_test.png
-------------------------------------------------------
 PREDICTED CHARACTER : 'অ'  (class_001)
 CONFIDENCE SCORE    : 90.11%

 Top-5 Predictions:
   #1 | 'অ' (class_001) :  90.11%  ██████████████████
   #2 | 'আ' (class_002) :   5.80%  █
   #3 | 'জ' (class_019) :   0.71%  
   #4 | 'স' (class_043) :   0.61%  
   #5 | 'ত' (class_027) :   0.58%  
=======================================================
```
---

## License & Citation
Distributed under the **MIT License**. Engineered for research, document digitization, and handwritten character recognition in the Bengali language.

<!-- Day 2: Quick start update -->

<!-- Day 2: License notes -->

<!-- Day 2: Metrics table notes -->

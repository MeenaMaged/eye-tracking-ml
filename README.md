# Eye Tracking ML System

> **CSE381 Introduction to Machine Learning - Course Project**

A real-time webcam-based eye tracking system implementing gaze estimation, blink detection, and eye movement classification using machine learning algorithms built entirely from scratch.

**GitHub Repository:** `https://github.com/YOUR_USERNAME/eye-tracking-ml`

---

## Features

- **Real-time Eye Tracking**: Webcam-based tracking using MediaPipe FaceMesh (478 landmarks)
- **Gaze Estimation**: Predict screen coordinates from eye features
- **Blink Detection**: Binary classification of eye state (open/closed)
- **Eye Movement Classification**: 3-class classification (Fixation, Saccade, Blink)
- **ML Algorithms from Scratch**: All core ML algorithms implemented without sklearn
- **9-Point Calibration System**: Custom calibration for gaze mapping
- **Comprehensive Evaluation**: Cross-validation, ROC-AUC, confusion matrices

---

## Project Structure

```
eye-tracking-ml/
├── src/                          # Core tracking modules
│   ├── capture/                  # Webcam capture
│   ├── detection/                # Face/landmark detection
│   ├── features/                 # Feature extraction
│   ├── calibration/              # Calibration system
│   └── dataset/                  # Data collection utilities
├── ml_from_scratch/              # ML algorithms (from scratch)
│   ├── linear_regression.py      # OLS + Ridge regression
│   ├── polynomial_features.py    # Polynomial feature expansion
│   ├── neural_network.py         # MLP with backpropagation
│   ├── decision_tree.py          # CART with entropy/gini
│   ├── random_forest.py          # Bagging + random features
│   ├── xgboost.py                # Gradient boosting
│   ├── knn.py                    # K-Nearest Neighbors
│   ├── svm.py                    # SVM with SMO algorithm
│   ├── naive_bayes.py            # Gaussian Naive Bayes
│   ├── pca.py                    # Principal Component Analysis
│   ├── genetic_algorithm.py      # GA feature selection
│   └── fishers_discriminant.py   # LDA / Fisher's discriminant
├── gaze_estimation/              # Gaze model comparison
├── classification/               # Classifier comparison
├── feature_selection/            # Feature selection comparison
├── evaluation/                   # Metrics & cross-validation
├── integration/                  # Live demo & batch evaluation
├── scripts/                      # Utility scripts
├── tests/                        # Unit tests
├── results/                      # Generated results & plots
└── main.py                       # Entry point
```

---

## Installation

### Requirements
- Python 3.10+
- Webcam (for live demo)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/eye-tracking-ml.git
cd eye-tracking-ml

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- OpenCV >= 4.8.0
- MediaPipe >= 0.10.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- SciPy >= 1.11.0

---

## Usage

### Live Demo (Webcam Required)

```bash
# Run real-time eye tracking demo
python main.py

# Run calibration
python main.py --calibrate

# Test individual modules
python main.py --test-webcam
python main.py --test-detection
python main.py --test-features
```

### Generate Results

```bash
# Generate all evaluation results (with synthetic data)
python scripts/generate_results.py --synthetic

# Run black-box demo for all ML features
python scripts/demo_all_features.py all --no-pause
```

### Python API

```python
# Gaze Estimation
from gaze_estimation import compare_gaze_models
comparison = compare_gaze_models(X, y, test_size=0.2)
best = comparison.get_best_model()

# Blink Detection
from classification import compare_classifiers
comparison = compare_classifiers(X, y, task='blink')
comparison.print_results()

# Movement Classification (3-class)
from classification import compare_movement_classifiers
comparison = compare_movement_classifiers(X, y)
comparison.print_per_class_results()

# Feature Selection
from feature_selection import compare_feature_selection
from ml_from_scratch import RandomForestClassifier
comparison = compare_feature_selection(X, y, RandomForestClassifier)

# Cross-Validation
from evaluation import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
```

---

## ML Algorithms Implemented

All algorithms implemented from scratch (numpy only, no sklearn for core logic):

### Regression
| Algorithm | Description |
|-----------|-------------|
| Linear Regression | Ordinary Least Squares via normal equation |
| Ridge Regression | L2-regularized normal equation |
| Polynomial Regression | Feature expansion (degree 2-3) + linear |
| Neural Network | MLP with ReLU, backpropagation |

### Classification
| Algorithm | Description |
|-----------|-------------|
| Decision Tree | CART with entropy/gini, pre-pruning |
| Random Forest | Bagging + random feature selection + OOB |
| XGBoost | Gradient boosting with 2nd-order Taylor expansion |
| K-Nearest Neighbors | Euclidean distance, uniform/weighted voting |
| SVM | SMO algorithm, linear/RBF kernels, One-vs-Rest |
| Gaussian Naive Bayes | Gaussian likelihood per class |
| Neural Network | MLP with softmax output |

### Feature Selection
| Method | Description |
|--------|-------------|
| PCA | Eigendecomposition, variance threshold |
| Genetic Algorithm | Tournament selection, crossover, mutation |
| Fisher's LDA | Multi-class Linear Discriminant Analysis |

### Evaluation Metrics
- Classification: Confusion Matrix, Accuracy, Precision, Recall, F1-Score
- ROC-AUC: Binary and Multi-class (One-vs-Rest)
- Regression: MSE, MAE, RMSE, R², Mean Pixel Error
- Cross-Validation: Stratified K-Fold

---

## Feature Extraction

### Base Features (10 dimensions)
```
[iris_x_L, iris_y_L, iris_x_R, iris_y_R, EAR_L, EAR_R, pitch, yaw, roll, inter_ocular_dist]
```

### Blink Detection Features (19 dimensions)
- Base features + temporal features (velocity, rolling statistics)

### Movement Classification Features (14 dimensions)
```
[velocity, mean_velocity, max_velocity, acceleration, mean_abs_acceleration,
 dispersion, rms_deviation, direction_consistency,
 mean_ear, min_ear, std_ear, velocity_std, x_range, y_range]
```

---

## Results

### Gaze Estimation (Regression)
| Model | Test Error (px) | R² Score |
|-------|-----------------|----------|
| **Polynomial (deg=2)** | **34.8** | **0.9932** |
| Linear Regression | 65.5 | 0.9721 |
| Ridge (α=1.0) | 65.8 | 0.9719 |

### Blink Detection (Binary Classification)
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **Decision Tree** | **100%** | **1.0000** |
| SVM (RBF) | 100% | 1.0000 |
| Naive Bayes | 100% | 1.0000 |

### Movement Classification (3-Class)
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **XGBoost** | **100%** | **1.0000** |
| Naive Bayes | 100% | 1.0000 |
| Decision Tree | 99.2% | 0.9891 |

### Cross-Validation
- Model: Random Forest (n_estimators=50)
- 5-Fold Stratified CV
- Mean Accuracy: **94.0% (± 1.6%)**

---

## Screenshots

Live demo screenshots available in `screenshots/` folder:
- Face detection visualization
- Eye state detection (both eyes, left/right open)
- Gaze direction estimation
- Blink detection

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific phase tests
python -m pytest tests/test_phase8_integration.py -v
```

---

## License

This project was developed for CSE381 Introduction to Machine Learning course.

---

## Acknowledgments

- MediaPipe for face mesh detection
- Course instructors and TAs for guidance

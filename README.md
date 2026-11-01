# bio-touch-id-blood-group-detection
Non-invasive blood group detection from fingerprint images using Python, OpenCV and machine learning (Scikit-learn, TensorFlow).
# BIO Touch ID: Fingerprint-Based Blood Group Prediction

A machine learning pipeline that predicts blood group (8 classes) from fingerprint images using classical image processing, hand-crafted feature extraction and Scikit-learn classifiers. Includes a Streamlit demo app. Research published at ARET-2026.

> **Research prototype, not a medical or diagnostic tool.**

## Overview
Fingerprint images are preprocessed, enhanced and converted into a 36-dimensional feature vector, which is then classified into one of 8 blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-).

## Dataset
- Kaggle: *Finger Print Based Blood Group Dataset* (`rajumavinmar/finger-print-based-blood-group-dataset`)
- 6,000 fingerprint images across 8 blood-group classes
- The notebook downloads it automatically with `kagglehub`. Raw images are not included in this repo.

## Pipeline
1. **Preprocessing:** border cropping, histogram equalization, normalization
2. **Enhancement:** orientation estimation, ridge frequency estimation, block-wise Gabor filtering
3. **Feature extraction (36 features):** minutiae (ridge endings, bifurcations), ridge/valley thickness, ridge frequency, Local Binary Pattern histogram (9), orientation histogram (8), wavelet statistics (14)
4. **Modeling:** feature standardization, 80/20 train-test split, comparison of six classifiers
5. **Deployment:** Streamlit app for single-image prediction

## Results
80/20 split, 1,200 test images:

| Model | Test accuracy |
|---|---|
| Logistic Regression | 80.25% |
| SVM (linear) | 79.42% |
| XGBoost | 79.00% |
| Gradient Boosting | 77.58% |
| Random Forest | 76.75% |
| KNN (k=3) | 64.33% |

Logistic Regression cross-validated accuracy: 77.5%. After IQR outlier removal (3,882 samples): 77.5%. The deployed model is Logistic Regression.

## Run the notebook
```
pip install -r requirements.txt
jupyter notebook notebooks/Project.ipynb
```

## Run the Streamlit app
```
pip install -r requirements.txt
streamlit run app/App.py
```
Upload a fingerprint image (samples are in `sample_images/`) to see the extracted features and predicted blood group.

## Limitations
- The relationship between fingerprint patterns and blood group is still debated in the literature. Results here are specific to this dataset.
- A random train/test split is used; if the dataset contains multiple fingerprints per person, this can inflate accuracy.
- Feature scaling was fitted on the full dataset before the split.
- The app's feature extraction is a simplified version of the notebook pipeline, so its predictions are for demonstration only.

## Repository structure
```
├── notebooks/Project.ipynb               # Full pipeline and experiments
├── app/App.py                            # Streamlit demo
├── models/logistic_regression_model.pkl  # Trained model
├── sample_images/                        # Example inputs
└── requirements.txt
```

## Publication
*Non-Invasive Blood Group Detection using Fingerprint Patterns with Artificial Intelligence and Machine Learning*, 4th International Conference on Advanced Research in Engineering and Technology (ARET-2026). [Add link/DOI if available]

## Roadmap
- Integrate with IoT hardware for end-to-end capture and prediction
- Evaluate with a subject-wise train/test split
- Try deep learning models

## Author
Ananya Vijay | ECE, Dayananda Sagar College of Engineering

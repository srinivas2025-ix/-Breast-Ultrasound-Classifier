# Breast Ultrasound Classifier (Random Forest)

An end-to-end machine learning project that classifies breast ultrasound images as Normal, Benign, or Malignant, with a simple Cancer / No Cancer summary, confidence scores, and an interactive web interface.

**Disclaimer:** This is a learning/portfolio project trained on a small public dataset (~780 images). It is not a validated medical diagnostic tool and should never be used for real health decisions.

**What it does:** Loads the public BUSI dataset (breast ultrasound images labeled by clinicians), cleans each image (noise removal, contrast enhancement), and extracts hand-crafted features — intensity statistics, texture (GLCM), edge density, and shape (HOG). A Random Forest classifier is trained on these features, evaluated with 5-fold cross-validation, and compared against Logistic Regression and SVM baselines. Predictions are served through a Gradio web interface showing a binary Cancer / No Cancer summary, the detailed medical class and confidence breakdown, a confidence bar chart, and a 3D pixel-intensity surface visualization.

**Why Random Forest instead of a CNN:** This was a deliberate choice to demonstrate classical ML feature engineering rather than relying on a pretrained deep learning backbone. Random Forest with hand-crafted features typically performs a bit below a fine-tuned CNN, but it required designing the texture/shape features by hand.

**Results:** Test accuracy: [0.79]%. 5-fold cross-validation accuracy: [ 70.77]% (+/- [0.79]%). Outperforms Logistic Regression and SVM baselines trained on the same features.

**Tech stack:** Python, scikit-learn, scikit-image, OpenCV, Gradio, Matplotlib.

**Limitations:** Small dataset (~780 images) limits generalization. Hand-crafted features cap performance below what a CNN would likely achieve. Not clinically validated — for portfolio/learning purposes only.

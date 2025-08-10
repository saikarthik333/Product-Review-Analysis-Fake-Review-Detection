# Product Review Analysis & Fake Review Detector 🕵️‍♂️

![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end data science application that performs real-time sentiment analysis and detects potentially fake or biased product reviews using NLP and Machine Learning, deployed in an interactive web app.

![Live App Demo](your_gif_link_here.gif)

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Tech-Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Model Iteration and Improvement](#-model-iteration-and-improvement)
- [How to Run](#-how-to-run)
- [Challenges and Learnings](#-challenges-and-learnings)
- [Contact](#-contact)

---

## 📋 Overview
This project is a complete data science application built from scratch. It takes raw text data from the Amazon Product Reviews Dataset, processes it, and trains two different models: a deep learning model for sentiment analysis and a machine learning model for fake review detection. These models are then served through a user-friendly web interface built with Streamlit, allowing for real-time analysis of any user-provided review.

---

## ✨ Features
- **Sentiment Analysis**: Classifies review sentiment as **POSITIVE** or **NEGATIVE** using a fine-tuned **DistilBERT** model from Hugging Face, powered by a **TensorFlow** backend.
- **Intelligent Fake Review Detection**: Uses an **XGBoost Classifier** to identify potentially biased or low-effort reviews based on lexical and text-based features.
- **Real-World Data**: Trained and evaluated on the **Amazon Product Reviews Dataset** for robust performance.
- **Interactive UI**: Clean, user-friendly web interface built with **Streamlit** for easy input and visualization.

---

## 🛠️ Tech-Stack
- **Languages:** `Python`
- **Data Science & ML/NLP:**
  - `Pandas` & `NumPy`: Data Manipulation
  - `Scikit-learn`: Feature Engineering (TF-IDF)
  - `NLTK`: NLP (Stopwords)
  - `XGBoost`: Fake Review Classification
  - `TensorFlow` & `Hugging Face Transformers`: Sentiment Analysis
- **Web Application:** `Streamlit`
- **Development & Version Control:**
  - `Jupyter Notebooks`: Prototyping & Training
  - `VS Code`: Code Editor
  - `Git` & `GitHub`: Version Control

---

## 🚀 Project Workflow
1. **Environment Setup**: Created `venv` for dependencies listed in `requirements.txt`.
2. **Data Processing**: Parsed, sampled, and cleaned Amazon Reviews dataset; engineered proxy "fake" label; saved as `processed_reviews.csv`.
3. **Feature Engineering**: Generated TF-IDF features, uppercase ratio, punctuation count, and other lexical metrics.
4. **Model Training**: Trained **XGBoost Classifier** on balanced dataset.
5. **Model Evaluation**: Tested on unseen data for accuracy, precision, and recall.
6. **Application Development**: Integrated models into **Streamlit** app for real-time inference.

---

## 🧠 Model Iteration and Improvement
- **Problem**: Initial small, imbalanced dataset led to overfitting.
- **Solution**: Built a **larger, balanced dataset** (40,000+ reviews).
- **Result**: Retrained model achieved much more robust performance, proving that data quality and quantity are critical.

---

## ⚙️ How to Run

## 1. Prerequisites
- Python 3.10 or 3.11
- Git

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/saikarthik333/Product-Review-Analysis-Fake-Review-Detection.git
cd Product-Review-Analysis-Fake-Review-Detection

# Create and activate a virtual environment
python -m venv venv

# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# On macOS/Linux
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt
````

### 3. Download Dataset

Get the Amazon Product Reviews dataset from Kaggle.

Download and extract:

* `train.ft.txt.bz2`
* `test.ft.txt.bz2`

Place the extracted `train.ft.txt` and `test.ft.txt` files inside the `data/` directory.

---

### 4. Process Data & Train Model

Open and run all cells in:

* `notebooks/1_Data_Processing.ipynb`
* `notebooks/2_Model_Training_and_Evaluation.ipynb`

---

### 5. Launch App

```bash
streamlit run app/app.py
```


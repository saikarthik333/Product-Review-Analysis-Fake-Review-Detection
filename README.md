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
Of course. Here is the updated How to Run section for your README.md file.

It now includes separate, clear instructions for running the app locally versus deploying it to the cloud. This explains the purpose of the two different model-loading methods in your app.py script.

## How to Use
Simply copy this updated section and replace the existing How to Run section in your README.md file on GitHub.

Markdown

## ⚙️ How to Run

This project is configured with two modes for loading the ML models: one for running locally on your machine and one optimized for deployment on platforms like Streamlit Community Cloud.

### 1. Prerequisites
* Python 3.10 or 3.11
* Git

### 2. Initial Setup (Do this once)
```bash
# Clone the repository
git clone [https://github.com/saikarthik333/Product-Review-Analysis-Fake-Review-Detection.git](https://github.com/saikarthik333/Product-Review-Analysis-Fake-Review-Detection.git)
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
### 3. Running the App Locally
For local development, the app loads model files directly from your machine.

#### A. Download the Dataset

Get the Amazon Product Reviews dataset from Kaggle.

Download and extract train.ft.txt.bz2 and test.ft.txt.bz2.

Place the extracted train.ft.txt and test.ft.txt files inside the data/ directory.

#### B. Process Data & Train the Model

You must run the Jupyter Notebooks to generate the model files locally.

Open and run all cells in notebooks/1_Data_Processing.ipynb.

Open and run all cells in notebooks/2_Model_Training_and_Evaluation.ipynb.

#### C. Launch the App

Before running, open app/app.py and ensure the local development code for loading models is active and the Hugging Face code is commented out.

Run the following command in your terminal:
````
Bash

streamlit run app/app.py

````

### 4. Deploying to Streamlit Cloud
For deployment, the app downloads the pre-trained models from Hugging Face Hub, so you don't need to run the notebooks.

#### A. Prepare the App for Deployment

Open app/app.py.

Ensure the Hugging Face Hub code for loading models is active and the local development code is commented out.

#### B. Deploy

Commit and push your changes to GitHub.

Go to Streamlit Community Cloud, link your GitHub account, and deploy the repository.

Repository: saikarthik333/Product-Review-Analysis-Fake-Review-Detection

Branch: main

Main file path: app/app.py


## 🧠 Model Iteration & Deployment Strategy

A key part of this project was not just building a model, but improving it and successfully deploying it while navigating real-world technical challenges.

### 1. Fixing an Overfitting Model
* **Problem**: An initial model trained on a small, 672-review dataset showed extreme overfitting. It had perfect scores on the validation set but failed completely on the large, unseen test data, producing unreliable predictions in the app.
* **Solution**: I re-architected the data processing pipeline in `1_Data_Processing.ipynb` to parse a much larger pool of 2 million reviews and create a larger, properly balanced dataset of over 40,000 reviews.
* **Result**: Retraining an XGBoost model on this superior dataset produced a significantly more robust and reliable model, proving that data quality and quantity are critical for success.

### 2. Solving the Deployment Challenge
* **The Initial Plan**: For deployment, my first approach was to use **Git LFS** to handle the large model files (`.pkl`) that needed to be on GitHub for Streamlit Cloud to access.
* **The Problem**: During the deployment process, the repeated cloning of the repository quickly exhausted the **free 1GB monthly bandwidth** provided by GitHub for Git LFS. This caused the deployment to fail.
* **The Solution**: To solve this, I re-architected the deployment strategy. I uploaded the final XGBoost model and TF-IDF vectorizer to **Hugging Face Hub**, a platform designed specifically for hosting model artifacts. I then modified the Streamlit application (`app.py`) to download these models directly from my Hugging Face repository (`SaiKarthik333/product-review-analysis-fake-review-detection`) when the app starts. This is the standard, professional method for deploying ML apps as it keeps the GitHub repository small and bypasses LFS bandwidth limits.

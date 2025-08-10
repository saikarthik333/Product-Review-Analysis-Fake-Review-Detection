# Product Review Analysis & Fake Review Detector 🕵️‍♂️

This project is an end-to-end data science application that analyzes customer product reviews in real-time. It performs sentiment analysis to determine if a review is positive or negative and uses a machine learning model to detect if the review shows characteristics of being fake or spammy. The entire system is deployed in an interactive web application built with Streamlit.

![Live App Demo](your_gif_link_here.gif)

---

## ✨ Core Features

- **Sentiment Analysis**: Classifies review sentiment (**POSITIVE** or **NEGATIVE**) using a fine-tuned **DistilBERT** model from the Hugging Face library, powered by a **TensorFlow** backend.
- **Intelligent Fake Review Detection**: Uses a powerful **XGBoost Classifier** to identify potentially biased or low-effort reviews based on a combination of text-based features.
- **Real-World Data**: The model was trained and evaluated on the large-scale **Amazon Product Reviews Dataset**.
- **Interactive UI**: A clean and user-friendly web interface built with **Streamlit** allows for easy input and clear visualization of the analysis results.

---

## 📖 Project Workflow & Model Iteration

This project followed a complete machine learning lifecycle, from data processing to model improvement and deployment.

### 1. Data Processing and Preparation

The project started with the raw Amazon Reviews dataset, which contained millions of reviews in FastText format. To create a usable training set, I performed the following steps:

1. **Parsing Large Files**: Successfully parsed 2,000,000 reviews from the raw `train.ft.txt` file.

    ![Parsing Image](your_parsing_image_link_here.png)

2. **Defining the Target Variable**: Since no "fake" label existed, I engineered a proxy: reviews with less than 15 words were labeled as potentially "fake" (1), and longer reviews were labeled "genuine" (0).

3. **Creating a Balanced Dataset**: The raw data contained very few short reviews. To prevent model bias, I created a balanced dataset by sampling an equal number of "fake" and "genuine" reviews.

    ![Balancing Image](your_balancing_image_link_here.png)

    This process resulted in a clean, balanced `processed_reviews.csv` file ready for model training.

    ![Dataframe Image](your_dataframe_image_link_here.png)

### 2. Model Development and Improvement

A key part of this project was iterating on the model to improve its performance. After an initial attempt with a RandomForestClassifier revealed overfitting issues, I upgraded the model with two major improvements:

1. **Smarter Features**: Implemented **TF-IDF (Term Frequency-Inverse Document Frequency)** to analyze the actual words in the text, combined with other nuanced features like uppercase word ratio and exclamation point counts.
2. **A More Powerful Algorithm**: Upgraded the model to an **XGBoost Classifier**, which is well-known for its high performance.

### 3. Model Evaluation

The improved XGBoost model was evaluated on both a validation set (a split from the training data) and a final, unseen test set (`test.ft.txt`).

**Validation Performance:**
The model showed excellent performance on the validation set, achieving **99% accuracy** and demonstrating that it learned the patterns in the training data effectively.

![Validation Report](your_validation_report_link_here.png)
![Validation Matrix](your_validation_matrix_link_here.png)

**Final Test Performance:**
On the completely unseen test set of 400,000 reviews, the model's true performance was revealed.

![Test Report](your_test_report_link_here.png)
![Test Matrix](your_test_matrix_link_here.png)

The key takeaways from the final evaluation are:

- **High Recall (1.00 for "Fake")**: The model successfully identified **all** of the short, spammy reviews.
- **Low Precision (0.06 for "Fake")**: The model produces a high number of false positives, showing that it is very sensitive but not always accurate when it flags a review as fake.

After training, the final XGBoost model and TF-IDF vectorizer were saved for use in the application.

![Saved Model](your_saved_model_image_link_here.png)

---

## 🚀 How to Run This Project

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.10 or 3.11
- Git

### 2. Set Up the Project

Clone the repository
git clone https://github.com/saikarthik333/Product-Review-Analysis-Fake-Review-Detection.git
cd Product-Review-Analysis-Fake-Review-Detection

Create and activate a virtual environment
python -m venv venv

On Windows (PowerShell)
.\venv\Scripts\Activate.ps1

On macOS/Linux
source venv/bin/activate

Install the required libraries
pip install -r requirements.txt


### 3. Download the Dataset

This project requires the Amazon Product Reviews dataset.

- Go to the Amazon Reviews for Sentiment Analysis page on Kaggle.
- Download and decompress the `train.ft.txt.bz2` and `test.ft.txt.bz2` files.
- Place the resulting `train.ft.txt` and `test.ft.txt` files inside the `data/` directory.

### 4. Process Data and Train the Model

You must run the Jupyter Notebooks in order to process the data and create the model file.

- Open and run all cells in `notebooks/1_Data_Processing.ipynb`.
- Open and run all cells in `notebooks/2_Model_Training_and_Evaluation.ipynb`.

### 5. Launch the Streamlit App

Once the model is trained and saved, you can launch the application.

streamlit run app/app.py


---

## 🛠️ Tech Stack

**Languages:** Python

**Data Science & ML/NLP Frameworks:** TensorFlow, Scikit-learn, XGBoost, Hugging Face Transformers

**Libraries:** Pandas, NumPy, NLTK, Matplotlib, Seaborn, tf-keras

**Web Application:** Streamlit

**Development & Version Control:** Git, GitHub, VS Code, Jupyter Notebooks, Virtual Environments (venv)

---

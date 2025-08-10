import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path
from transformers import pipeline
from scipy.sparse import hstack

# Add the parent directory to the system path to allow imports from the 'utils' folder
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import create_features, clean_text

# --- Page Configuration ---
st.set_page_config(
    
    page_title="Product Review Analysis & Fake Review Detection 🕵️‍♂️",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Caching Models for Performance ---
@st.cache_resource
def load_models():
    """Load all models and vectorizers."""
    model_path = Path(__file__).resolve().parent.parent / "models/fake_review_model.pkl"
    vectorizer_path = Path(__file__).resolve().parent.parent / "models/tfidf_vectorizer.pkl"
    
    if not model_path.exists() or not vectorizer_path.exists():
        st.error("Model or vectorizer file not found! Please run the training notebooks first.")
        return None, None, None
    
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="tf")
    fake_detector = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    return sentiment_analyzer, fake_detector, tfidf_vectorizer

# --- Load Models ---
sentiment_analyzer, fake_detector, tfidf_vectorizer = load_models()

# --- UI Layout ---
st.title("Product Review Analysis & Fake Review Detection 🕵️‍♂️")
st.markdown("Enter a product review to analyze its sentiment and check for signs of being fake or biased.")

# --- Input Area ---
review_input = st.text_area(
    "Enter the review text here:",
    height=150,
    placeholder="e.g., 'This product is absolutely fantastic! I highly recommend it to everyone.'",
)

# --- Analysis Button ---
if st.button("Analyze Review", type="primary", use_container_width=True):
    if review_input and fake_detector and tfidf_vectorizer:
        # --- Sentiment Analysis ---
        with st.spinner("Analyzing sentiment..."):
            sentiment_result = sentiment_analyzer(review_input)[0]
            sentiment_label = sentiment_result['label']
            sentiment_score = sentiment_result['score']

        st.subheader("Sentiment Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            if sentiment_label == 'POSITIVE':
                st.success(f"**Sentiment: {sentiment_label}**")
            else:
                st.error(f"**Sentiment: {sentiment_label}**")
        with col2:
            st.metric(label="Confidence Score", value=f"{sentiment_score:.2%}")
        
        st.progress(float(sentiment_score if sentiment_label == 'POSITIVE' else 1 - sentiment_score))
        
        st.markdown("---")

        # --- Fake Review Detection ---
        with st.spinner("Checking for fake review patterns..."):
            # Create a DataFrame for the single review
            input_df = pd.DataFrame([review_input], columns=['review_text'])
            
            # 1. Create base features using the helper function
            base_features = create_features(input_df)
            
            # 2. Clean the text and apply the loaded TF-IDF vectorizer
            cleaned_input = clean_text(review_input)
            tfidf_features = tfidf_vectorizer.transform([cleaned_input])
            
            # 3. Combine base features and TF-IDF features in the correct order
            combined_features = hstack([tfidf_features, base_features])
            
            # 4. Predict using the XGBoost model
            prediction = fake_detector.predict(combined_features)[0]
            prediction_proba = fake_detector.predict_proba(combined_features)[0]
        
        # --- Display Fake Review Results ---
        st.subheader("Fake Review Detection Results")
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.warning("**Result: Potential Fake Detected**")
                st.markdown("This review shows characteristics often found in biased, spam, or unverified reviews.")
            else:
                st.info("**Result: Appears Genuine**")
                st.markdown("This review seems to be authentic based on our analysis.")
        with col2:
            # Check if the model predicted both classes or was 100% certain of one.
            if len(prediction_proba) > 1:
                fake_prob = prediction_proba[1]
            else:
                fake_prob = 1.0 if prediction == 1 else 0.0

            st.metric(label="Fake Probability", value=f"{fake_prob:.2%}")
        
        st.progress(float(fake_prob))

    elif not fake_detector:
        pass # Error is handled in the loading function
    else:
        st.warning("Please enter a review text to analyze.")

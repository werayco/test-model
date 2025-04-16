import streamlit as st
import re
import subprocess
import pickle
from nltk.stem import WordNetLemmatizer
import os
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'^\s*"|"\s*$', '', text)  
    text = re.sub(r'[^\w\s]', ' ', text)     
    text = re.sub(r'\s+', ' ', text)         
    text = text.lower().strip()             
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

vec_path = os.path.join("ml_models","tfidf_vectorizer.pkl")
model_path = os.path.join("ml_models", "logistic_model.pkl")
encod = os.path.join("ml_models", "label_encoder.pkl")

with open(model_path, 'rb') as handle:
    model = pickle.load(handle)

with open(vec_path, 'rb') as handle:
    vec = pickle.load(handle)

with open(encod, 'rb') as handle:
    label_encoder = pickle.load(handle)

def load_model_and_predict(email_text):
    cleaned_email = preprocess_text(email_text)
    email_seq = vec.transform([cleaned_email])
    prediction = model.predict(email_seq)
    predicted_category = label_encoder.inverse_transform(prediction)
    return predicted_category[0]

st.header("Email Categorizer")
st.title("Kindly insert your email to categorize")
txt = st.text_area("insert your text here")
button = st.button("Categorize")
if button:
    result = load_model_and_predict(email_text=txt)
    st.write(f"The email you provided is categorized into: {result}")

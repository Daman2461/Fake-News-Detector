import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load NLTK stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Preprocess input text
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    return result

def predict(text):
    clean_text = preprocess(text)
    clean_joined = " ".join(clean_text)
    seq = tokenizer.texts_to_sequences([clean_joined])
    padded = pad_sequences(seq, maxlen=4405, padding='post', truncating='post')
    pred = model.predict(padded)
    return 'Real' if pred > 0.5 else 'Fake'

# Streamlit app interface
st.title('News Authenticity Checker Chatbot')

user_input = st.text_input('Enter the news text:')
if st.button('Predict'):
    result = predict(user_input)
    st.write(f'The news is: **{result}**')

st.write("This is a simple chatbot interface to check if the news is real or fake.")

import streamlit as st
import pickle
from preprocessor import transform_text

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email/SMS SPAM Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # steps to follow
    # 1 - preprocess
    transformed_sms = transform_text(input_sms)
    # 2 - vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3 - predict
    result = model.predict(vector_input)[0]
    # 4 - display
    if result ==1:
        st.header("SPAM")
    else:
        st.header("Not SPAM")

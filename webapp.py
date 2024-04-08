from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle
import os

file_path = r"Model.pkl"
file_path2= r"tfidfvectorizer.pkl"

with open(file_path, 'rb') as file:  
    Model = pickle.load(file)

with open(file_path2, 'rb') as file2:  
    feature_extraction = pickle.load(file2)

def predictor(input_text):
    
    input_text_features=feature_extraction.transform([input_text])
    prediction=Model.predict(input_text_features)

    if(prediction[0]==1):
        return 1
    else:
        return 0


def main():
    st.title('Message Classifier')
    message = st.text_input('Enter a message:')
    if st.button('Predict'):
        prediction = predictor(message)
        if(prediction==0):
            st.header("Spam")
        else:
            st.header("Ham")    

if __name__ == '__main__':
    main()




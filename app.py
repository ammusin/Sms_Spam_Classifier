import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.data.path.append('C:/Users/Sinchana/AppData/Roaming/nltk_data')
nltk.download('punkt_tab')  # This ensures punkt is available
nltk.download('stopwords')


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return" " .join(y)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("SMS CLASSIFIER")

input_sms=st.text_area("Enter the message")
if st.button("Classify"):

    Transformed_sms=transform_text(input_sms)
    vector_input=tfidf.transform([Transformed_sms])
    result=model.predict(vector_input)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")



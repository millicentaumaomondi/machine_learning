import streamlit as st
import pandas as pd

st.title('Machine Learning App 💻')

st.info('This is a machine learning app, it builds machine learning models')
st.write("**Raw Data**")
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
df 


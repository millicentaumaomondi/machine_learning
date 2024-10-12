import streamlit as st
import pandas as pd

st.title('Machine Learning App 💻')

st.info('This is a machine learning app, it builds machine learning models')
st.write("**Raw Data**")
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
df 

st.write("**X**")
X = df.drop('species',axis=1)
X

st.write("**y**")
y = df.species
y

with st.expander("Data Visualisation"):
  st.scatter_chart(data=df, x="bill_length_mm", y="body_mass_g",color='species')


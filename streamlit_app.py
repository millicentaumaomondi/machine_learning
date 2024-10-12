import streamlit as st
import pandas as pd

st.title('Machine Learning App 💻')

st.info('This is a machine learning app, it builds machine learning models')
with st.expander("Data"):
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

#Data Preparation
with st.sidebar:
  st.header("Input Features")
  "island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island = st.selectbox("Island",("Biscoe", "Dream", "Torgersen"))
  gender = st.selectbox("Gender",("male", "female"))
  bill_length_mm = st.slider("Bill length (mm)",32.1,59.6)

    #st.write("You selected:", option)


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Set up the background image using CSS
st.markdown(
    """
    <style>
    body {
        background-image: url('https://drive.google.com/file/d/134ZdZG806W2fDENYL2WxM9dL_1SawRq1/view?usp=sharing'); /* Replace with your direct image link */
        background-size: cover; /* Cover the entire background */
        background-position: center; /* Center the image */
        background-repeat: no-repeat; /* Do not repeat the image */
        height: 100vh; /* Full height */
        margin: 0; /* Remove default margin */
        padding: 0; /* Remove default padding */
    }
    .stApp {
        background: rgba(255, 255, 255, 0.8); /* Optional: Add a background with some transparency to the app for readability */
        min-height: 100vh; /* Ensure the app takes full height */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Machine Learning App 💻')

st.info('This app builds a random forest machine learning model to predict penguin species')

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.write(df)

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    st.write(X_raw)

    st.write('**y**')
    y_raw = df.species
    st.write(y_raw)

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input features
with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))

    # Create a DataFrame for the input features
    data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
    st.write('**Input penguin**')
    st.write(input_df)
    st.write('**Combined penguins data**')
    st.write(input_penguins)

# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    st.write(input_row)
    st.write('**Encoded y**')
    st.write(y)

# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'})

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),
               'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
               'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1),
             }, hide_index=True)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))

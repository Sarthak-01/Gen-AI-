import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data #st.cache_data is used to cache the results of the function so that it doesn't have to be re-run every time the app is refreshed
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target
    return df, data.target_names

df,data_target_names = load_data()
print(df.head())

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

st.sidebar.title('Input features')
sepal_length = st.sidebar.slider('Sepal length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider('Sepal width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider('Petal length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider('Petal width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

#take the input features from the sliders and make a prediction using the trained model
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = data_target_names[prediction][0]
st.write('### Predicted species based on input features:')
st.write(f' The Predicted species is: {predicted_species}')

#slider to select a sample from the dataset
#model is trained on the features (all columns except the last column 'species') and the target variable ('species')
#in model.fit(), the first argument is the features and the second argument is the target variable


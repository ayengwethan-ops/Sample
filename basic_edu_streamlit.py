import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

df = pd.read_csv('basic_edu.csv')
df = df[df["Code"] != 'OWID_WRL']

encoder = LabelEncoder()
df['Code'] = encoder.fit_transform(df['Code']) + 1

x = df.drop(columns=['Entity', 'Share of population with at least some basic education'])
y = df['Share of population with at least some basic education']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

with open('basic_edu.pkl', 'wb') as f:
    pickle.dump(lr, f)

with open('basic_edu.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

st.title("Basic Edu Prediction Mi")
st.write("Basic Edu Prediction Mi")

code = st.number_input("Code", min_value=1, max_value=24)
year = st.number_input("Year", min_value=1950, max_value=2100)
no_edu = st.number_input(
    "Share of population with no education (%)",
    min_value=0.0,
    max_value=100.0)

if st.button("Predict"):
    arr = np.array([[code, year, no_edu]])
    prediction = loaded_model.predict(arr)

    entity = encoder.inverse_transform([code - 1])[0]

    st.write("Country:", entity)
    st.write(
        "Predicted share of population with at least some basic education:",
        prediction[0])

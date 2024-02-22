import streamlit as st
import pickle
import numpy as np


def load_model():
    with open("saved_steps.pk1", "rb") as file:
        data = pickle.load(file)

    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

countries = {
    "United States",
    "India",
    "United Kingdom",
    "Germany",
    "Canada",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "Netherlands",
    "Poland",
    "Italy",
    "Russian Federation",
    "Sweden",
}

educations = {
    "Bachelor’s degree",
    "Master’s degree",
    "Post grad",
}


def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""")

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education", educations)
    experiance = st.slider("Years of Experiance", 0, 50, 3)

    ok = st.button("Calculate")
    if ok:
        X = np.array([[country, education, experiance]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])

        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

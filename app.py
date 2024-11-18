import streamlit as st
import numpy as np
import pickle
import pandas as pd

df = pd.read_csv('obesity.csv')
df.to_pickle('obesity.pkl')

st.header("Obesity Predictor")

st.write("Enter the info on health below")

Gender = st.radio("What gender are you?", "Male", "Female")
Age = st.slider("How old are you?", 0, 100, 0, 1)
Height = st.slider("How tall are you?", 0.00, 2.00, 0.00, 0.01)
Weight = st.slider("How much do you weigh?", 0, 150, 0, 1)
family_history_with_overweight = st.radio(
    "Does your family have a history with health problems", "yes", "no")
FAVC = st.radio("FAVC", "yes", "no")
FCVC = st.slider("FCVC", 0.0, 4.0, 0.0, 1.0)
NCP = st.slider("NCP", 0.0, 4.0, 0.0, 1.0)
CAEC = st.radio("CAEC", "Sometimes", "Always", "Frequently")
SMOKE = st.radio("SMOKE?", "yes", "no")
CH2O = st.slider("CH20", 0.0, 4.0, 0.0, 1.0)
SCC = st.radio("SCC", "yes", "no")
FAF = st.slider("FAF", 0.0, 3.0, 0.0, 1.0)
TUE = st.slider("TUE", 0.0, 3.0, 0.0, 1.0)
Calc = st.radio("Calc", "Sometimes", "Always", "Frequently")
MTRANS = st.radio("Mode of Transportation", "Public_Transportation",
                  "Walking", "Automobile", "Moterbike", "Bike")

ObesityLevel = np.array([[Gender, Age, Height, Weight,
                        family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, Calc, MTRANS]])

with open('obesity.pkl', 'OL') as f:
    model = pickle.load(f)

    prediction = model.predict(ObesityLevel)
    st.write(prediction)

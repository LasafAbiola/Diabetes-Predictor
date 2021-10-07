import streamlit as st
import pandas as pd
import pickle
from PIL import Image

st.title('Diabetes Predictor')

image = Image.open('dp_pic_2.jpeg')
st.image(image, use_column_width = True)

st.write("This app uses eight (8) inputs to predict if an individual is at risk of being diabetic or not using a classifier built on the PIMA Indian Dataset.") 
st.write("Use the form below to get started")

st.subheader('Training Data')
pima = pd.read_csv(r'diabetes.csv')

st.write(pima.head(10))
st.write(pima.describe())

st.subheader('Visualization')
st.bar_chart(pima)
#st.line_chart(pima)
#st.area_chart(pima)

rf_pickle = open('classifier.pkl', 'rb')
#map_pickle = open('output_pima.pkl', 'rb')

rf_model = pickle.load(rf_pickle)
#unique_pima_mapping = pickle.load(map_pickle)


#rf_pickle.close()
#map_pickle.close()

st.subheader('User Data')

pregnancies = st.number_input('Pregnancies (mnths)', min_value = 0)
glucose = st.number_input('Glucose', min_value = 0)
bp = st.number_input('Blood Pressure (mmHg)', min_value = 0)
skin_thickness = st.number_input('Skin Thickness (mm)', min_value = 0)
insulin = st.number_input('Insulin (muU/ml)', min_value = 0)
bmi = st.number_input('BMI', min_value = 0)
dpf = st.number_input('Diabetes Pedigree Function', min_value = 0)
age  = st.number_input('Age (yrs)', min_value = 10)
new_prediction = ''

st.write('The user inputs are {}'.format([pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]))


def prediction(pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age):
    prediction = rf_model.predict( 
        [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

    if prediction == 0:
        pred = 'No risk of diabetes'
    else:
        pred = 'Risk of diabetes. Please visit a hospital for further diagnosis'
    return pred

if st.button("Predict"):
    new_prediction = prediction(pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age)
    #prediction_diabetes = unique_pima_mapping[new_prediction][0]
    st.success('Your Diabetes result is: {}'.format(new_prediction))
    
    
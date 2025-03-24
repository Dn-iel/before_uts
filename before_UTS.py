import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load Model & Encoder
def load_model(filename):
    return joblib.load(filename)

def load_encoder(filename):
    return joblib.load(filename)

# Function to Encode User Input
def encode_input(user_input, label_encoders, onehot_encoder):
    encoded_input = user_input.copy()
    
    # Label Encoding untuk fitur kategori biner
    categorical_features = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    for feature in categorical_features:
        if feature in label_encoders:
            if user_input[feature] in label_encoders[feature].classes_:
                encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
            else:
                encoded_input[feature] = -1  # Default jika nilai tidak dikenal

    # One-Hot Encoding untuk fitur multi-kategori
    onehot_features = ["CAEC", "CALC", "MTRANS"]
    onehot_values = onehot_encoder.transform(pd.DataFrame([user_input], columns=onehot_features)).toarray()

    # Konversi dictionary ke list
    encoded_data = list(encoded_input.values())[:-3]  # Hapus fitur yang akan di-One-Hot Encode
    encoded_data.extend(onehot_values[0])  # Tambahkan hasil encoding

    return np.array(encoded_data).reshape(1, -1)

# Main Streamlit App
def main():
    st.title('Machine Learning App')
    st.info('This app will predict your obesity level!')

    # User Input
    user_input = {
        "Gender": st.selectbox('Gender', ('Male', 'Female')),
        "Age": st.slider('Age', min_value=0, max_value=75, value=21),
        "Height": st.slider('Height', min_value=0.50, max_value=2.00, value=1.62),
        "Weight": st.slider('Weight', min_value=10.0, max_value=140.0, value=64.0), 
        "family_history_with_overweight": st.selectbox('Family History With Overweight', ('yes', 'no')),
        "FAVC": st.selectbox('Frequent Consumption of High Caloric Food', ('yes', 'no')), 
        "FCVC": st.slider('Vegetable Consumption Frequency', min_value=0.0, max_value=3.0, value=2.0),
        "NCP": st.slider('Number of Meals per Day', min_value=0.0, max_value=3.0, value=2.0),
        "CAEC": st.selectbox('Consumption of Food Between Meals', ('no', 'Sometimes', 'Frequently', 'Always')), 
        "SMOKE": st.selectbox('Smoking Habit', ('yes', 'no')),
        "CH2O": st.slider('Daily Water Intake', min_value=0.0, max_value=3.0, value=2.0),
        "SCC": st.selectbox('Caloric Drinks Consumption', ('yes', 'no')),
        "FAF": st.slider('Physical Activity Frequency', min_value=0.0, max_value=3.0, value=2.0),
        "TUE": st.slider('Time Using Technology', min_value=0.0, max_value=3.0, value=2.0),
        "CALC": st.selectbox('Alcohol Consumption', ('no', 'Sometimes', 'Frequently', 'Always')), 
        "MTRANS": st.selectbox('Transportation Mode', ('Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'))
    }

    # Load Model & Encoders
    model_filename = 'before_UTS.pkl'
    label_encoder_filename = 'label_encoders.pkl'
    onehot_encoder_filename = 'onehot_encoder.pkl'

    model = load_model(model_filename)
    label_encoders = load_model(label_encoder_filename)  # Dictionary of LabelEncoders
    onehot_encoder = load_encoder(onehot_encoder_filename)

    # Encode input data
    encoded_input = encode_input(user_input, label_encoders, onehot_encoder)

    # Make Prediction
    prediction = model.predict(encoded_input)
    st.write("Model Prediction is: ", prediction[0])

if __name__ == '__main__':
    main()

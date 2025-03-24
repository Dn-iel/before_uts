import streamlit as st
import joblib

def load_model(filename):
  model = joblib.load(filename)
  return model

def predict_with_model(model, user_input):
  prediction = model.predict([user_input])
  return prediction[0]

def main():
  st.title('Machine Learning App')
  st.info('This app will predict your obesity level!')

    # Input data oleh user
  gender = st.selectbox('Gender', ('Male', 'Female'))
  age = st.slider('Age', min_value=0, max_value=75, value=21)
  height = st.slider('Height', min_value=0.50, max_value=2.00, value=1.62)
  weight = st.slider('Weight', min_value=10.0, max_value=140.0, value=64.0) 
  family_history_with_overweight = st.selectbox('Family History With Overweight', ('yes', 'no'))
  FAVC = st.selectbox('Frequent Consumption of High Caloric Food', ('yes', 'no')) 
  FCVC = st.slider('Vegetable Consumption Frequency', min_value=0.0, max_value=3.0, value=2.0)
  NCP = st.slider('Number of Meals per Day', min_value=0.0, max_value=3.0, value=2.0)
  CAEC = st.selectbox('Consumption of Food Between Meals', ('no', 'Sometimes', 'Frequently', 'Always')) 
  SMOKE = st.selectbox('Smoking Habit', ('yes', 'no'))
  CH2O = st.slider('Daily Water Intake', min_value=0.0, max_value=3.0, value=2.0)
  SCC = st.selectbox('Caloric Drinks Consumption', ('yes', 'no'))
  FAF = st.slider('Physical Activity Frequency', min_value=0.0, max_value=3.0, value=2.0)
  TUE = st.slider('Time Using Technology', min_value=0.0, max_value=3.0, value=2.0)
  CALC = st.selectbox('Alcohol Consumption', ('no', 'Sometimes', 'Frequently', 'Always')) 
  MTRANS = st.selectbox('Transportation Mode', ('Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'))

  # Mengumpulkan input data dalam list
  user_input = [gender, age, height, weight, family_history_with_overweight, 
                FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]

  # Memuat model
  model_filename = 'before_UTS.pkl'
  model = load_model(model_filename)

  prediction = predict_with_model(model, user_input)
  st.write("Model Prediction is : ", prediction)


if __name__ == '__main__':
    main()


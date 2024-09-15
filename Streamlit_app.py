import streamlit as st
import numpy as np

import os
import pickle

# Define the directory path where your pickle files are located
pickle_dir = 'C:/Users/ACER/OneDrive/Desktop/Python Projects/Crop Recommendation'

# Load multiple pickle files
models = {}
for filename in os.listdir(pickle_dir):
    if filename.endswith('.pkl'):
        model_path = os.path.join(pickle_dir, filename)
        model_name = filename.split('.')[0]  # Remove the .pkl extension
        models[model_name] = pickle.load(open(model_path, 'rb'))

# Print the loaded models
print("Loaded models:")
for model_name, model in models.items():
    print(f"  {model_name}")

# Extract the required models from the loaded models
rfc = models['model']
ms = models['minmaxscaler']
sc = models['standardscaler']

# Create a function to make predictions
def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)
    transformed_features = sc.transform(transformed_features)
    prediction = rfc.predict(transformed_features)
    return prediction[0]

# Create a dictionary to map crop numbers to names
crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

# Create a Streamlit app
st.title("Crop Recommendation System")
st.write("Enter the following parameters to get a crop recommendation:")

N = st.number_input("Nitrogen (N)")
P = st.number_input("Phosphorus (P)")
k = st.number_input("Potassium (K)")
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Get Recommendation"):
    predict = recommendation(N, P, k, temperature, humidity, ph, rainfall)
    if predict in crop_dict:
        crop = crop_dict[predict]
        st.write("The best crop to cultivate is: **{}**".format(crop))
    else:
        st.write("Sorry, unable to recommend a proper crop for this environment")
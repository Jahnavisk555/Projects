import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pickle
import numpy as np

# Load the dataset and train the model
dataset = pd.read_csv("diabetes.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train, Y_train)

# Save the model
with open('knn_diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

# Load the saved model
loaded_model = pickle.load(open('knn_diabetes_model.pkl', 'rb'))

# Function for Diabetes Prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    st.title('Diabetes Prediction Web App')
    
    pregnancies = st.text_input('Pregnancies', 0)  # Default value set to 0
    glucose = st.text_input('Glucose', 0)
    blood_pressure = st.text_input('Blood Pressure', 0)
    skin_thickness = st.text_input('Skin Thickness', 0)
    insulin = st.text_input('Insulin', 0)
    bmi = st.text_input('BMI', 0.0)  # Default value set to 0.0
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function', 0.0)  # Default value set to 0.0
    age = st.text_input('Age', 0)  # Default value set to 0
    
    result = ''
    
    if st.button('Predict Diabetes'):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        input_data = [float(val) for val in input_data]  # Convert inputs to float
        prediction = diabetes_prediction(input_data)
        if prediction[0] == 0:
            result = 'No Diabetes'
        else:
            result = 'Diabetes'
    
    st.write(f'Prediction: {result}')

if __name__ == '__main__':
    main()

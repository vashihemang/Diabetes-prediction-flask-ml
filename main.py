from flask import Flask,render_template,request
import pandas as pd
import pickle as pk
model = pk.load(open(r"C:\Users\vashi\Projects\healthcare\diabetes_stack_model.pkl", "rb"))
encode = pk.load(open(r"C:\Users\vashi\Projects\healthcare\label_encoders.pkl", "rb"))

app = Flask(__name__,template_folder="Template")

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/Prediction",methods=["GET","POST"])
def Predict(): 
    if request.method=="POST":
        gender = request.form["gender"]
        Age = int(request.form["Age"])
        Hypertension=int(request.form["Hypertension"])
        Heart_Disease=int(request.form["Heart_disease"])
        Smoking_History=request.form["Smoking_history"]
        BMI12 = float(request.form["Bmi"])
        HbA1c_level = float(request.form["HbA1c_level"])
        Blood_glucose_level=int(request.form["Blood_glucose_level"])
        
        gender_encoder=encode["gender"].transform([gender])[0]
        Smoking_History_encode=encode["smoking_history"].transform([Smoking_History])[0]



        features =[[gender_encoder,Age,Hypertension,Heart_Disease,Smoking_History_encode,BMI12,HbA1c_level,Blood_glucose_level]]
        Predictions = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        result = "Diabetic" if Predictions == 1 else "No Diabetic"
        return render_template("predict.html",result=result,prob=prob)
    return render_template("predict.html",result=None, prob=None)
if __name__=="__main__":
    app.run(debug=True)





 
# import numpy as np


# import streamlit as st 

# st.title("Diabetes Prediction")
# gender = st.selectbox("Gender :",["Male", "Female"])
# Age=st.number_input("Age :",min_value=0,max_value=100,step=1)
# Hypertension=st.number_input("Hypertension :",min_value=0,step=1)
# Heart_disease=st.number_input("Heart_disease :",min_value=0,step=1)
# Smoking_history=st.selectbox("Smoking_history :", ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
# Bmi=st.number_input("Bmi :",min_value=0)
# HbA1c_level=st.number_input('HbA1c_level:',min_value=0)
# Blood_glucose_level=st.number_input('Blood_glucose_level :',min_value=0)


# gender_encoder = encode["gender"].transform([gender])[0]
# Smoking_history_encoder = encode["smoking_history"].transform([Smoking_history])[0]


# input_data = np.array([[gender_encoder,Age,Hypertension,Heart_disease,Smoking_history_encoder,Bmi,HbA1c_level,Blood_glucose_level]])


# if st.button("predict"):
#     prediction = model.predict(input_data)
#     if prediction[0] == 1:
#         st.error("Patient is likely to have Diabetes")
#         prob = model.predict_proba(input_data)[0][1]
#         st.write(f"Diabetes Probability: {prob:.2%}")
      
#     else:
#         st.success("Patient is NOT likely to have Diabetes")    

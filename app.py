import streamlit as st 
import joblib 

linear_regression = joblib.load(open(r"C:\Users\vashi\AIOPD\Assignment-1 ai\linear_regression.pkl","rb"))
logistic_regression = joblib.load(open(r"C:\Users\vashi\AIOPD\Assignment-1 ai\logisticregression.pkl","rb"))
random_forest= joblib.load(open(r"C:\Users\vashi\AIOPD\Assignment-1 ai\randomforest.pkl","rb"))
k_n_n= joblib.load(open(r"C:\Users\vashi\AIOPD\Assignment-1 ai\k_n_n.pkl","rb"))


st.title("prediction_model")

tab1,tab2,tab3=st.tabs(["tab1","tab2","tab3"])

#  age_salary_prediction.

with tab1:
    st.subheader("age_salary_prediction")
    age = st.number_input("Age :",min_value=0,max_value=60,key="age_salary")
    
    if st.button("predict",key="predict_salary"): 
        prediction=linear_regression.predict([[age]])
        st.success(f"prediction :{ prediction[0]:.2f}")

# customer purchase data with logistic regression.

with tab2:
    st.subheader("customer purchase data with logistic regression")        
    age1 = st.number_input("Age :",min_value=0,max_value=60,key="age_customer")
    salary = st.number_input("Salary :",min_value=0)
    if st.button("predict",key="predict_purchase"): 
        prediction1=logistic_regression.predict([[age1,salary]])
        st.success(f"prediction : {prediction1[0]}")

# customer purchase data with k_n_n.

with tab3:  
    st.subheader("customer purchase data with k_n_n")
    sepal_length = st.number_input("Sepal Length", min_value=0.0)
    sepal_width  = st.number_input("Sepal Width", min_value=0.0)
    petal_length = st.number_input("Petal Length", min_value=0.0)
    petal_width  = st.number_input("Petal Width", min_value=0.0)
    if st.button("predict",key="predict3"):
        prediction2=k_n_n.predict([[sepal_length,sepal_width,petal_length,petal_width]])
        st.success(f"prediction : {prediction2[0]}")






# # Q4 Linear Model Prediction with gradio.

# import gradio as gr
# import joblib
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")
# model = joblib.load(open("model1.pkl","rb"))
# def predict(Hours_Studied,Attendance):
#     prediction = model.predict(np.array([[Hours_Studied,Attendance]]))
#     return prediction[0]

# # Gradio interface
# interface = gr.Interface(
#     fn=predict,
#     inputs=[gr.Number(label="Enter a Hours_Studied:"),gr.Number(label="Enter a Attendance:")], 	
#     outputs=gr.Number(label="Prediction"),
#     title="Linear Model Predictor",
#     description="Enter a number to get prediction using Linear Regression"
# )

# # Launch app
# interface.launch()
## * Running on local URL:  http://127.0.0.1:7868
# #* To create a public link, set `share=True` in `launch()`.

# # Q5  Decision tree with gradio.

# import gradio as gr
# import joblib
# import numpy as np
# D_f= joblib.load(open("D_f.pkl","rb"))
# def prediction(Hours_Studied,Attendance,Marks): 
#     prediction1 = D_f.predict([[Hours_Studied,Attendance,Marks]])
#     return prediction1[0]


# interface=gr.Interface(
#     fn=prediction,
#     inputs=[gr.Number(label="Enter a Hours_Studied:"),
#             gr.Number(label="Enter a Attendance:"),
#             gr.Number(label="Enter a Marks:")], 
#     outputs="text",
#     title="Decision tree Model",
#     description="Enter a number to get prediction using Decision tree"
# )
# interface.launch()
# #* Running on local URL:  http://127.0.0.1:7870
# #* To create a public link, set `share=True` in `launch()`.


       
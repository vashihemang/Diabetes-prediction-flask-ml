# Diabetes Prediction Flask ML

A Machine Learning web application built with Flask that predicts whether a person is diabetic based on medical input parameters.

## 🚀 Project Overview

This project uses a trained Machine Learning model integrated with a Flask web application to predict diabetes risk. Users can enter health-related details through a web interface, and the model returns a prediction result instantly.

The application is simple, lightweight, and suitable for beginners learning:
  
  * Machine Learning deployment
  * Flask web development
  * Model serialization with Pickle
  * End-to-end ML projects

## 📸 Application Preview

<img width="2814" height="1536" alt="diabetes" src="https://github.com/user-attachments/assets/32d67cca-1db4-491d-9a45-ea2c659bad36" />


## 📌 Features

* User-friendly web interface
* Diabetes prediction using ML model
* Flask backend integration
* Real-time prediction results
* Easy deployment and customization

## 🛠️ Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* HTML/CSS
* Pickle

## 📂 Project Structure

```bash
Diabetes-prediction-flask-ml/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── model/
│   └── diabetes_model.pkl
│
├── app.py
├── requirements.txt
├── diabetes.csv
└── README.md
```

## 📊 Dataset

The project uses the PIMA Indians Diabetes Dataset containing medical predictor variables such as:

  * Pregnancies
  * Glucose Level
  * Blood Pressure
  * Skin Thickness
  * Insulin
  * BMI
  * Diabetes Pedigree Function
  * Age

  * 
## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Diabetes-prediction-flask-ml.git
```

### 2. Navigate to Project Directory

```bash
cd Diabetes-prediction-flask-ml
```

### 3. Create Virtual Environment (Optional)

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
python app.py
```

The application will start on:

```bash
http://127.0.0.1:5000/
```

## 🧠 Machine Learning Model

The prediction model is trained using supervised learning algorithms from Scikit-learn.

Possible algorithms:

* Logistic Regression
* Random Forest
* Decision Tree
* Support Vector Machine

The trained model is saved using Pickle for deployment.

```bash
screenshots/homepage.png
```
## 📈 Future Improvements

* Add user authentication
* Deploy on Heroku/AWS/Render
* Improve UI/UX
* Add API support
* Use Deep Learning models
* Add prediction probability score





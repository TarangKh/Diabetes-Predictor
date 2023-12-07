from flask import Flask, request, app, render_template

import pandas as pd
import numpy as np
import pickle

application = Flask(__name__)
app = application
scaler = pickle.load(open("Model/standardScaler.pkl", "rb"))
model = pickle.load(open("Model/DiabetesPrediction.pkl", "rb"))

# route for home page
@app.route("/")
def index():
    return render_template("index.html")

# route for single data prediction point
@app.route("/predictdata", methods = ["GET", "POST"])
def predict_datapoint() :
    result = ""
    if request.method == "POST" :
        pregnancies = int(request.form.get("Pregnancies"))
        glucose = float(request.form.get("Glucose"))
        bp = float(request.form.get("BloodPressure"))
        sk = float(request.form.get("SkinThickness"))
        insulin = float(request.form.get("Insulin"))
        bmi = float(request.form.get("BMI"))
        dpf = float(request.form.get("DiabetesPedigreeFunction"))
        age = int(request.form.get("Age"))

        new_data = scaler.transform(pregnancies, glucose, bp, sk, insulin, bmi, dpf, age)
        predict = model.predict(new_data)
        if (predict[0] == 1) :
            result = "Diabetic"
        else :
            result = "Non-diabetic"
        
        return render_template("single_prediction.html", result = result)
    else :
        return render_template("home.html")


if __name__=="__main__":
    app.run(host="0.0.0.0")

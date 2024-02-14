import pickle
import pandas 
import numpy 
from flask import Flask,render_template,request

application = Flask(__name__)
app = application

scaler = pickle.load(open(r"model/scaler.pkl","rb"))
model = pickle.load(open(r"model/model.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_data():
    result = ""

    if request.method == "POST":
        pregnancies = int(request.form.get("Pregnancies"))
        glucose =  float(request.form.get("Glucose"))
        bloodpressure = float(request.form.get("BloodPressure"))
        skinthickness = float(request.form.get("SkinThickness"))
        insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        diabetespedigree = float(request.form.get("DiabetesPedigreeFunction"))
        age = float(request.form.get("Age"))

        new_data = scaler.transform([[pregnancies,glucose,bloodpressure,skinthickness,insulin,BMI,diabetespedigree,age]])
        predict = model.predict(new_data)

        if predict[0] == 0:
            result = "Non-Diabetic"
        else:
            result = "Diabetic"

        return render_template("prediction.html",result=result)
    
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")

    
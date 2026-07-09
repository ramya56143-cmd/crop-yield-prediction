from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#  Load model
model = pickle.load(open("crop_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        fertilizer = float(request.form['fertilizer'])
        pesticide = float(request.form['pesticide'])

        features = np.array([[rainfall, temperature, fertilizer, pesticide]])

        prediction = model.predict(features)[0]

        result = f"🌾 Predicted Crop Yield: {round(prediction, 2)} tons/hectare"

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text=" Invalid Input")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=1000)
    

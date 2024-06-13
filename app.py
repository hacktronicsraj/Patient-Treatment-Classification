from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_treatment():
    # Get input features from the form
    haematocrit = float(request.form.get("HAEMATOCRIT"))
    haemoglobins = float(request.form.get("HAEMOGLOBINS"))
    erythrocyte = float(request.form.get("ERYTHROCYTE"))
    leucocyte = float(request.form.get("LEUCOCYTE"))
    thrombocyte = float(request.form.get("THROMBOCYTE"))
    age = float(request.form.get("Age"))
    sex = request.form.get("Sex")

    # Label encoding for gender
    if sex.upper() == "M":
        sex = 1
    elif sex.upper() == "F":
        sex = 0
    else:
        return render_template("index.html", result="Invalid Gender input")

    # Create a feature array
    features = np.array(
        [haematocrit, haemoglobins, erythrocyte, leucocyte, thrombocyte, age]
    ).reshape(1, -1)

    # Scale numerical features
    scaled_features = scaler.transform(features)

    # Combine scaled features with gender
    final_features = np.append(scaled_features, sex).reshape(1, -1)

    # Make prediction using the loaded model
    prediction = model.predict(final_features)[0]

    # Interpret prediction result
    if prediction == 1:
        result = "Incare Patient"
    else:
        result = "Outcare Patient"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)

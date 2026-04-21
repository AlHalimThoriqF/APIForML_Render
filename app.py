import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("linear_regression_model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values in the correct order
        age = float(request.form.get("Age"))
        gender = request.form.get("Gender")
        blood_type = request.form.get("Blood Type")
        medical_condition = request.form.get("Medical Condition")

        # Create dataframe with the same structure as training data
        input_data = pd.DataFrame(
            {
                "Age": [age],
                "Gender": [gender],
                "Blood Type": [blood_type],
                "Medical Condition": [medical_condition],
            }
        )

        # Transform the input using the same transformer as training
        transformed_features = transformer.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_features)[0]

        return render_template(
            "index.html",
            prediction_text="Predicted Billing Amount: ${:.2f}".format(prediction),
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
        )


if __name__ == "__main__":
    flask_app.run(debug=True)

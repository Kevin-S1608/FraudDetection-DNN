from flask import Flask, render_template, request
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --------------------------------
# Load Model and Scaler
# --------------------------------

model = load_model("fraud_model.keras")
scaler = joblib.load("scaler.pkl")


# --------------------------------
# Model Performance
# --------------------------------

accuracy = 89.80
precision = 15.57
recall = 95.00
f1_score = 26.76


# --------------------------------
# Confusion Matrix
# --------------------------------

true_negative = 897
false_positive = 103
false_negative = 1
true_positive = 19


# --------------------------------
# Transaction History
# --------------------------------

transaction_history = []


# --------------------------------
# Quick Test Samples
# --------------------------------

legitimate_sample = {
    "time": 51707,
    "amount": 19.01,
    "v1": 0.25149367,
    "v2": -0.96808359,
    "v3": 1.49805028,
    "v4": -2.56276786,
    "v5": 0.02034271,
    "v6": 0.50734434
}


fraud_sample = {
    "time": 112962,
    "amount": 20.17,
    "v1": -0.78770894,
    "v2": 0.24086921,
    "v3": 0.59800414,
    "v4": 0.45773467,
    "v5": -0.73853469,
    "v6": 0.75583384
}


# --------------------------------
# Home Route
# --------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probability = None
    confidence = None
    status = "Ready for analysis"

    if request.method == "POST":

        try:

            # ----------------------------
            # Get input values
            # ----------------------------

            time = float(request.form["time"])
            amount = float(request.form["amount"])
            v1 = float(request.form["v1"])
            v2 = float(request.form["v2"])
            v3 = float(request.form["v3"])
            v4 = float(request.form["v4"])
            v5 = float(request.form["v5"])
            v6 = float(request.form["v6"])


            # ----------------------------
            # Create DataFrame
            # ----------------------------

            transaction = pd.DataFrame(
                [[
                    time,
                    amount,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6
                ]],
                columns=[
                    "Time",
                    "Amount",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "V5",
                    "V6"
                ]
            )


            # ----------------------------
            # Scale Input
            # ----------------------------

            transaction_scaled = scaler.transform(transaction)


            # ----------------------------
            # Prediction
            # ----------------------------

            prediction_probability = model.predict(
                transaction_scaled,
                verbose=0
            )[0][0]


            probability = round(
                float(prediction_probability) * 100,
                2
            )


            # ----------------------------
            # Classification
            # ----------------------------

            if prediction_probability >= 0.5:

                result = "FRAUDULENT TRANSACTION"

                confidence = round(
                    float(prediction_probability) * 100,
                    2
                )

            else:

                result = "LEGITIMATE TRANSACTION"

                confidence = round(
                    (1 - float(prediction_probability)) * 100,
                    2
                )


            # ----------------------------
            # Add to History
            # ----------------------------

            transaction_history.insert(
                0,
                {
                    "time": time,
                    "amount": amount,
                    "prediction": result,
                    "probability": probability,
                    "confidence": confidence
                }
            )


            # Keep latest 10

            if len(transaction_history) > 10:
                transaction_history.pop()


            status = "Analysis completed"


        except Exception as e:

            result = "INVALID INPUT"
            probability = 0
            confidence = 0
            status = "Please enter valid transaction values."

            print("Error:", e)


    return render_template(
        "index.html",

        result=result,
        probability=probability,
        confidence=confidence,
        status=status,

        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,

        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,

        transaction_history=transaction_history,

        legitimate_sample=legitimate_sample,
        fraud_sample=fraud_sample
    )


# --------------------------------
# Run Application
# --------------------------------

if __name__ == "__main__":
    app.run(debug=True)
import pickle
import pandas as pd
import numpy as np # Import numpy if you're not already
from flask import Flask, request, jsonify

with open('log_reg.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

prediction_mapping = {
    0: "Abnormal",
    1: "Normal"
}

def predict(features_dict):
    feature_names = [
        "pelvic_incidence",
        "_pelvic_tilt",
        "_lumbar_lordosis_angle",
        "_sacral_slope",
        "_pelvic_radius",
        "_grade_of_spondylolisthesis"
    ]

    features_df = pd.DataFrame([features_dict], columns=feature_names)

    # Convert the DataFrame to a NumPy array to match the fitting style
    preds = model.predict(features_df.to_numpy())[0] # <--- Change here

    descriptive_prediction = prediction_mapping[preds]

    return descriptive_prediction


app = Flask('Spine-prediction')

@app.route('/predict', methods = ['POST'])
def predict_endpoint():
    diag = request.get_json()

    pred = predict(diag)

    result = {"Patient_is": pred}

    return jsonify(result)

# ... (rest of your predict.py code) ...

if __name__ == '__main__': # <-- Corrected: two underscores for __name__ and __main__
    app.run(debug=True, host='0.0.0.0', port=9696)

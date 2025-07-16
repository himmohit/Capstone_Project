from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

# Define the exact features used during training
features_to_use = [
    'years_of_insurance_with_us',
    'regular_checkup_last_year',
    'adventure_sports',
    'daily_avg_steps',
    'age',
    'avg_glucose_level',
    'bmi',
    'Year_last_admitted',
    'weight',
    'weight_change_in_last_one_year',
    'fat_percentage',
    'covered_by_any_other_company_Y'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract all features in correct order
        input_values = []
        for feature in features_to_use:
            val = float(request.form[feature])
            input_values.append(val)

        input_data = np.array([input_values])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=round(prediction, 2))

    except KeyError as e:
        missing_field = str(e)
        return f"Missing form input: {missing_field}", 400
    except Exception as e:
        return f"Error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)

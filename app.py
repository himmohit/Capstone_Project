from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
import csv
import os
from datetime import datetime
import pandas as pd

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
        prediction = round(prediction, 2)

        # Log prediction with timestamp
        log_file = 'prediction_logs.csv'
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp'] + features_to_use + ['prediction'])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + input_values + [prediction])

        return render_template('index.html', prediction=prediction)

    except KeyError as e:
        missing_field = str(e)
        return f"Missing form input: {missing_field}", 400
    except Exception as e:
        return f"Error occurred: {e}", 500

# ✅ Download logs as a CSV file
@app.route('/download_logs')
def download_logs():
    log_file = 'prediction_logs.csv'
    if os.path.exists(log_file):
        return send_file(log_file, as_attachment=True)
    else:
        return "No log file found yet.", 404

# ✅ View logs in table format
@app.route('/view_logs')
def view_logs():
    log_file = 'prediction_logs.csv'
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        return render_template('view_logs.html', tables=[df.to_html(classes='table table-bordered')])
    else:
        return "No log file found yet.", 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)

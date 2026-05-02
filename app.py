from flask import Flask, request, render_template
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        hours_studied = float(request.form.get('hours_studied', 0))
        attendance    = float(request.form.get('attendance', 0))
        sleep         = float(request.form.get('sleep', 0))

        features = np.array([[hours_studied, attendance, sleep]])
        prediction = model.predict(features)
        output = round(float(prediction[0]), 2)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Student Score: {output}'
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f'Prediction failed: {str(e)}'
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
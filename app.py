from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('logistic_regression.joblib')
feature_columns = joblib.load("feature_columns.pkl")

default_values = {
    'contact': 'cellular',
    'month': 'nov',
    'day_of_week': 'thu',
    'campaign': 2,
    'pdays': 1,
    'previous': 0,
    'poutcome': 'nonexistent',
    'emp.var.rate': -1.1,
    'cons.price.idx': 94.767,
    'cons.conf.idx': -50.8,
    'euribor3m': 4.857,
    'nr.employed': 4963.6
}

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        client_data = {**default_values, **data}
        client_df = pd.DataFrame([client_data])

        for col in feature_columns:
            if col not in client_df.columns:
                client_df[col] = 0 

        client_df = client_df[feature_columns]
        prediction = model.predict(client_df)
        result = "Subscribe" if prediction[0] == 1 else "Do not subscribe"

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
    # app.run(debug=True, port=5001)

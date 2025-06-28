import os
import time
import psutil
import logging
import threading
import pandas as pd
import numpy as np
import requests
from flask import Flask, request, render_template
from prometheus_client import Counter, Gauge, Summary, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# ========== CONFIG ==========
logging.basicConfig(level=logging.INFO)
MLFLOW_SERVE_URL = "http://localhost:5002/invocations"

REQUIRED_COLUMNS = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

app = Flask(__name__)

# ========== PROMETHEUS METRICS ==========
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP request', ['endpoint', 'method'])
FAILED_REQUESTS = Counter('http_requests_failed', 'Jumlah gagal', ['endpoint'])
PREDICTION_SUCCESS = Counter('predictions_successful', 'Prediksi sukses')
PREDICTION_FAILED = Counter('predictions_failed', 'Prediksi gagal')
PREDICTION_LABEL = Counter('prediction_output', 'Prediksi per label', ['label'])
INFERENCE_TIME = Summary('prediction_latency_seconds', 'Waktu prediksi')
CSV_SIZE = Gauge('csv_file_size_bytes', 'Ukuran CSV')
CSV_ROWS = Gauge('csv_row_count', 'Jumlah baris CSV')

CPU = Gauge('cpu_usage_percent', 'Penggunaan CPU')
RAM = Gauge('memory_usage_bytes', 'Penggunaan RAM')

# ========== MONITORING BACKGROUND ==========
def track_system():
    while True:
        CPU.set(psutil.cpu_percent())
        RAM.set(psutil.virtual_memory().used)
        time.sleep(5)

threading.Thread(target=track_system, daemon=True).start()

# Prometheus route
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

@app.before_request
def count_request():
    REQUEST_COUNT.labels(request.endpoint, request.method).inc()

# ========== ROUTES ==========
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        data = {col: request.form[col] for col in REQUIRED_COLUMNS}
        for col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
            data[col] = float(data[col])
        data['FastingBS'] = int(data['FastingBS'])

        df = pd.DataFrame([data])

        start = time.time()
        response = requests.post(
            MLFLOW_SERVE_URL,
            json={"columns": REQUIRED_COLUMNS, "data": df.values.tolist()}
        )
        elapsed = time.time() - start

        if response.status_code != 200:
            raise Exception(f"MLflow serving gagal: {response.text}")

        prediction = response.json()[0]

        INFERENCE_TIME.observe(elapsed)
        PREDICTION_SUCCESS.inc()
        PREDICTION_LABEL.labels(label=str(prediction)).inc()

        result = "Positif" if prediction == 1 else "Negatif"
        logging.info(f"Prediksi berhasil → {result} | Durasi: {elapsed:.3f}s")
        return render_template('index.html', prediction_text=f"Hasil: {result}")
    except Exception as e:
        logging.warning(f"❌ Error prediksi form: {e}")
        PREDICTION_FAILED.inc()
        FAILED_REQUESTS.labels('/predict').inc()
        return render_template('index.html', prediction_text=f"Terjadi error: {e}")

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return "❌ File tidak ditemukan.", 400

        content = file.read()
        CSV_SIZE.set(len(content))
        file.seek(0)

        df = pd.read_csv(file)
        CSV_ROWS.set(len(df))

        if set(REQUIRED_COLUMNS) - set(df.columns):
            return "❌ Kolom CSV tidak lengkap", 400

        start = time.time()
        response = requests.post(
            MLFLOW_SERVE_URL,
            json={"columns": REQUIRED_COLUMNS, "data": df[REQUIRED_COLUMNS].values.tolist()}
        )
        elapsed = time.time() - start

        if response.status_code != 200:
            raise Exception(f"MLflow serving error: {response.text}")

        y_pred = response.json()
        df['Predicted_HeartDisease'] = y_pred

        INFERENCE_TIME.observe(elapsed)
        PREDICTION_SUCCESS.inc(len(y_pred))

        for label in np.unique(y_pred):
            PREDICTION_LABEL.labels(label=str(label)).inc(np.sum(np.array(y_pred) == label))

        return render_template('index.html', tables=df.to_html(classes="table table-bordered", index=False))
    except Exception as e:
        logging.error(f"❌ CSV Prediction Error: {e}")
        PREDICTION_FAILED.inc()
        FAILED_REQUESTS.labels('/predict_csv').inc()
        return f"Error: {e}", 500

# ========== RUN ==========
if __name__ == '__main__':
    print("🧠 Sistem gabungan Flask + MLflow Serving aktif di http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)

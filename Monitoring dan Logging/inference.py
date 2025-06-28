import os
import time
import logging
import traceback
import pandas as pd
import numpy as np
import requests
import threading
import joblib
from flask import Flask, request, render_template, Response
from prometheus_client import (
    Counter, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
)

# ========== KONFIGURASI ==========
logging.basicConfig(level=logging.INFO)
MLFLOW_URL = "http://localhost:5002/invocations"

PREPROCESSOR_PATH = r"D:\00 - DATA\LASKAR AI\Submission Membangun Sistem Machine Learning\SMSML_Idha_Kurniawati\Monitoring dan Logging\model\preprocessor_pipeline.joblib"
preprocessor = joblib.load(PREPROCESSOR_PATH)

REQUIRED_COLUMNS = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

app = Flask(__name__)
start_time = time.time()

# ========== METRIK PROMETHEUS ==========
REQUEST_TOTAL = Counter('inference_requests_total', 'Jumlah semua request masuk', ['method', 'endpoint'])
PREDICTION_SUCCESS = Counter('inference_success_total', 'Jumlah prediksi sukses')
PREDICTION_FAILED = Counter('inference_failed_total', 'Jumlah prediksi gagal')
PREDICTION_BY_LABEL = Counter('inference_by_label', 'Jumlah prediksi per label', ['label'])

INFERENCE_LATENCY = Summary('inference_latency_seconds', 'Durasi inferensi')
HTTP_DURATION = Summary('http_request_duration_seconds', 'Durasi waktu request HTTP')
CSV_ROW_COUNT = Gauge('csv_input_row_count', 'Jumlah baris dalam file CSV')
CSV_FILE_SIZE = Gauge('csv_input_file_size_bytes', 'Ukuran file CSV dalam byte')
THREAD_COUNT = Gauge('active_thread_count', 'Jumlah thread aktif')
FLASK_UPTIME = Gauge('flask_app_uptime_seconds', 'Lama aplikasi aktif dalam detik')

# ========== MONITORING LATAR BELAKANG ==========
def monitor_background():
    while True:
        THREAD_COUNT.set(threading.active_count())
        FLASK_UPTIME.set(time.time() - start_time)
        time.sleep(5)

threading.Thread(target=monitor_background, daemon=True).start()

@app.before_request
def count_requests():
    REQUEST_TOTAL.labels(request.method, request.path).inc()
    request.start_time = time.time()

@app.after_request
def track_http_time(response):
    duration = time.time() - request.start_time
    HTTP_DURATION.observe(duration)
    return response

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        form_data = {col: request.form[col] for col in REQUIRED_COLUMNS}
        for col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
            form_data[col] = float(form_data[col])
        form_data['FastingBS'] = int(form_data['FastingBS'])

        df = pd.DataFrame([form_data])

        # Transformasi dan perbaikan nama kolom
        transformed = preprocessor.transform(df)
        columns = preprocessor.get_feature_names_out()
        columns = [c.replace("num__", "").replace("cat__", "") for c in columns]
        df_transformed = pd.DataFrame(transformed, columns=columns)

        start = time.time()
        response = requests.post(
            MLFLOW_URL,
            json={"dataframe_split": df_transformed.to_dict(orient="split")}
        )
        elapsed = time.time() - start

        if response.status_code != 200:
            raise Exception(f"MLflow error: {response.text}")

        response_json = response.json()
        logging.info(f"[DEBUG] Respon MLflow: {response_json}")
        logging.info(f"[DEBUG] JSON terkirim: {df_transformed.to_dict(orient='split')}")


        # Coba ambil langsung hasil array
        if isinstance(response_json, list):
            result = response_json[0]
        elif isinstance(response_json, dict) and "predictions" in response_json:
            result = response_json["predictions"][0]
        else:
            raise ValueError(f"Format response tidak dikenali: {response_json}")

        INFERENCE_LATENCY.observe(elapsed)
        PREDICTION_SUCCESS.inc()
        PREDICTION_BY_LABEL.labels(label=str(result)).inc()

        label_text = "🫀 Positif" if result == 1 else "✅ Negatif"
        logging.info(f"[PREDIKSI] Hasil: {label_text} | Waktu: {elapsed:.3f}s")

        return render_template("index.html", prediction_text=f"Hasil Prediksi: {label_text}")

    except Exception as e:
        logging.error("❌ Gagal melakukan prediksi")
        logging.error(traceback.format_exc())  # debug
        PREDICTION_FAILED.inc()
        return render_template("index.html", prediction_text="❌ Terjadi kesalahan. Silakan cek log.")

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            raise Exception("File CSV tidak ditemukan")

        content = file.read()
        CSV_FILE_SIZE.set(len(content))
        file.seek(0)

        df = pd.read_csv(file)
        CSV_ROW_COUNT.set(len(df))

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise Exception(f"Kolom tidak lengkap: {', '.join(missing)}")

        transformed = preprocessor.transform(df)
        columns = preprocessor.get_feature_names_out()
        columns = [c.replace("num__", "").replace("cat__", "") for c in columns]
        df_transformed = pd.DataFrame(transformed, columns=columns)

        response = requests.post(
            MLFLOW_URL,
            json={"dataframe_split": df_transformed.to_dict(orient="split")}
        )

        if response.status_code != 200:
            raise Exception(response.text)

        y_pred = response.json()
        df["Predicted_HeartDisease"] = y_pred

        for label in np.unique(y_pred):
            count = np.sum(np.array(y_pred) == label)
            PREDICTION_BY_LABEL.labels(label=str(label)).inc(count)
        PREDICTION_SUCCESS.inc(len(y_pred))

        return render_template("index.html", tables=df.to_html(classes="table table-bordered", index=False))

    except Exception as e:
        logging.error(f"❌ Gagal prediksi CSV: {e}")
        PREDICTION_FAILED.inc()
        return f"Terjadi kesalahan saat memproses CSV: {e}", 500

# ========== JALANKAN APLIKASI ==========
if __name__ == '__main__':
    print("🚀 Inference + Monitoring aktif di http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)

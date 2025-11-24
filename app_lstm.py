from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
import pickle
import requests

app = Flask(__name__)


LARAVEL_PREDIKSI_ENDPOINT = "API-web"
LARAVEL_KLASIFIKASI_UDARA_ENDPOINT = "API-web"

# --- Konfigurasi ---
ROLLING_CSV_PATH = 'Model_Prediksi/prediksi_1/rolling_window_1.csv'
MODEL_PATH = 'Model_Prediksi/prediksi_1/LSTMfull.keras'
SCALER_PATH = 'Model_Prediksi/prediksi_1/scalerfull.pkl'
MODEL_KLASIFIKASI_PATH = 'Model_Klasifikasi/klasifikasi_1/model_klasifikasi_1.pkl'
ROLLING_WINDOW_SIZE = 840
N_INPUT = 168
N_OUTPUT = 24
MIN_TRAIN = N_INPUT + N_OUTPUT
PREDICTION_FEATURES = ['Suhu (C)', 'Kelembaban (%)', 'Tekanan Udara (hPa)', 'Kecepatan Angin (m/s)']
TIME_FEATURES = ['sin_jam', 'cos_jam', 'sin_hari', 'cos_hari', 'sin_bulan', 'cos_bulan']
ALL_FEATURES = PREDICTION_FEATURES + TIME_FEATURES
KLASIFIKASI_FEATURES = PREDICTION_FEATURES


# --- Validasi input sesuai app_4.py ---
def validate_input(data, required_fields):
    if not data:
        return False, "No data provided"
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Field berikut wajib ada: {', '.join(missing)}"
    return True, ""

# --- Helper: Tambah fitur waktu ---
def add_time_features(df):
    jam = pd.to_datetime(df['Timestamp']).dt.hour
    hari = pd.to_datetime(df['Timestamp']).dt.dayofweek
    bulan = pd.to_datetime(df['Timestamp']).dt.month
    df['sin_jam'] = np.sin(2 * np.pi * jam / 24)
    df['cos_jam'] = np.cos(2 * np.pi * jam / 24)
    df['sin_hari'] = np.sin(2 * np.pi * hari / 7)
    df['cos_hari'] = np.cos(2 * np.pi * hari / 7)
    df['sin_bulan'] = np.sin(2 * np.pi * (bulan-1) / 12)
    df['cos_bulan'] = np.cos(2 * np.pi * (bulan-1) / 12)
    return df

# --- Update rolling window ---
def update_rolling_window(df_new):
    if os.path.exists(ROLLING_CSV_PATH):
        df_hist = pd.read_csv(ROLLING_CSV_PATH)
        if not all(col in df_new.columns for col in df_hist.columns):
            raise ValueError(f"Kolom data baru tidak cocok dengan header rolling window: {df_hist.columns.tolist()}")
        df_new = df_new[df_hist.columns]
        df_hist = pd.concat([df_hist, df_new], ignore_index=True)
        df_hist = df_hist.drop_duplicates(subset=['Timestamp'], keep='last')
    else:
        df_hist = df_new.copy()
    if len(df_hist) > ROLLING_WINDOW_SIZE:
        df_hist = df_hist.tail(ROLLING_WINDOW_SIZE)
    df_hist.to_csv(ROLLING_CSV_PATH, index=False)
    print(f"[INFO] Rolling window diupdate. Jumlah data sekarang: {len(df_hist)}")
    return df_hist

# --- Windowing, scaling, dsb ---
def windowing_and_scaling(rolling_df):
    rolling_df = add_time_features(rolling_df)
    scaler = MinMaxScaler()
    rolling_df[PREDICTION_FEATURES] = scaler.fit_transform(rolling_df[PREDICTION_FEATURES])
    data_values = rolling_df[ALL_FEATURES].values
    target_values = rolling_df[PREDICTION_FEATURES].values
    X, y = [], []
    for i in range(len(data_values) - N_INPUT - N_OUTPUT + 1):
        X.append(data_values[i:i+N_INPUT])
        y.append(target_values[i+N_INPUT:i+N_INPUT+N_OUTPUT])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# --- Model builder ---
def build_lstm_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32, return_sequences=True))
    model.add(TimeDistributed(Dense(output_shape[-1])))
    model.compile(optimizer=Adam(learning_rate=0.002), loss='mse')
    return model

# --- Retrain LSTM ---
def retrain_lstm_model():
    df = pd.read_csv(ROLLING_CSV_PATH)
    if len(df) < MIN_TRAIN:
        print("[LSTM] Data rolling window belum cukup untuk retrain.")
        return None, None
    X, y, scaler = windowing_and_scaling(df)
    if os.path.exists(MODEL_PATH):
        print("[LSTM] Load model lama untuk fine-tuning...")
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        model.fit(X, y, epochs=5, batch_size=8, verbose=1)
    else:
        print("[LSTM] Model belum ada, training dari awal...")
        model = build_lstm_model(X.shape[1:], y.shape[1:])
        model.fit(X, y, epochs=30, batch_size=8, verbose=1)
    model.save(MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print("[LSTM] Retrain model selesai.")
    return model, scaler

# --- Prediksi 24 jam ---
def predict_24h(model, scaler):
    df = pd.read_csv(ROLLING_CSV_PATH)
    if len(df) < N_INPUT:
        print("[LSTM] Data tidak cukup untuk prediksi.")
        return []
    df = add_time_features(df)
    X_input = df[ALL_FEATURES].values[-N_INPUT:]
    X_input[:, :4] = scaler.transform(X_input[:, :4])
    X_input = np.expand_dims(X_input, axis=0)
    y_pred = model.predict(X_input, verbose=0)[0]  # (N_OUTPUT, 4)
    y_pred_inv = scaler.inverse_transform(y_pred)
    last_time = pd.to_datetime(df['Timestamp'].iloc[-1])
    pred_list = []
    for i, row in enumerate(y_pred_inv):
        pred_time = last_time + pd.Timedelta(hours=i+1)
        pred_dict = {
            "Timestamp": str(pred_time),
            "Suhu (C)": float(row[0]),
            "Kelembaban (%)": float(row[1]),
            "Tekanan Udara (hPa)": float(row[2]),
            "Kecepatan Angin (m/s)": float(row[3])
        }
        pred_list.append(pred_dict)
    print("[LSTM] Prediksi 24 jam selesai.")
    return pred_list

# --- Load model klasifikasi sekali di awal ---
with open(MODEL_KLASIFIKASI_PATH, 'rb') as f:
    model_klasifikasi = pickle.load(f)

# --- Endpoint: Prediksi + Klasifikasi 24 jam ---
@app.route('/predik_klasifikasi_1', methods=['POST'])
def predik_klasifikasi():
    try:
        data = request.get_json()
        required_fields = ['Timestamp'] + PREDICTION_FEATURES
        valid, msg = validate_input(data, required_fields)
        if not valid:
            return jsonify({"status": "error", "message": msg}), 400

        # 1. Update rolling window
        df_new = pd.DataFrame([data])
        rolling_df = update_rolling_window(df_new)

        # 2. Cek data cukup untuk retrain
        if len(rolling_df) < MIN_TRAIN:
            return jsonify({"status": "error", "message": f"Rolling window belum cukup data (current: {len(rolling_df)})"}), 400

        # 3. Retrain model
        print("[LSTM] Mulai retrain model...")
        model, scaler = retrain_lstm_model()
        if model is None or scaler is None:
            return jsonify({"status": "error", "message": "Retrain gagal, data kurang atau model error"}), 500

        # 4. Prediksi 24 jam ke depan
        print("[LSTM] Prediksi 24 jam ke depan...")
        preds = predict_24h(model, scaler)
        if not preds:
            return jsonify({"status": "error", "message": "Prediksi gagal (data rolling window kurang)"}), 500

        # 5. Klasifikasikan setiap hasil prediksi 24 jam (batch)
        X_pred_batch = np.array([[p["Suhu (C)"], p["Kelembaban (%)"], p["Tekanan Udara (hPa)"], p["Kecepatan Angin (m/s)"]] for p in preds])
        kategori_batch = model_klasifikasi.predict(X_pred_batch)
        for i, p in enumerate(preds):
            p["kategori_cuaca"] = kategori_batch[i]

        # 6. Siapkan batch untuk Laravel
        payload_batch = []
        for p in preds:
            payload_batch.append({
                "timestamp": p["Timestamp"],
                "suhu": p["Suhu (C)"],
                "kelembaban": p["Kelembaban (%)"],
                "tekananudara": p["Tekanan Udara (hPa)"],
                "kecepatanangin": p["Kecepatan Angin (m/s)"],
                "kategori_cuaca": p["kategori_cuaca"]
            })
        try:
            resp = requests.post(LARAVEL_PREDIKSI_ENDPOINT, json=payload_batch, timeout=10)
            resp.raise_for_status()
            try:
                laravel_response = resp.json()
            except Exception:
                laravel_response = {"raw": resp.text}
        except Exception as e:
            laravel_response = {"error": str(e)}

        return jsonify({
            "status": "success",
            "message": "Retrain dan prediksi 24 jam selesai.",
            "prediksi_24_jam": payload_batch,
            "laravel_response": laravel_response
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Endpoint: Klasifikasi Cuaca Real-Time ---
@app.route('/klasifikasi_udara_1', methods=['POST'])
def klasifikasi_udara_1():
    try:
        data = request.get_json()
        required_fields = KLASIFIKASI_FEATURES
        valid, msg = validate_input(data, required_fields)
        if not valid:
            return jsonify({"status": "error", "message": msg}), 400
        X_input = np.array([[data[f] for f in KLASIFIKASI_FEATURES]])
        kategori_pred = model_klasifikasi.predict(X_input)[0]
        result = {
            "suhu": data["Suhu (C)"],
            "kelembaban": data["Kelembaban (%)"],
            "tekananudara": data["Tekanan Udara (hPa)"],
            "kecepatanangin": data["Kecepatan Angin (m/s)"],
            "kategori_cuaca": kategori_pred
        }

        # Kirim hasil ke Laravel (khusus endpoint klasifikasi udara)
        try:
            resp = requests.post(LARAVEL_KLASIFIKASI_UDARA_ENDPOINT, json=result, timeout=10)
            resp.raise_for_status()
            try:
                laravel_response = resp.json()
            except Exception:
                laravel_response = {"raw": resp.text}
        except Exception as e:
            laravel_response = {"error": str(e)}

        return jsonify({
            "status": "success",
            "result": result,
            "laravel_response": laravel_response
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

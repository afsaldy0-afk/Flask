from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import requests
from tensorflow.keras.models import load_model

app = Flask(__name__)

LARAVEL_KALIBRASI_KUALITAS_UDARA_ENDPOINT = "API-Web"

# === DEKLARASI PATH FILE DAN VARIABEL PENTING ===
MODEL_PATH         = "Lokasi File Model (.keras)"
SCALER_X_PATH      = "Lokasi Scaler (.joblib)"
EMA_CSV            = "Lokasi ema.csv"
BUFFER_SIZE        = 6  # rolling window untuk span=5
TARGET_SCALER_PATH = {
    "PM 2.5_BMKG": "Lokasi Scaler pm (.joblib)",
    "CO_BMKG": "Lokasi Scaler co (.joblib)",
    "CH4_BMKG": "Lokasi Scaler ch4 (.joblib)",
    "OZON_BMKG": "Lokasi Scaler o3 (.joblib)"
}

FEATURE_ORDER = [
    "PM 2.5 (ug/m3)", "CO (ADC)", "CH4 (ADC)", "OZON (ADC)",
    "CO (ADC)_EMA", "CH4 (ADC)_EMA", "OZON (ADC)_EMA", "Suhu (C)", "Kelembaban (%)",
    "Tekanan Udara (hPa)", "Kecepatan Angin (m/s)", "PM 1 (ug/m3)", "PM 10 (ug/m3)",
    "NH3 (ADC)", "NO2 (ADC)", "NH3 (ADC)_EMA", "NO2 (ADC)_EMA"
]
TARGET_COLS = ["PM 2.5_BMKG", "CO_BMKG", "CH4_BMKG", "OZON_BMKG"]
EMA_CONFIG = {
    "CO (ADC)": 5, "CH4 (ADC)": 5, "OZON (ADC)": 5, "NO2 (ADC)": 5, "NH3 (ADC)": 5
}
LOG_TARGET_COLS = []  # Kosong jika tidak ada target yang di-log waktu training

RAW_FIELDS = [
    "PM 2.5 (ug/m3)", "CO (ADC)", "CH4 (ADC)", "OZON (ADC)",
    "Suhu (C)", "Kelembaban (%)", "Tekanan Udara (hPa)", "Kecepatan Angin (m/s)",
    "PM 1 (ug/m3)", "PM 10 (ug/m3)", "NH3 (ADC)", "NO2 (ADC)"
]

# === LOAD MODEL DAN SCALER SEKALI DI AWAL ===
model = load_model(MODEL_PATH)
input_scaler = joblib.load(SCALER_X_PATH)
target_scalers = {col: joblib.load(TARGET_SCALER_PATH[col]) for col in TARGET_COLS}

# === FUNGSI BUFFER DAN EMA ===
def update_ema_csv(data):
    df_new = pd.DataFrame([data])
    # Cek apakah file tidak ada atau kosong
    if not os.path.exists(EMA_CSV) or os.path.getsize(EMA_CSV) == 0:
        # Buat file baru dengan header
        df = df_new
    else:
        try:
            df = pd.read_csv(EMA_CSV)
            df = pd.concat([df, df_new], ignore_index=True)
            if len(df) > BUFFER_SIZE:
                df = df.tail(BUFFER_SIZE)
        except Exception:
            # Jika file corrupt/kosong, mulai file baru
            df = df_new
    df.to_csv(EMA_CSV, index=False)
    return df

def hitung_fitur_ema(df, ema_config):
    """Ambil baris terakhir, tambahkan fitur EMA"""
    ema_features = {}
    for col, span in ema_config.items():
        ema_col = f"{col}_EMA"
        if col in df.columns and len(df) >= 1:
            ema_val = df[col].ewm(span=span, adjust=False).mean().iloc[-1]
            ema_features[ema_col] = float(ema_val)
        else:
            ema_features[ema_col] = 0.0
    return ema_features

# === ENDPOINT FLASK ===
@app.route('/kalibrasi_kualitas_udara_1', methods=['POST'])
def kalibrasi_kualitas_udara():
    try:
        data = request.get_json()
        # --- Validasi input mentah ---
        missing = [f for f in RAW_FIELDS if f not in data]
        if missing:
            return jsonify({"status": "error", "message": f"Missing fields: {', '.join(missing)}"}), 400

        # --- Update ema.csv rolling window ---
        df_hist = update_ema_csv({f: data[f] for f in RAW_FIELDS})

        # --- Hitung semua fitur EMA dari isi ema.csv ---
        ema_features = hitung_fitur_ema(df_hist, EMA_CONFIG)

        # --- Gabungkan fitur input sesuai FEATURE_ORDER ---
        row = dict(data)
        row.update(ema_features)
        input_vector = [float(row.get(f, 0.0)) for f in FEATURE_ORDER]

        # --- Scaling input ---
        X_scaled = input_scaler.transform([input_vector])

        # --- Prediksi ANN ---
        y_pred_scaled = model.predict(X_scaled)
        y_pred = []
        for i, col in enumerate(TARGET_COLS):
            val = target_scalers[col].inverse_transform(y_pred_scaled[:, i].reshape(-1, 1))[0][0]
            if col in LOG_TARGET_COLS:
                val = np.expm1(val)
            y_pred.append(val)
        result = {col: float(y_pred[i]) for i, col in enumerate(TARGET_COLS)}

        # --- Ambil suhu dan tekanan dari input ---
        temp_C = float(data["Suhu (C)"])
        pressure_hPa = float(data["Tekanan Udara (hPa)"])

        # --- Konversi suhu dan tekanan ke satuan SI ---
        T = temp_C + 273.15  # K
        P = pressure_hPa * 100  
        R = 8.314  

        # --- Berat molekul gas (g/mol) ---
        MW = {
            "CO": 28.01,
            "CH4": 16.04,
            "O3": 48.00
        }

        # --- Fungsi konversi ---
        def ppm_to_ugm3(ppm, mw):  # untuk CO, CH4
            return (ppm * P * mw) / (R * T)

        def ppb_to_ugm3(ppb, mw):  # untuk O3
            return (ppb * P * mw) / (R * T * 1000)


        # --- Konversi hasil ANN ke ug/m3 sesuai jurnal ---
        pm25_bmkg_ugm3 = result.get("PM 2.5_BMKG", 0.0)  
        co_bmkg_ugm3   = ppm_to_ugm3(result.get("CO_BMKG", 0.0), MW["CO"])
        ch4_bmkg_ugm3  = ppm_to_ugm3(result.get("CH4_BMKG", 0.0), MW["CH4"])
        ozon_bmkg_ugm3 = ppb_to_ugm3(result.get("OZON_BMKG", 0.0), MW["O3"])

        laravel_payload = {
            'pm25_bmkg': round(pm25_bmkg_ugm3, 2),
            'co_bmkg': round(co_bmkg_ugm3, 2),
            'ch4_bmkg': round(ch4_bmkg_ugm3, 2),
            'ozon_bmkg': round(ozon_bmkg_ugm3, 2),
        }

        # --- Kirim ke endpoint Laravel ---
        try:
            resp = requests.post(LARAVEL_KALIBRASI_KUALITAS_UDARA_ENDPOINT, json=laravel_payload, timeout=10)
            resp.raise_for_status()
            try:
                laravel_response = resp.json()
            except Exception:
                laravel_response = {"raw": resp.text}
        except Exception as e:
            laravel_response = {"error": str(e)}

        return jsonify({
            "status": "success",
            "prediction": result,
            "prediction_ugm3": laravel_payload,
            "laravel_response": laravel_response
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')

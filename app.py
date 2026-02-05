
import os
import io
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Risiko Stroke", page_icon="🧠", layout="centered")

# =========================
# Config
# =========================
FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

CANDIDATE_MODEL_PATHS = [
    "stroke_risk_model.joblib",
    "models/stroke_risk_model.joblib",
    "/mnt/data/stroke_risk_model.joblib",
    "/mnt/data/models/stroke_risk_model.joblib",
]

# =========================
# Helpers
# =========================
@st.cache_resource
def load_model():
    """
    Load model pipeline (preprocess + model) saved with joblib.
    Will try multiple default paths; if not found, return None.
    """
    for p in CANDIDATE_MODEL_PATHS:
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None

def predict_df(model, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Return dataframe with probability and predicted class.
    Assumes model is a sklearn pipeline with predict_proba.
    """
    X = X.copy()

    # Ensure all required columns exist
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ada di input: {missing}")

    # Keep only needed columns & correct order
    X = X[FEATURES]

    # Predict
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = X.copy()
    out["prob_stroke"] = proba
    out["pred_class"] = pred
    return out

# =========================
# UI
# =========================
st.title("🧠 Prediksi Risiko Stroke")
st.caption(
    "Aplikasi ini menampilkan prediksi risiko stroke (0 = tidak berisiko, 1 = berisiko) "
    "beserta probabilitasnya, menggunakan model Machine Learning yang sudah dilatih & disimpan (joblib)."
)

with st.expander("ℹ️ Petunjuk singkat"):
    st.markdown(
        """
        **Sesuai ketentuan deployment:**  
        - Input data bisa melalui **form** (slider/text input) atau **upload CSV**.  
        - Klik tombol **Prediksi**.  
        - Muncul hasil **kelas/nilai** dan **probabilitas**.  

        **Model file:** tempatkan file model `stroke_risk_model.joblib` di folder yang sama dengan `app.py`  
        (atau di folder `models/`), lalu jalankan:
        ```bash
        streamlit run app.py
        ```

        > Catatan: Prediksi ini hanya alat bantu deteksi dini, **bukan diagnosis medis**.
        """
    )

model, model_path = load_model()

# Optional: allow upload model if not found
if model is None:
    st.warning(
        "Model belum ditemukan. Letakkan `stroke_risk_model.joblib` di folder yang sama dengan `app.py` "
        "atau folder `models/`, atau upload file model di bawah ini."
    )
    uploaded_model = st.file_uploader("Upload model (.joblib)", type=["joblib"])
    if uploaded_model is not None:
        # Save to temp path and load
        tmp_path = "stroke_risk_model.joblib"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        model = joblib.load(tmp_path)
        model_path = os.path.abspath(tmp_path)

if model is not None:
    st.success(f"✅ Model loaded: `{model_path}`")

# Threshold control (optional but useful)
threshold = st.slider("Threshold prediksi (probabilitas ≥ threshold → kelas 1)", 0.05, 0.95, 0.50, 0.01)

tabs = st.tabs(["🧍 Prediksi 1 Pasien (Form)", "📄 Prediksi Banyak Pasien (Upload CSV)"])

# -------------------------
# Tab 1: Single input
# -------------------------
with tabs[0]:
    st.subheader("Input Data Pasien")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
        age = st.slider("Age", 0, 100, 45)
        ever_married = st.selectbox("Ever married", ["Yes", "No"], index=0)
        work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], index=0)
        residence = st.selectbox("Residence type", ["Urban", "Rural"], index=0)

    with col2:
        hypertension = st.selectbox("Hypertension (0/1)", [0, 1], index=0)
        heart_disease = st.selectbox("Heart disease (0/1)", [0, 1], index=0)
        avg_glucose_level = st.number_input("Average glucose level", min_value=0.0, max_value=400.0, value=100.0, step=0.1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        smoking_status = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"], index=0)

    if st.button("Prediksi (1 Pasien)", type="primary", disabled=(model is None)):
        X_one = pd.DataFrame([{
            "gender": gender,
            "age": float(age),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence,
            "avg_glucose_level": float(avg_glucose_level),
            "bmi": float(bmi),
            "smoking_status": smoking_status,
        }], columns=FEATURES)

        try:
            out = predict_df(model, X_one, threshold=threshold).iloc[0]
            prob = float(out["prob_stroke"])
            pred = int(out["pred_class"])

            st.markdown("### Hasil")
            st.metric("Prediksi kelas", pred, help="0 = tidak berisiko, 1 = berisiko")
            st.metric("Probabilitas stroke", f"{prob:.4f}")
            st.progress(min(max(prob, 0.0), 1.0))

        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

# -------------------------
# Tab 2: Batch input (CSV)
# -------------------------
with tabs[1]:
    st.subheader("Upload CSV untuk Prediksi Banyak Pasien")
    st.markdown(
        """
        **Format CSV wajib** memiliki kolom berikut (case-sensitive):  
        `gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status`
        """
    )

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.write("Preview data:", df_in.head())

            if st.button("Prediksi (CSV)", type="primary", disabled=(model is None)):
                out = predict_df(model, df_in, threshold=threshold)
                st.success("Prediksi selesai ✅")
                st.write("Hasil (dengan kolom `prob_stroke` dan `pred_class`):")
                st.dataframe(out)

                # Download result
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download hasil prediksi (CSV)",
                    data=csv_bytes,
                    file_name="hasil_prediksi_stroke.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Gagal membaca/prediksi CSV: {e}")

st.divider()
st.caption("© Prediksi Risiko Stroke - Deployment Streamlit (Capstone Project)")

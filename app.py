import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="Asteroid Class Predictor", page_icon="☄️", layout="wide")

# ---------------------------------------------------------------- Load artifacts
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("asteroid_model.keras")
    scaler = joblib.load("scaler.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    with open("meta.json") as f:
        meta = json.load(f)
    return model, scaler, label_encoder, meta


try:
    model, scaler, label_encoder, meta = load_artifacts()
except Exception as e:
    st.error(
        "Could not load model files. Make sure `asteroid_model.keras`, `scaler.joblib`, "
        "`label_encoder.joblib` and `meta.json` are in the same folder as app.py "
        "(run `save_artifacts.py` code at the end of your notebook to create them).\n\n"
        f"Details: {e}"
    )
    st.stop()

FEATURES = meta["features"]                 # exact training column order (33)
DEFAULTS = meta["defaults"]                 # median raw values
SCALED_COLS = list(scaler.feature_names_in_)  # numeric columns the scaler was fit on
CLASSES = label_encoder.classes_

CLASS_INFO = {
    "MBA": "Main-belt Asteroid",
    "OMB": "Outer Main-belt Asteroid",
    "IMB": "Inner Main-belt Asteroid",
    "MCA": "Mars-crossing Asteroid",
    "APO": "Apollo (Earth-crossing, a > 1 AU)",
    "AMO": "Amor (Earth-approaching)",
    "ATE": "Aten (Earth-crossing, a < 1 AU)",
    "TJN": "Jupiter Trojan",
    "CEN": "Centaur",
    "TNO": "Trans-Neptunian Object",
    "AST": "Asteroid (other)",
    "IEO": "Interior Earth Object (Atira)",
    "HYA": "Hyperbolic Asteroid",
    "PAA": "Parabolic Asteroid",
}

# Fields shown up-front; everything else goes in the advanced expander
PRIMARY = ["H", "e", "a", "q", "i", "om", "w", "ma", "ad", "n", "per_y", "moid", "rms"]
LABELS = {
    "H": "Absolute magnitude (H)",
    "e": "Eccentricity (e)",
    "a": "Semi-major axis (a, AU)",
    "q": "Perihelion distance (q, AU)",
    "i": "Inclination (i, deg)",
    "om": "Longitude of ascending node (om, deg)",
    "w": "Argument of perihelion (w, deg)",
    "ma": "Mean anomaly (ma, deg)",
    "ad": "Aphelion distance (ad, AU)",
    "n": "Mean motion (n, deg/day)",
    "per_y": "Orbital period (years)",
    "moid": "Earth MOID (AU)",
    "rms": "Orbit-fit RMS",
}


# ---------------------------------------------------------------- Helpers
def build_model_input(df_raw: pd.DataFrame) -> np.ndarray:
    """Raw feature dataframe -> scaled array in the training column order."""
    df = df_raw.copy()

    # Convert neo / pha (Y/N) -> neo_Y / pha_Y if the raw columns are present
    for col in ("neo", "pha"):
        if col in df.columns:
            df[f"{col}_Y"] = (df[col].astype(str).str.upper() == "Y").astype(float)
    for col in ("neo_Y", "pha_Y"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].astype(float)

    # Fill any missing feature with its training median
    for col in FEATURES:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, 0.0)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULTS.get(col, 0.0))

    df[SCALED_COLS] = scaler.transform(df[SCALED_COLS])
    return df[FEATURES].values.astype("float32")


def predict(df_raw: pd.DataFrame) -> np.ndarray:
    return model.predict(build_model_input(df_raw), verbose=0)


# ---------------------------------------------------------------- UI
st.title("☄️ Asteroid Orbit Class Predictor")
st.caption("Deep neural network trained on the JPL small-body dataset (~958k asteroids).")

with st.sidebar:
    st.header("About")
    st.write(
        "Enter orbital parameters for an asteroid and the model predicts its "
        f"orbit class out of **{len(CLASSES)}** classes."
    )
    st.write("**Classes:** " + ", ".join(CLASSES))
    st.divider()
    st.write("**Model:** Dense(192) → Dropout → Dense(64) → Dropout → Softmax")
    st.write("Tuned with Keras Tuner (RandomSearch).")

tab_single, tab_batch = st.tabs(["🔭 Single prediction", "📄 Batch (CSV)"])

# ------------------------------------------------------- Single prediction
with tab_single:
    st.subheader("Orbital parameters")
    values = {}

    cols = st.columns(3)
    for idx, feat in enumerate(PRIMARY):
        if feat in FEATURES:
            with cols[idx % 3]:
                values[feat] = st.number_input(
                    LABELS.get(feat, feat),
                    value=float(DEFAULTS.get(feat, 0.0)),
                    format="%.6f",
                    key=f"in_{feat}",
                )

    c1, c2 = st.columns(2)
    with c1:
        values["neo_Y"] = 1.0 if st.selectbox("Near-Earth Object (neo)?", ["N", "Y"]) == "Y" else 0.0
    with c2:
        values["pha_Y"] = 1.0 if st.selectbox("Potentially Hazardous (pha)?", ["N", "Y"]) == "Y" else 0.0

    other = [f for f in FEATURES if f not in PRIMARY and f not in ("neo_Y", "pha_Y")]
    with st.expander("Advanced parameters (epoch, uncertainties, etc.) – defaults are dataset medians"):
        acols = st.columns(3)
        for idx, feat in enumerate(other):
            with acols[idx % 3]:
                values[feat] = st.number_input(
                    feat,
                    value=float(DEFAULTS.get(feat, 0.0)),
                    format="%.8g",
                    key=f"adv_{feat}",
                )

    if st.button("Predict class", type="primary"):
        probs = predict(pd.DataFrame([values]))[0]
        top = int(np.argmax(probs))
        code = CLASSES[top]

        st.success(f"Predicted class: **{code}** – {CLASS_INFO.get(code, '')}")
        st.metric("Confidence", f"{probs[top] * 100:.2f}%")

        prob_df = (
            pd.DataFrame({"Class": CLASSES, "Probability": probs})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        st.bar_chart(prob_df.set_index("Class"))
        st.dataframe(prob_df.style.format({"Probability": "{:.4%}"}), use_container_width=True)

# ------------------------------------------------------- Batch prediction
with tab_batch:
    st.subheader("Upload a CSV")
    st.write(
        "The CSV can include any of the model's features (e.g. `H, e, a, q, i, om, w, ma, ad, n, "
        "per_y, moid, rms, neo, pha, ...`). Missing columns are filled with training medians."
    )
    file = st.file_uploader("CSV file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file, low_memory=False)
        st.write("Preview:", data.head())

        if st.button("Run batch prediction"):
            probs = predict(data)
            out = data.copy()
            out["predicted_class"] = CLASSES[probs.argmax(axis=1)]
            out["confidence"] = probs.max(axis=1)

            st.success(f"Predicted {len(out):,} rows.")
            st.dataframe(out.head(100), use_container_width=True)
            st.bar_chart(out["predicted_class"].value_counts())
            st.download_button(
                "Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                file_name="asteroid_predictions.csv",
                mime="text/csv",
            )

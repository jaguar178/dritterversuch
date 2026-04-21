import streamlit as st
import json
from PIL import Image
from utils.predict import predict_image

# Datenbank laden
with open("database/motorcycles.json") as f:
    db = json.load(f)

st.title("Motorrad Erkennung App")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    label, confidence = predict_image(image)

    st.subheader("Erkanntes Modell:")
    st.write(f"{label} ({confidence:.2f})")

    # Daten anzeigen
    if label in db:
        st.subheader("Technische Daten:")
        for key, value in db[label].items():
            st.write(f"**{key}:** {value}")
    else:
        st.warning("Keine Daten gefunden.")

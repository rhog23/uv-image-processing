import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

st.set_page_config(page_title="Penguin Predictor", page_icon=":penguin:")

st.title("🐧 Penguin Classifier")
st.markdown(
    """
Curious which penguin you're looking at?  
Enter the penguin details below — this app uses a trained model from the Palmer Penguins dataset to predict its species! We use ANN 🧠 model for prediction.
"""
)

species = ["Adelie", "Chinstrap", "Gentoo"]

# load saved scaler
with open("scaler_penguin.pkl", "rb") as f:
    scaler = pickle.load(f)

# me-load model ann
ann_model = tf.keras.models.load_model("ann_penguin.keras")

# input user (dipindahkan ke sidebar)
st.sidebar.header("Input Features")

island = st.sidebar.selectbox(
    "Penguin Island", options=["Biscoe", "Dream", "Torgersen"]
)
sex = st.sidebar.selectbox("Sex", options=["Female", "Male"])
bill_length = st.sidebar.number_input("Bill Length (mm)", min_value=0.0)
bill_depth = st.sidebar.number_input("Bill Depth (mm)", min_value=0.0)
flipper_length = st.sidebar.number_input("Flipper Length (mm)", min_value=0.0)
body_mass = st.sidebar.number_input("Body Mass (g)", min_value=0.0)

island_Biscoe, island_Dream, island_Torgersen = 0, 0, 0
if island == "Biscoe":
    island_Biscoe = 1
elif island == "Dream":
    island_Dream = 1
elif island == "Torgersen":
    island_Torgersen = 1

sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

# ubah input jadi dataframe (karena nantinya mau di transform)
input_data = pd.DataFrame(
    [
        {
            "bill_length_mm": bill_length,
            "bill_depth_mm": bill_depth,
            "flipper_length_mm": flipper_length,
            "body_mass_g": body_mass,
            "island_Biscoe": island_Biscoe,
            "island_Dream": island_Dream,
            "island_Torgersen": island_Torgersen,
            "sex_female": sex_female,
            "sex_male": sex_male,
        }
    ]
)

# transformasi inputan user menggunakan scaler yang sudah di load
scaled_columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
input_data[scaled_columns] = scaler.transform(input_data[scaled_columns])

st.markdown("### Input Preview")
st.dataframe(input_data)

# melakukan prediksi
if st.button("Predict Penguin Species"):
    prediction_probs = ann_model.predict(input_data)[0]

    predicted_index = np.argmax(prediction_probs)
    confidence = prediction_probs[predicted_index]

    st.success(
        f"🎉 Predicted species: **{species[predicted_index]}** "
        f"with confidence **{confidence:.2%}**"
    )

    st.progress(float(confidence))

    st.markdown("### Probability detail")
    prob_df = pd.DataFrame(prediction_probs.reshape(1, -1), columns=species)
    st.dataframe(prob_df.style.highlight_max(axis=1, color="green"))

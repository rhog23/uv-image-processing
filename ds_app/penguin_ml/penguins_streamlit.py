import streamlit as st
import pickle
import numpy as np

st.title("Penguin Classifier :penguin::snowman:")
st.markdown(
    "Curious which penguin you're looking at? Just enter six details below—this app will use a trained model from the Palmer Penguins dataset to make a prediction! We use the RandomForest model for prediction :deciduous_tree:"
)

species = ["Adelie", "Chinstrap", "Gentoo"]

# me-load file pickle model
rf_pickle = open("random_forest_penguin.pickle", "rb")
rfc = pickle.load(rf_pickle)
rf_pickle.close()


island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
sex = st.selectbox("Sex", options=["Female", "Male"])
bill_length = st.number_input("Bill Length (mm)", min_value=0)
bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
body_mass = st.number_input("Body Mass (g)", min_value=0)
user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]
st.markdown(f"Inputs: {user_inputs}")

# konversi value untuk island dan gender
island_value = None
if island == "Biscoe":
    island_value = 0
elif island == "Dream":
    island_value = 1
elif island == "Torgerson":
    island_value = 2
gender = None
if sex == "Female":
    gender = 0
elif sex == "Male":
    gender = 1

new_prediction = rfc.predict_proba(
    [
        [
            island_value,
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            gender,
        ]
    ]
)
predicted_class_index = np.argmax(new_prediction)
confidence = new_prediction[0][predicted_class_index]

st.markdown(
    f"We predict your penguin is of the {species[predicted_class_index]} ({confidence:.2%}) species :penguin:"
)

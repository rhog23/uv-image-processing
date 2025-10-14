import streamlit as st
import pandas as pd

st.title("SF Trees Map 🗺️🌳")
st.write(
    "Explore San Francisco’s urban forest with this app, powered by tree data from SF DPW."
)

trees_df = pd.read_csv("trees.csv")
clean_df = trees_df.dropna(subset=["longitude", "latitude"])
st.map(data=clean_df.sample(1000), size=20, color="#e9edc9")

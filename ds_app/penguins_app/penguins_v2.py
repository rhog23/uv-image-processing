import streamlit as st
import altair as alt
import pandas as pd

st.title("Palmer's Penguins 🐧 - version 2")

penguins_df = pd.read_csv("penguins.csv")

st.markdown("Let's make Streamlit app to show scatterplot about penguins! 🐧")

selected_x_var = st.selectbox(
    "Pick a variable to explore on the x-axis:",
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
)
selected_y_var = st.selectbox(
    "Now, pick a variable for the y-axis:",
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
)

alt_chart = (
    alt.Chart(penguins_df, title=f"Scatterplot of Palmer's Penguins🐧")
    .mark_circle()
    .encode(
        x=selected_x_var,
        y=selected_y_var,
        color="species",  # untuk membedakan warna antar spesies penguin
    )
    .interactive()
)
st.altair_chart(alt_chart)

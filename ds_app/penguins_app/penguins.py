import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns

st.title("Palmer's Penguins")

# import the data
penguins_df = pd.read_csv("penguins.csv")
# st.write(penguins_df.head())

st.markdown("Let's make Streamlit app to show scatterplot about penguins!🐧")
selected_species = st.selectbox(
    "Pick a penguin species to explore!", ["Adelie", "Gentoo", "Chinstrap"]
)
selected_x_var = st.selectbox(
    "Pick a variable to explore on the x-axis:",
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
)
selected_y_var = st.selectbox(
    "Now, pick a variable for the y-axis:",
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
)

penguins_df = penguins_df[penguins_df["species"] == selected_species]
alt_chart = (
    alt.Chart(penguins_df, title=f"Scatterplot of {selected_species} Penguins🐧")
    .mark_circle()
    .encode(
        x=selected_x_var,
        y=selected_y_var,
    )
    .interactive() # memungkinkan kita untuk zoom-in / out
)
st.altair_chart(alt_chart)

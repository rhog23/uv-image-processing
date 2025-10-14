import streamlit as st
import pandas as pd

st.title("SF Trees")
st.write(
    """Explore San Francisco’s urban forest with this app 🌳🌲, powered by tree data from SF DPW"""
)
st.markdown(
    "[Click here to visit the full dataset](https://data.sfgov.org/City-Infrastructure/Street-Tree-List/tkzw-k3nq)"
)

trees_df = pd.read_csv("trees.csv")

df_dbh_grouped = pd.DataFrame(trees_df.groupby(["dbh"]).count()["tree_id"])
# df_dbh_grouped.rename("tree_count", inplace=True) # kalau series (tidak pakai DataFrame)
df_dbh_grouped.columns = ["tree_count"]
st.write(df_dbh_grouped)

st.line_chart(df_dbh_grouped)
st.bar_chart(df_dbh_grouped)
st.area_chart(df_dbh_grouped)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.write("Helloo world!!")
x = np.linspace(0, 20, 100)  # membuat data dummy 0 sampai 20 dengan 100 titik data

fig, ax = plt.subplots()
ax.plot(x)
st.pyplot(fig)

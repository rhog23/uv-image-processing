import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.write("Website dengan Dua Grafik Matplotlib")

fig1, ax1 = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 50)
ax1.plot(np.sin(x), label="sin(x)")
ax1.plot(np.cos(x), label="cos(x)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)


y = np.array([30, 70, 25, 15])
# labels
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

fig2, ax2 = plt.subplots()
ax2.pie(y, labels=mylabels, autopct="%.1f%%")
ax2.set_title("Proporsi Buah")
ax2.legend(title="Jenis Buah", bbox_to_anchor=(1, 1))
st.pyplot(fig2)

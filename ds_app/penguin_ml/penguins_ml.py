import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import streamlit as st
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Penguin Model Trainer", page_icon=":penguin:")

st.title("Penguins ML Model Trainer :penguin::gear:")

penguin_df = pd.read_csv("penguins.csv")

# bersihkan dari null values
clean_df = penguin_df.dropna()
st.markdown("Dataset yang telah dibersihkan :broom::sparkles:")
st.write(
    clean_df.isna()
    .sum()
    .reset_index()
    .rename(columns={"index": "features", 0: "count"})
)
st.write(clean_df["species"].value_counts())

le = LabelEncoder()
clean_df["island"] = le.fit_transform(clean_df["island"])
clean_df["sex"] = le.fit_transform(clean_df["sex"])
st.write(clean_df)

# memisahkan antara fitur (X) dan label (y)
X = clean_df.drop(["species", "year"], axis=1)
y = clean_df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23
)

# Model random forest
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

# untuk menampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(cm, display_labels=penguin_df["species"].unique())
disp.plot(ax=ax, colorbar=False)
st.write(fig)

# performa dari random forest
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

st.subheader("Model Evaluation Metrics :bar_chart:")
st.write(f"Accuracy: {accuracy:.2%}")
st.write(f"Precision (macro): {precision:.2%}")
st.write(f"Recall (macro): {recall:.2%}")
st.write(f"F1-score (macro): {f1:.2%}")

report_dict = classification_report(y_test, y_pred, digits=5, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

st.subheader("Laporan Klasifikasi :memo:")
st.dataframe(report_df)


# Menyimpan model
rf_pickle = open("random_forest_penguin.pickle", "wb")
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

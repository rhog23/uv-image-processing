import pandas as pd
import numpy as np
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import plot_model

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

encoded_penguins = pd.get_dummies(clean_df, columns=["island", "sex"])

st.write(clean_df)

# ubah skala dari fitur-fitur penguins
scaled_columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
scaler = StandardScaler()
encoded_penguins[scaled_columns] = scaler.fit_transform(
    encoded_penguins[scaled_columns]
)
st.write(encoded_penguins.head())

# memisahkan antara fitur (X) dan label (y)
X = encoded_penguins.drop(["species", "year"], axis=1)
y = clean_df["species"]
le_y = LabelEncoder()
y = le_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0
)

# Model ANN
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(np.unique(y)), activation="softmax"),
    ],
    name="ann_penguins_model",
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

st.markdown("Training Model")

with st.spinner("Training model...:running_woman::running_man:"):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        verbose=1,
    )
st.success("Training selesai :checkered_flag:")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(history.history["accuracy"], label="Train Accuracy", color="#003049")
ax[0].plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    color="#fca311",
    linestyle="dashed",
)
ax[0].set_title("Model Accuracy")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy")
ax[0].legend()

ax[1].plot(history.history["loss"], label="Train Loss", color="#003049")
ax[1].plot(
    history.history["val_loss"],
    label="Validation Loss",
    color="#fca311",
    linestyle="dashed",
)
ax[1].set_title("Model Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].legend()

st.pyplot(fig)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
# untuk menampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(cm, display_labels=penguin_df["species"].unique())
disp.plot(ax=ax, cmap="Blues", colorbar=False)
st.write(fig)

# performa dari random forest
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average="macro")
recall = recall_score(y_test, y_pred_classes, average="macro")
f1 = f1_score(y_test, y_pred_classes, average="macro")

st.subheader("Model Evaluation Metrics :bar_chart:")
st.write(f"Accuracy: {accuracy:.2%}")
st.write(f"Precision (macro): {precision:.2%}")
st.write(f"Recall (macro): {recall:.2%}")
st.write(f"F1-score (macro): {f1:.2%}")

report_dict = classification_report(y_test, y_pred_classes, digits=5, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

st.subheader("Laporan Klasifikasi :memo:")
st.dataframe(report_df)


# Menyimpan model
model.save("ann_penguin.keras")

scaler_pickle = open("scaler_penguin.pkl", "wb")
pickle.dump(scaler, scaler_pickle)
scaler_pickle.close()

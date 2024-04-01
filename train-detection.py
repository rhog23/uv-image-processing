"""
Mengimport library yang diperlukan
1. Kita membutuhkan library ultralytics untuk memanfaatkan  model YOLOv8s dari Ultralytics.
2. Kita membutuhkan library opencv untuk dapat memanfaatkan kamera, serta beberapa  fungsi-fungsi lainnya yang berkaitan dengan pengolahan citra
seperti mengubah ukuran citra dan segmentasi
"""

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt", task="detect")  #   Load the pre-trained YOLOv8n model
clahe_model = cv2.createCLAHE(
    clipLimit=2.0, tileGridSize=(8, 8)
)  #  membuat objek CLAHE (salah satu metode histogram equalization)

# cv2.VideoCapture(0) membuat objek VideoCapture dan menghubungkan ke kamera. Argumen `0` menandakan kamera yang akan digunakan (di dalam kasus ini adalah kamera pertama yang terhubung ke laptop / raspberry pi)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    frame_b = clahe_model.apply(
        frame[:, :, 0]
    )  # mengaplikasikan clahe terhadap blue channel
    frame_g = clahe_model.apply(
        frame[:, :, 1]
    )  # mengaplikasikan clahe terhadap green channel
    frame_r = clahe_model.apply(
        frame[:, :, 2]
    )  # mengaplikasikan clahe terhadap red channel

    frame_clahe = np.stack((frame_b, frame_g, frame_r), axis=2)

    results = model(
        frame_clahe, imgsz=128
    )  #   Deteksi objek pada citra (dalam kasus ini adalah setiap frame dari webcam)

    for result in results:
        box = result.boxes
        coords = box.xyxy
        if len(coords) > 0:
            x = int(coords[0][0])
            y = int(coords[0][1])
            w = int(coords[0][2])
            h = int(coords[0][3])

            cv2.rectangle(
                frame_clahe, (x, y), (w, h), (0, 255, 255), 2
            )  #  menggambar kotak objek yang telah dideteksi

    cv2.imshow("train detection demo", frame_clahe)
    if cv2.waitKey(1) == ord("q"):
        """
        Untuk memberhentikan program dan menutup window aplikasi jika tombol 'q' di tekan
        """
        break

cap.release()
cv2.destroyAllWindows()

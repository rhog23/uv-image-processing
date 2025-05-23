import cv2
import numpy as np
import time
from collections import Counter
import socket
from scipy import stats as st

# Konfigurasi IP ESP32
esp_ip = "192.168.30.33"  # Ganti dengan IP ESP32 yang muncul di Serial Monitor
port = 80

# Inisialisasi socket TCP
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((esp_ip, port))
    print("[INFO] Terhubung ke ESP32")
except Exception as e:
    print("[INFO] Gagal terhubung ke ESP32:", e)
    exit()
    # pass

detected_colors = []
start_time = time.time()

# Pakai kamera USB index 0
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[INFO] Gagal membuka kamera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Gagal membaca frame dari kamera")
        break

    # Crop tengah 200x200
    height, width, _ = frame.shape
    center_y, center_x = height // 2, width // 2
    half_size = 100
    cropped_frame = frame[
        center_y - half_size : center_y + half_size,
        center_x - half_size : center_x + half_size,
    ]

    # Grayscale dan threshold
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    _, threshold = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # HSV dan ambil hue yang di area threshold
    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
    hue_channel = hsv[:, :, 0]
    v_channel = hsv[:, :, 2]
    target_color = hue_channel[threshold == 255]
    # target_color = st.mode(st.mode(hue_channel).mode).mode
    masked_frame = cv2.subtract(
        cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR), cropped_frame
    )
    masked_frame = cv2.subtract(
        cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR), masked_frame
    )
    dominance_v_channel = st.mode(st.mode(v_channel).mode).mode

    # Check LAB
    # lab = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2LAB)
    # lab_channel = lab[:, :, 1]
    # target_color = lab_channel[threshold == 255]

    warna = ""
    print(f"[INFO] Mode Value: {dominance_v_channel}")
    if dominance_v_channel > 10:
        if target_color.size > 0:
            # median_hue = target_color
            # print(f"{target_color}")
            median_hue = int(np.median(target_color))

            print("[INFO] Median Hue:", median_hue)

            if 0 <= median_hue <= 5 or 160 <= median_hue <= 180:
                warna = "me"
                print("🔴 Warna: Merah")
            elif 11 <= median_hue <= 20:
                warna = "ji"
                print("🟠 Warna: Oranye")
            elif 21 <= median_hue <= 35:
                warna = "ku"
                print("🟡 Warna: Kuning")
            elif 36 <= median_hue <= 85:
                warna = "hi"
                print("🟢 Warna: Hijau")
            elif 100 <= median_hue <= 120:
                warna = "bi"
                print("🔵 Warna: Biru")
            elif 125 <= median_hue <= 170:
                warna = "un"
                print("🟣 Warna: Ungu")
            else:
                warna = "0"
                print("⚠️ Warna di luar klasifikasi:", median_hue)

            detected_colors.append(str.lower(warna))
    else:
        print("⚠️ Tidak ada warna terdeteksi")

    # Tampilkan jendela
    cv2.imshow("Asli", cropped_frame)
    cv2.imshow("Hue", hue_channel)
    cv2.imshow("Value", v_channel)
    # cv2.imshow("Hue", lab_channel)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Threshold", masked_frame)
    # cv2.imshow("Blurred Threshold", thres5)

    # Kirim hasil dominan setiap 3 detik
    if time.time() - start_time >= 3:
        if detected_colors:
            hasil = Counter(detected_colors).most_common(1)[0][0]
            print(f"\n✅ Kesimpulan Warna Dominan 3 Detik: {hasil}\n")
            try:
                client_socket.sendall((hasil + "\n").encode())
                print("📤 Dikirim ke ESP32:", hasil)
            except Exception as e:
                print("❌ Gagal kirim:", e)
        else:
            print("\n⚠️ Tidak ada warna dominan.\n")

        detected_colors.clear()
        start_time = time.time()

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
client_socket.close()
cv2.destroyAllWindows()

from pymata4 import pymata4
import cv2
import numpy as np
import time
import threading

# ===================== KONFIGURASI =====================
# Kamera
CAMERA_FOV_H_DEG = 60  # Derajat field of view horizontal
FRAME_SIZE = (640, 480)
DEAD_ZONE_PERCENT = 0.07  # 7% toleransi dari lebar frame

# Servo Horizontal
SERVO_PAN_PRIMARY_PIN = 9  # Servo utama 270°
SERVO_PAN_SECONDARY_PIN = 3  # Servo sekunder 180°
PAN_NEUTRAL = 1500  # Posisi netral mikro detik
PAN_PRIMARY_RANGE = (500, 2500)
PAN_SECONDARY_RANGE = (1000, 2000)

# Kontrol Pergerakan
KP = 0.4  # Gain proporsional
MAX_STEP = 3  # Maksimum step per update

# ===================== INISIALISASI =====================
# Hitung dead zone dalam pixel
DEAD_ZONE_PX = int(FRAME_SIZE[0] * DEAD_ZONE_PERCENT / 2)

# Setup papan Arduino
board = pymata4.Pymata4()

# Inisialisasi servo
servo_config = {
    SERVO_PAN_PRIMARY_PIN: PAN_PRIMARY_RANGE,
    SERVO_PAN_SECONDARY_PIN: PAN_SECONDARY_RANGE,
}

for pin, (min_, max_) in servo_config.items():
    board.set_pin_mode_servo(pin)
    board.servo_write(pin, PAN_NEUTRAL)

current_pan_primary = PAN_NEUTRAL
current_pan_secondary = PAN_NEUTRAL

# Setup deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Setup kamera
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])


# ===================== FUNGSI UTAMA =====================
def calculate_pan_offset(bbox):
    x, _, w, _ = bbox
    frame_center = FRAME_SIZE[0] // 2
    object_center = x + w // 2
    offset = object_center - frame_center

    # Jika dalam zona toleransi, return 0
    if abs(offset) <= DEAD_ZONE_PX:
        return 0.0

    # Hitung offset sudut
    return (offset * CAMERA_FOV_H_DEG) / FRAME_SIZE[0]


def move_servo_smooth(pin, target, current_pos, min_max):
    min_, max_ = min_max
    error = target - current_pos

    # Hitung step dengan limit kecepatan
    step = KP * error
    step = np.clip(step, -MAX_STEP, MAX_STEP)

    if abs(step) > 0.5:
        new_pos = current_pos + step
        new_pos = np.clip(new_pos, min_, max_)
        board.servo_write(pin, int(new_pos))
        return new_pos
    return current_pos


def adjust_horizontal(angle_offset):
    global current_pan_primary, current_pan_secondary

    if angle_offset == 0.0:
        # Kembalikan servo sekunder ke netral jika perlu
        if current_pan_secondary != PAN_NEUTRAL:
            step = -np.sign(current_pan_secondary - PAN_NEUTRAL) * MAX_STEP
            current_pan_secondary = move_servo_smooth(
                SERVO_PAN_SECONDARY_PIN,
                current_pan_secondary + step,
                current_pan_secondary,
                PAN_SECONDARY_RANGE,
            )
        return

    microsec_needed = -angle_offset * 11  # Konversi derajat ke mikrodetik

    # Gerakkan servo utama
    primary_target = current_pan_primary + microsec_needed
    primary_actual = move_servo_smooth(
        SERVO_PAN_PRIMARY_PIN, primary_target, current_pan_primary, PAN_PRIMARY_RANGE
    )

    # Hitung sisa gerakan yang diperlukan
    remaining = primary_target - primary_actual

    # Gerakkan servo sekunder jika diperlukan
    if remaining != 0:
        secondary_target = current_pan_secondary + remaining
        current_pan_secondary = move_servo_smooth(
            SERVO_PAN_SECONDARY_PIN,
            secondary_target,
            current_pan_secondary,
            PAN_SECONDARY_RANGE,
        )

    current_pan_primary = primary_actual


def tracking_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi ke grayscale dan deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        if len(faces) > 0:
            # Ambil wajah terbesar
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            # Gambar kotak dan UI
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(
                frame,
                (FRAME_SIZE[0] // 2 - DEAD_ZONE_PX, 0),
                (FRAME_SIZE[0] // 2 + DEAD_ZONE_PX, FRAME_SIZE[1]),
                (0, 0, 255),
                1,
            )

            # Hitung dan sesuaikan posisi
            pan_angle = calculate_pan_offset((x, y, w, h))
            adjust_horizontal(pan_angle)

            # Tampilkan informasi
            cv2.putText(
                frame,
                f"Primary: {current_pan_primary}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Secondary: {current_pan_secondary}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Dead Zone: {DEAD_ZONE_PX*2}px",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Smart Pan Tracking", frame)
        if cv2.waitKey(1) == ord("q"):
            break


# ===================== MAIN PROGRAM =====================
try:
    print("Starting Smart Pan Tracking System...")
    print(f"Dead Zone: {DEAD_ZONE_PX*2} pixels")
    print("Press 'q' in window to quit")

    tracking_thread = threading.Thread(target=tracking_loop)
    tracking_thread.start()

    while tracking_thread.is_alive():
        tracking_thread.join(timeout=1)

except KeyboardInterrupt:
    print("Shutting down by user request...")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    board.servo_write(SERVO_PAN_PRIMARY_PIN, PAN_NEUTRAL)
    board.servo_write(SERVO_PAN_SECONDARY_PIN, PAN_NEUTRAL)
    time.sleep(0.5)
    board.shutdown()
    print("System shutdown complete")

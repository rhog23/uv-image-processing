# smart_pan_tilt_system/config.py

"""
Configuration settings for the Smart Pan Tracking System.
Includes settings for unified face detection (frontal + profile).
"""

# ===================== KAMERA =====================
CAMERA_ID: int = 0  # Camera device ID (try 0, 1, 2, etc.)
CAMERA_FOV_H_DEG: float = 60.0  # Horizontal Field of View (degrees)
FRAME_SIZE: tuple[int, int] = (640, 480)  # Lebar, Tinggi frame
DEAD_ZONE_PERCENT: float = 0.07  # 7% toleransi dari lebar frame

# ===================== SERVO HORIZONTAL =====================
SERVO_PAN_PRIMARY_PIN: int = 9
SERVO_PAN_SECONDARY_PIN: int = 3

PAN_NEUTRAL_PRIMARY: int = 1500
PAN_NEUTRAL_SECONDARY: int = 1500

PAN_PRIMARY_RANGE: tuple[int, int] = (500, 2500)  # Untuk servo 270°
PAN_SECONDARY_RANGE: tuple[int, int] = (1000, 2000)  # Untuk servo 180°

# ===================== KONTROL PERGERAKAN =====================
KP: float = 0.4  # Gain proporsional
MAX_STEP: float = 3.0  # Maksimum step (perubahan pulse width) per update
ANGLE_TO_MICROSEC_FACTOR: float = (
    11.0  # Konversi derajat ke mikrodetik (perlu kalibrasi)
)

# ===================== DETEKSI WAJAH (HAAR CASCADE) =====================
# Path ke file XML Haar Cascade
# OpenCV biasanya menyertakan ini. Jika tidak, unduh dan letakkan di proyek.
HAAR_CASCADE_FRONTAL_FILENAME: str = "haarcascade_frontalface_default.xml"
HAAR_CASCADE_PROFILE_FILENAME: str = "haarcascade_profileface.xml"

# Parameter Deteksi Wajah (dapat disesuaikan untuk performa/akurasi)
# Untuk Frontal Face Detector
FRONTAL_SCALE_FACTOR: float = 1.2
FRONTAL_MIN_NEIGHBORS: int = 4

# Untuk Profile Face Detector
PROFILE_SCALE_FACTOR: float = 1.2
PROFILE_MIN_NEIGHBORS: int = 4  # Mungkin perlu lebih rendah untuk profil

# Ukuran minimum wajah yang akan dideteksi
MIN_FACE_SIZE: tuple[int, int] = (40, 40)  # (lebar, tinggi) dalam pixel

# ===================== NON-MAXIMUM SUPPRESSION (NMS) =====================
# Threshold untuk IoU (Intersection over Union) dalam NMS.
# Nilai lebih rendah -> NMS lebih agresif (lebih banyak box digabung/dihilangkan)
# Nilai lebih tinggi -> NMS kurang agresif (lebih banyak box dipertahankan)
NMS_OVERLAP_THRESHOLD: float = 0.3

# ===================== NILAI TERKOMPUTASI (JANGAN DIUBAH MANUAL) =====================
# Hitung dead zone dalam pixel berdasarkan persentase
DEAD_ZONE_PX: int = int(FRAME_SIZE[0] * DEAD_ZONE_PERCENT / 2)

import time
import cv2 as cv
import sys
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology, segmentation, color, measure

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 160)
cap.set(cv.CAP_PROP_FPS, 36)

if not cap.isOpened():
    print("[INFO] Unable to open camera ...")
    sys.exit()

previous_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("[INFO] Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # gray_eq = exposure.equalize_adapthist(gray)
    thresh = filters.threshold_otsu(gray)
    bw = morphology.closing(gray > thresh, morphology.disk(2.6))
    cleared = segmentation.clear_border(bw)
    cleared = morphology.opening(cleared, morphology.disk(1.8))
    label_image = measure.label(cleared)
    for region in measure.regionprops(label_image):
        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox
            rect = cv.rectangle(frame, (minc, minr), (maxc, maxr), (255, 0, 0), 1)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - previous_frame_time)
    previous_frame_time = new_frame_time

    fps = str(int(fps))

    cv.putText(
        frame,
        fps,
        (0, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        3,
        cv.LINE_AA,
    )

    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

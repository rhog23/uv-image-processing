import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 320)

while True:
    ret, frame = cap.read()
    # low_b = np.uint8([0, 0, 0])
    # high_b = np.uint8([80, 80, 80])
    # mask = cv2.inRange(frame, low_b, high_b)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        # print(c)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            print("cx:" + str(cx) + " cy:" + str(cy))
            if cx >= 120:
                print("turn left")
            if cx > 40 and cx < 120:
                print("on track")
            if cx <= 40:
                print("turn right")

            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
    cv2.imshow("mask", thresh)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

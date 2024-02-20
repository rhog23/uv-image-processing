import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 160)
cap.set(4, 120)

while True:
    ret, frame = cap.read()
    low_b = np.uint8([5, 5, 5])
    high_b = np.uint8([0, 0, 0])
    mask = cv2.inRange(frame, high_b, low_b)
    contours, hierarchy = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            print("cx:"+str(cx) + " cy:" + str(cy))
            if cx >= 120:
                print("turn left")
            if cx > 40 and cx < 120:
                print("on track")
            if cx <= 40:
                print("turn right")
            
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
    
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
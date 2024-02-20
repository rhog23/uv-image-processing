import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 160)
cap.set(4, 120)

while True:
    ret, frame = cap.read()
    blackline = cv2.inRange(frame, (0,0,0), (60,60,60))
    kernel = np.ones((3,3), np.uint8)
    blackline = cv2.erode(blackline, kernel, iterations=5)
    blackline = cv2.dilate(blackline, kernel, iterations=9)
    contours_blk, hierarchy_blk = cv2.findContours(blackline.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_blk) > 0:
        blackbox = cv2.minAreaRect(contours_blk[0])
        (x_min, y_min), (w_min, h_min), ang = blackbox

        if ang < -45:
            ang = 90 + ang
        if w_min < h_min and ang > 0:
            ang = (90-ang) * -1
        if w_min > h_min and ang < 0:
            ang = 90 + ang
        setpoint = 320
        error = int(x_min - setpoint)
        ang = int(ang)
        box = cv2.boxPoints(blackbox)
        box = np.int64(box)
        cv2.drawContours(frame,[box],0,(0,0,255),3)
        cv2.putText(frame,str(ang),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame,str(error),(10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(frame, (int(x_min),200 ), (int(x_min),250 ), (255,0,0),3)
    
    cv2.imshow("orginal with line", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
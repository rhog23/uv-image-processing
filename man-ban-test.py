from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("best.pt", task="detect")

results = model.predict("Bananavarieties.jpg")

for result in results:
    img = result.plot()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open("../images/boneka-clean.png")
print(im.histogram())
plt.hist(im.histogram(), bins=256)
plt.show()

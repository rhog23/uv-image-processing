import matplotlib.pyplot as plt

im = plt.imread("../images/1.png")
print(f"{'Image Shape':<20}:{im.shape}")
print(f"{'Image Data Type':<20}:{im.dtype}")
print(f"{'Type':<20}:{type(im)}")
plt.title("Example Image")
plt.imshow(im)
# plt.axis("off")
plt.show()

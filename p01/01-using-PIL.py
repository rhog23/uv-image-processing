from PIL import Image

im = Image.open("../images/1.png")
print(f"{'Image Size':<15}:{im.size}")
print(f"{'Image Width':<15}:{im.width}")
print(f"{'Image Height':<15}:{im.height}")
print(f"{'Image Mode':<15}:{im.mode}")
im.show()  # displays image

from PIL import Image

im = Image.open("../images/1.png")
im_g = im.convert("L")
im_g.save("../images/1-gray.png")
print(Image.open("../images/1-gray.png").mode)

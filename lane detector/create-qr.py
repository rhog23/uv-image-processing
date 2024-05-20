import pyqrcode

qr_1 = pyqrcode.create("1")
qr_2 = pyqrcode.create("2")
qr_3 = pyqrcode.create("3")
qr_1.png("./lane detector/1-qr.png", scale=5)
qr_2.png("./lane detector/2-qr.png", scale=5)
qr_3.png("./lane detector/3-qr.png", scale=5)

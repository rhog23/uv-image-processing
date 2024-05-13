import easyocr

reader = easyocr.Reader(["en"])

result = reader.readtext("lane detector/pyimagesearch_address-768x399.jpg")

print(result)
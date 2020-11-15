import qrcode

#TODO image name change
def generate_qr_code(data="https://habr.com", img_name="habr.png"):
    img = qrcode.make(data)  # generate QRcode
    img.save(img_name)
    return img
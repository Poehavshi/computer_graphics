from PIL import Image


def save_image(matrix, path):
    img = Image.fromarray(matrix)
    img.save(path)

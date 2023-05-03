import numpy as np


def draw_star_at_center(image: np.ndarray, color: tuple, line_method: callable):
    """
    Creates a star in the center of the image

    :param image: image to draw on
    :param color: color of the star
    :param line_method: method to create a line
    :return: image with star
    """
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    for i in range(0, 13):
        angle = 2 * np.pi * i / 13
        x = int(center_x + 95 * np.cos(angle))
        y = int(center_y + 95 * np.sin(angle))
        line_method(center_x, center_y, x, y, image, color)
    return image

from task1 import create_matrix_full_of_value
from saving_utils import save_image
import numpy as np
import os
from config import OUTPUT_PATH
from config import HEIGHT, WIDTH


def create_simple_line(x0: int, y0: int, x1: int, y1: int, image: np.ndarray, color: tuple = (255, 255, 255)):
    for t in np.arange(0, 1, 0.01):
        x = int(x0 * (1 - t) + t * x1)
        y = int(y0 * (1 - t) + t * y1)
        image[y, x] = color
    return image


def create_advanced_line(x0: int, y0: int, x1: int, y1: int, image: np.ndarray, color: tuple = (255, 255, 255)):
    # Algorithm works only for x0 < x1
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = int(y0 * (1 - t) + y1 * t)
        image[y, x] = color
    return image


def _correct_x_y(x0: int, y0: int, x1: int, y1: int, steep):
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:  # make it left−to−right
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def create_line_with_steep_without_error(x0: int, y0: int, x1: int, y1: int,
                                         image: np.ndarray,
                                         color: tuple = (255, 255, 255)):
    steep = abs(x1 - x0) < abs(y1 - y0)
    x0, y0, x1, y1 = _correct_x_y(x0, y0, x1, y1, steep)
    for x in range(x0, x1 + 1):
        t = (x - x0) / (x1 - x0)
        y = int(y0 * (1 - t) + y1 * t)
        if steep:
            image[x, y] = color
        else:
            image[y, x] = color
    return image


def create_line_with_bresenham(x0: int, y0: int, x1: int, y1: int, image: np.ndarray, color: tuple = (255, 255, 255)):
    steep = abs(x1 - x0) < abs(y1 - y0)
    x0, y0, x1, y1 = _correct_x_y(x0, y0, x1, y1, steep)
    dx = x1 - x0
    dy = y1 - y0
    derror = abs(dy) / dx
    error = 0
    y = y0
    for x in range(x0, x1 + 1):
        if steep:
            image[x, y] = color
        else:
            image[y, x] = color
        error += derror
        if error > 0.5:
            y += 1 if y1 > y0 else -1
            error -= 1
    return image


def create_star(image: np.ndarray, color: tuple, create_line=create_simple_line):
    """
    Creates a star in the center of the image

    :param image: image to draw on
    :param color: color of the star
    :param create_line: method to create a line
    :return: image with star
    """
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    for i in range(0, 13):
        angle = 2 * np.pi * i / 13
        x = int(center_x + 95 * np.cos(angle))
        y = int(center_y + 95 * np.sin(angle))
        create_line(center_x, center_y, x, y, image, color)
    return image


def save_star_image_in_fs(size, path, color=(255, 255, 255), create_line=create_simple_line):
    image = create_matrix_full_of_value(size, value=(0, 0, 0))
    create_star(image, color=color, create_line=create_line)
    save_image(image, path)


if __name__ == '__main__':
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    save_star_image_in_fs((HEIGHT, WIDTH, 3), os.path.join(OUTPUT_PATH, "star.png"))
    save_star_image_in_fs((HEIGHT, WIDTH, 3), os.path.join(OUTPUT_PATH, "star_advanced.png"),
                          create_line=create_advanced_line)
    save_star_image_in_fs((HEIGHT, WIDTH, 3), os.path.join(OUTPUT_PATH, "star_steep.png"),
                          create_line=create_line_with_steep_without_error)
    save_star_image_in_fs((HEIGHT, WIDTH, 3), os.path.join(OUTPUT_PATH, "star_bresenham.png"),
                          create_line=create_line_with_bresenham)

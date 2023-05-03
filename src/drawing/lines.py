import numpy as np
from configs.colors import WHITE


def draw_line(x0: int, y0: int, x1: int, y1: int,
              image: np.ndarray,
              color: tuple = WHITE):
    return draw_line_with_bresenham(x0, y0, x1, y1, image, color)


def draw_simple_line(x0: int, y0: int, x1: int, y1: int,
                     image: np.ndarray,
                     color: tuple = WHITE):
    for t in np.arange(0, 1, 0.01):
        x = int(x0 * (1 - t) + t * x1)
        y = int(y0 * (1 - t) + t * y1)
        image[y, x] = color
    return image


def draw_advanced_line(x0: int, y0: int, x1: int, y1: int,
                       image: np.ndarray,
                       color: tuple = WHITE):
    # Algorithm works only for x0 < x1
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = int(y0 * (1 - t) + y1 * t)
        image[y, x] = color
    return image


def draw_line_only_with_steep(x0: int, y0: int, x1: int, y1: int,
                              image: np.ndarray,
                              color: tuple = WHITE):
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


def draw_line_with_bresenham(x0: int, y0: int, x1: int, y1: int,
                             image: np.ndarray,
                             color: tuple = (255, 255, 255)):
    steep = abs(x1 - x0) < abs(y1 - y0)
    x0, y0, x1, y1 = _correct_x_y(x0, y0, x1, y1, steep)
    dx = x1 - x0
    if dx == 0:
        return image
    dy = y1 - y0
    d_error = abs(dy) / dx
    error = 0
    y = y0
    for x in range(x0, x1 + 1):
        if steep:
            image[x, y] = color
        else:
            image[y, x] = color
        error += d_error
        if error > 0.5:
            y += 1 if y1 > y0 else -1
            error -= 1
    return image


def _correct_x_y(x0: int, y0: int, x1: int, y1: int, steep):
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:  # make it left−to−right
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    return x0, y0, x1, y1

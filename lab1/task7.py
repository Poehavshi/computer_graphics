from lab1.task11 import calculate_cos_angle_of_light, calculate_normal_for_triangle
from task1 import create_matrix_full_of_value
from PIL import Image


def calculate_baricentric_coords(x0, y0, x1, y1, x2, y2, x, y):
    """
    Calculate baricentric coordinates of point (x, y) in triangle (x0, y0), (x1, y1), (x2, y2)
    """
    try:
        lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
        lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
        lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    except ZeroDivisionError:
        lambda0 = 0
        lambda1 = -1
        lambda2 = 0
    if not 0.9999 < lambda0 + lambda1 + lambda2 < 1.0001:
        lambda0 = 0
        lambda1 = 0
        lambda2 = -1
    return lambda0, lambda1, lambda2


def render_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, image, color=(0, 0, 0), z_buffer=None, l0=None, l1=None, l2=None):
    x_min = min(x0, x1, x2)
    x_max = max(x0, x1, x2)
    y_min = min(y0, y1, y2)
    y_max = max(y0, y1, y2)
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > image.shape[1]:
        x_max = image.shape[1]
    if y_max > image.shape[0]:
        y_max = image.shape[0]

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            lambda0, lambda1, lambda2 = calculate_baricentric_coords(x0, y0, x1, y1, x2, y2, x, y)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z_buffer[y, x] > z:
                    color = list(color)
                    color = [0, 0, 255 * (lambda0*l0 + lambda1*l1 + lambda2*l2)]
                    image[y, x] = color
                    z_buffer[y, x] = z
    return image


def save_image(image_to_save, path):
    image_to_save = Image.fromarray(image_to_save)
    image_to_save.save(path)


if __name__ == '__main__':
    # test function on some white image
    HEIGHT = 1000
    WIDTH = 1000
    BACKGROUND_COLOR = (255, 255, 255)
    POINT_COLOR = (0, 0, 0)
    image = create_matrix_full_of_value((HEIGHT, WIDTH, 3), BACKGROUND_COLOR)
    render_triangle(100, 100, 200, 200, 300, 100, image, POINT_COLOR)
    save_image(image, "output/triangle.png")

    image_with_out_of_bounds = create_matrix_full_of_value((HEIGHT, WIDTH, 3), BACKGROUND_COLOR)
    render_triangle(1100, 100, 200, 200, 300, 300, image_with_out_of_bounds, POINT_COLOR)
    save_image(image_with_out_of_bounds, "output/triangle_out_of_bounds.png")

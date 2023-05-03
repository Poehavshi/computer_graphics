import numpy as np
from src.schema.matrix_image import MatrixImageRGB
from configs.colors import WHITE, BLACK
from src.drawing.lines import draw_line


class ModelRenderer:
    def __init__(self, model):
        self.model = model

        self.ax = 4000
        self.ay = 4000
        self.u0 = None
        self.v0 = None

    def render(self, image) -> MatrixImageRGB:
        if self.v0 is None:
            self.v0 = image.shape[0] // 2
        if self.u0 is None:
            self.u0 = image.shape[1] // 2
        # image.matrix = self._render_points(image.matrix)
        # image.matrix = self._render_edges(image.matrix)
        image.matrix = self._render_faces(image.matrix)
        return image

    def _render_points(self, image):
        for i, point in enumerate(self.model.points):
            x, y, z = point
            x, y, z = projective_transform(x, y, z, self.ax, self.ay, self.u0, self.v0)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y, x] = WHITE
        return image

    def _render_edges(self, image):
        for edge in self.model.edges:
            point1 = self.model.points[edge[0] - 1]
            point2 = self.model.points[edge[1] - 1]
            x1, y1, z1 = projective_transform(point1[0], point1[1], point1[2], self.ax, self.ay, self.u0, self.v0)
            x2, y2, z2 = projective_transform(point2[0], point2[1], point2[2], self.ax, self.ay, self.u0, self.v0)
            if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and 0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]:
                image = draw_line(x1, y1, x2, y2, image, WHITE)
        return image

    def _render_faces(self, image):
        z_buffer = np.random.randint(10_000, 20_000, size=image.shape[:2])
        for face, face_normal in zip(self.model.faces, self.model.faces_normals):
            point1, point2, point3 = [self.model.points[i - 1] for i in face]
            x1, y1, z1 = point1
            x2, y2, z2 = point2
            x3, y3, z3 = point3

            normal = calculate_normal_for_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3)
            cos_angle_of_light = calculate_cos_angle_of_light(normal, light_vector=(0, 0, 1))
            BLUE = (0, 0, 255)
            color = (0, 0, int(BLUE[2] * cos_angle_of_light))

            normal_1, normal_2, normal_3 = [self.model.points_normals[i - 1] for i in face_normal]

            l0 = calculate_cos_angle_of_light(normal_1, light_vector=(0, 0, 1))
            l1 = calculate_cos_angle_of_light(normal_2, light_vector=(0, 0, 1))
            l2 = calculate_cos_angle_of_light(normal_3, light_vector=(0, 0, 1))

            x1, y1, z1 = projective_transform(x1, y1, z1, self.ax, self.ay, self.u0, self.v0)
            x2, y2, z2 = projective_transform(x2, y2, z2, self.ax, self.ay, self.u0, self.v0)
            x3, y3, z3 = projective_transform(x3, y3, z3, self.ax, self.ay, self.u0, self.v0)

            if cos_angle_of_light > 0:
                render_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, image, color=color, z_buffer=z_buffer, l0=l0, l1=l1,
                                l2=l2)

        return image


def projective_transform(x, y, z, ax, ay, u0=500, v0=500, tz=1):
    """
    Transforms 3D point to 2D point using projective transformation.
    [u v 1] ~ [ax 0 u0 0 ay v0 0 0 1][[x y z] + [0, 0, tz]]
    """
    z = z + tz
    u = ax * x / z + u0
    v = ay * y / z + v0
    return int(u), int(v), int(z)


def calculate_normal_for_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    """
    Calculate normal for triangle (x0, y0, z0), (x1, y1, z1), (x2, y2, z2)
    """
    v1 = (x1 - x0, y1 - y0, z1 - z0)
    v2 = (x2 - x0, y2 - y0, z2 - z0)
    normal = (v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0])
    return normal


def calculate_cos_angle_of_light(normal, light_vector=(0, 0, 0)):
    """
    Calculate cos angle of light for triangle with normal and light vector
    """
    try:
        cos_angle = (normal[0] * light_vector[0] + normal[1] * light_vector[1] + normal[2] * light_vector[2]) / (
                (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5 * (
                light_vector[0] ** 2 + light_vector[1] ** 2 + light_vector[2] ** 2) ** 0.5)
    except ZeroDivisionError:
        cos_angle = 0
    return cos_angle



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
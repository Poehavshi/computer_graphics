import numpy as np
from PIL import Image

from configs.config import INPUT_PATH, OUTPUT_PATH
import os
import logging

from src.task11 import calculate_normal_for_triangle, calculate_cos_angle_of_light
from src.task7 import calculate_baricentric_coords
from saving_utils import save_image
from job_1 import create_matrix_full_of_value
from src.jobs.task2 import create_line_with_bresenham
from task7 import render_triangle

log = logging.getLogger(__name__)
HEIGHT = 1000
WIDTH = 1000


class Model3d:
    def __init__(self):
        self.points = None
        self.edges = None
        self.faces = None
        self.normals = None
        self.faces_normals = None

        self.ax = 4_000
        self.ay = 4_000
        self.u0 = None
        self.v0 = None

    def from_file(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            self.points = []
            self.faces = []
            self.edges = []
            self.normals = []
            self.faces_normals = []

            for line in lines:
                if line.startswith('v '):
                    x, y, z = line[2:].split(' ')
                    self.points.append((float(x), -float(y), float(z)))
                elif line.startswith('f '):
                    v1, v2, v3 = line[2:].split(' ')
                    num_of_point1, num_of_point2, num_of_point3 = map(int, (
                        v1.split('/')[0], v2.split('/')[0], v3.split('/')[0]))

                    num_of_normal1, num_of_normal2, num_of_normal3 = map(int, (
                        v1.split('/')[2], v2.split('/')[2], v3.split('/')[2]))

                    self.faces.append(((num_of_point1, num_of_normal1), (num_of_point2, num_of_normal2),
                                       (num_of_point3, num_of_normal3)))
                    self.edges.append((num_of_point1, num_of_point2))
                    self.edges.append((num_of_point1, num_of_point3))
                    self.edges.append((num_of_point2, num_of_point3))
                elif line.startswith("vn "):
                    x, y, z = line[3:].split(' ')
                    self.normals.append((float(x), float(y), float(z)))

        log.info(f'Loaded {len(self.points)} points from {path} file')

    def render_on_image(self, matrix_with_image, coefficient=4000, shift=500):
        rendered_image = matrix_with_image.copy()
        self.v0 = rendered_image.shape[0] // 2
        self.u0 = rendered_image.shape[1] // 2
        self.rotate(0, 0, 0)
        self._render_points(rendered_image, coefficient, shift)
        self._render_edges(rendered_image, coefficient, shift)
        self._render_faces(rendered_image, coefficient, shift)
        return rendered_image

    def _render_points(self, image, coefficient=4000, shift=500):
        for i, point in enumerate(self.points):
            x, y, z = point
            x, y, z = map(int, projective_transform(x, y, z, self.ax, self.ay, self.u0, self.v0))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y, x] = (0, 0, 0)
        return image

    def _render_edges(self, image, coefficient=4000, shift=500):
        for edge in self.edges:
            point1 = self.points[edge[0] - 1]
            point2 = self.points[edge[1] - 1]
            x1, y1, z1 = point1
            x2, y2, z2 = point2
            # x1 = int(x1 * coefficient + shift)
            # y1 = int(y1 * coefficient + shift)
            # z1 = int(z1 * coefficient + shift)
            # x2 = int(x2 * coefficient + shift)
            # y2 = int(y2 * coefficient + shift)
            # z2 = int(z2 * coefficient + shift)

            x1, y1, z1 = map(int, projective_transform(x1, y1, z1, self.ax, self.ay, self.u0, self.v0))
            x2, y2, z2 = map(int, projective_transform(x2, y2, z2, self.ax, self.ay, self.u0, self.v0))
            if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and \
                    0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]:
                create_line_with_bresenham(x1, y1, x2, y2, image, (0, 0, 0))
        return image

    def rotate(self, x_angle, y_angle, z_angle):
        x_angle = np.deg2rad(x_angle)
        y_angle = np.deg2rad(y_angle)
        z_angle = np.deg2rad(z_angle)
        x_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(x_angle), -np.sin(x_angle)],
            [0, np.sin(x_angle), np.cos(x_angle)]
        ])
        y_matrix = np.array([
            [np.cos(y_angle), 0, np.sin(y_angle)],
            [0, 1, 0],
            [-np.sin(y_angle), 0, np.cos(y_angle)]
        ])
        z_matrix = np.array([
            [np.cos(z_angle), -np.sin(z_angle), 0],
            [np.sin(z_angle), np.cos(z_angle), 0],
            [0, 0, 1]
        ])
        rotation_matrix = np.dot(x_matrix, np.dot(y_matrix, z_matrix))
        for i, point in enumerate(self.points):
            x, y, z = point
            x, y, z = np.dot(rotation_matrix, np.array([x, y, z]))
            self.points[i] = (x, y, z)
        return self

    def _render_faces(self, image, coefficient, shift):
        z_buffer = np.random.randint(10_000, 20_000, size=(HEIGHT, WIDTH))
        for face in self.faces:
            point1 = self.points[face[0][0] - 1]
            point2 = self.points[face[1][0] - 1]
            point3 = self.points[face[2][0] - 1]
            x1, y1, z1 = point1
            x2, y2, z2 = point2
            x3, y3, z3 = point3

            normal = calculate_normal_for_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3)
            cos_angle_of_light = calculate_cos_angle_of_light(normal, light_vector=(0, 0, 1))
            BLUE = (0, 0, 255)
            color = (0, 0, int(BLUE[2] * cos_angle_of_light))

            normal_1 = self.normals[face[0][1] - 1]
            normal_2 = self.normals[face[1][1] - 1]
            normal_3 = self.normals[face[2][1] - 1]

            l0 = calculate_cos_angle_of_light(normal_1, light_vector=(0, 0, 1))
            l1 = calculate_cos_angle_of_light(normal_2, light_vector=(0, 0, 1))
            l2 = calculate_cos_angle_of_light(normal_3, light_vector=(0, 0, 1))

            x1, y1, z1 = projective_transform(x1, y1, z1, self.ax, self.ay, self.u0, self.v0)
            x2, y2, z2 = projective_transform(x2, y2, z2, self.ax, self.ay, self.u0, self.v0)
            x3, y3, z3 = projective_transform(x3, y3, z3, self.ax, self.ay, self.u0, self.v0)
            # x1 = int(x1 * coefficient + shift)
            # y1 = int(y1 * coefficient + shift)
            # z1 = int(z1 * coefficient + shift)
            # x2 = int(x2 * coefficient + shift)
            # y2 = int(y2 * coefficient + shift)
            # z2 = int(z2 * coefficient + shift)
            # x3 = int(x3 * coefficient + shift)
            # y3 = int(y3 * coefficient + shift)
            # z3 = int(z3 * coefficient + shift)

            if cos_angle_of_light > 0:
                render_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, image, color=color, z_buffer=z_buffer, l0=l0, l1=l1,
                                l2=l2)


def projective_transform(x, y, z, ax, ay, u0=500, v0=500, tz=1):
    """
    Transforms 3D point to 2D point using projective transformation.
    [u v 1] ~ [ax 0 u0 0 ay v0 0 0 1][[x y z] + [0, 0, tz]]
    """
    z = z + tz
    u = ax * x / z + u0
    v = ay * y / z + v0
    return int(u), int(v), int(z)


def projective_transform2(x, y, z, ax, ay, u0, v0, tz=0.25):
    """
    Transforms 3D point to 2D point using projective transformation.
    [u v 1] ~ [ax 0 u0 0 ay v0 0 0 1][[x y z] + [0, 0, tz]]
    """
    za = z
    u = ax * x / (z + tz) + u0
    v = ay * y / (z + tz) + v0
    return int(u), int(v), int(za)


def save_rendered_image(model3d, image, path, coefficient=4000, shift=500):
    image = model3d.render_on_image(image, coefficient, shift)
    save_image(image, path)


def do_experiment_with_obj(path):
    model = Model3d()
    model.from_file(path)
    blank_image = create_matrix_full_of_value((HEIGHT, WIDTH, 3), (255, 255, 255))
    save_rendered_image(model, blank_image, os.path.join(OUTPUT_PATH, 'task3', 'model_1_50.png'), 50)
    save_rendered_image(model, blank_image, os.path.join(OUTPUT_PATH, 'task3', 'model_1_100.png'), 100)
    save_rendered_image(model, blank_image, os.path.join(OUTPUT_PATH, 'task3', 'model_1_500.png'), 500)
    save_rendered_image(model, blank_image, os.path.join(task3_dirname, 'model_1_4000.png'), 4000)


texture = Image.open('../data/input/texture.jpg')
texturing_parameters = []
array_of_numbers_parameters = []
array_of_numbers_normals = []
with open('../data/input/model_1.obj', 'r') as file:
    data = file.read()
    lines = data.splitlines()
    for line in lines:
        word = line.split()
        if word[0] == 'vt':
            parameters = []
            parameters.append(float(word[1]))
            parameters.append(float(word[2]))
            texturing_parameters.append(parameters)
        if word[0] == 'f':
            parameters = []
            list_normals = []
            for i in range(1, len(word)):
                parameters.append(int(word[i].split('/')[1]))
                list_normals.append(int(word[i].split('/')[2]))
            array_of_numbers_parameters.append(parameters)
            array_of_numbers_normals.append(list_normals)


def extra_drawing(x0, x1, x2, y0, y1, y2, z0, z1, z2, u0, v0, u1, v1, u2, v2, normal_1, normal_2, normal_3):
    x_min, y_min = min(x0, x1, x2), min(y0, y1, y2)
    x_max, y_max = max(x0, x1, x2), max(y0, y1, y2)
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > 2000:  # 2000 - size
        x_max = 2000
    if y_max > 2000:
        y_max = 2000

    l0 = calculate_cos_angle_of_light(normal_1, light_vector=(0, 0, 1))
    l1 = calculate_cos_angle_of_light(normal_2, light_vector=(0, 0, 1))
    l2 = calculate_cos_angle_of_light(normal_3, light_vector=(0, 0, 1))

    z_buffer = np.full((2000, 2000), np.inf, dtype=float)

    for i in range(x_min, x_max + 1):
        for j in range(y_min, y_max + 1):
            lambda0, lambda1, lambda2 = calculate_baricentric_coords(x0, y0, x1, y1, x2, y2, i, j)
            if (lambda0 > 0) & (lambda1 > 0) & (lambda2 > 0):
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z <= z_buffer[i, j]:
                    u = 2000 * (lambda0 * u0 + lambda1 * u1 + lambda2 * u2)
                    v = 2000 * (lambda0 * v0 + lambda1 * v1 + lambda2 * v2)
                    color = texture.getpixel((u, v))
                    image_matrix[j, i][0] = color[0] * (abs(lambda0 * l0 + lambda1 * l1 + lambda2 * l2))
                    image_matrix[j, i][1] = color[1] * (abs(lambda0 * l0 + lambda1 * l1 + lambda2 * l2))
                    image_matrix[j, i][2] = color[2] * (abs(lambda0 * l0 + lambda1 * l1 + lambda2 * l2))
                    z_buffer[i, j] = z


def extra():
    for i in range(len(model.faces)):
        point0 = model.points[model.faces[i][0][0] - 1]
        point1 = model.points[model.faces[i][1][0] - 1]
        point2 = model.points[model.faces[i][2][0] - 1]
        x0, y0, z0 = point0
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        normal_1 = model.normals[model.faces[i][0][1] - 1]
        normal_2 = model.normals[model.faces[i][1][1] - 1]
        normal_3 = model.normals[model.faces[i][2][1] - 1]

        x0, y0, z0 = projective_transform2(x0, y0, z0, 2000, 2000, 1000, 1000)
        x1, y1, z1 = projective_transform2(x1, y1, z1, 2000, 2000, 1000, 1000)
        x2, y2, z2 = projective_transform2(x2, y2, z2, 2000, 2000, 1000, 1000)

        u0 = texturing_parameters[array_of_numbers_parameters[i][0] - 1][0]
        v0 = texturing_parameters[array_of_numbers_parameters[i][0] - 1][1]
        u1 = texturing_parameters[array_of_numbers_parameters[i][1] - 1][0]
        v1 = texturing_parameters[array_of_numbers_parameters[i][1] - 1][1]
        u2 = texturing_parameters[array_of_numbers_parameters[i][2] - 1][0]
        v2 = texturing_parameters[array_of_numbers_parameters[i][2] - 1][1]
        extra_drawing(x0, x1, x2, y0, y1, y2, z0, z1, z2, u0, v0, u1, v1, u2, v2, normal_1, normal_2, normal_3)


def get_model(path):
    model = Model3d()
    model.from_file(path)
    return model


image_matrix = create_matrix_full_of_value((2000, 2000, 3), (255, 255, 255))
model = get_model(os.path.join(INPUT_PATH, 'model_1.obj'))
extra()
image = Image.fromarray(image_matrix, 'RGB')
image.save('extra.jpg')

if __name__ == '__main__':
    task3_dirname = os.path.join(OUTPUT_PATH, 'task3')
    os.makedirs(task3_dirname, exist_ok=True)
    do_experiment_with_obj(os.path.join(INPUT_PATH, 'model_1.obj'))
    # do_experiment_with_obj(os.path.join(INPUT_PATH, 'model_2.obj')) too large

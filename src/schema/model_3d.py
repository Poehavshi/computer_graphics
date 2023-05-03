from dataclasses import dataclass, field
import logging

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Model3d:
    points: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    points_normals: list = field(default_factory=list)
    faces_normals: list = field(default_factory=list)

    textures: list = field(default_factory=list)
    faces_textures: list = field(default_factory=list)

    def add_point(self, point):
        self.points.append(point)

    def add_edge(self, edge):
        self.edges.append(edge)

    def add_face(self, face):
        self.faces.append(face)

    def add_point_normal(self, normal):
        self.points_normals.append(normal)

    def add_faces_normal(self, normal):
        self.faces_normals.append(normal)

    def add_face_texture(self, face_texture):
        self.faces_textures.append(face_texture)

    def add_texture(self, texture):
        self.textures.append(texture)

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

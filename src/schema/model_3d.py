from dataclasses import dataclass, field
import logging


logger = logging.getLogger(__name__)


@dataclass
class Model3d:
    points: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    points_normals: list = field(default_factory=list)
    faces_normals: list = field(default_factory=list)

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

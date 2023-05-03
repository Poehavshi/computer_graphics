from src.schema.model_3d import Model3d
import logging

logger = logging.getLogger(__name__)


class ObjModelReader:
    def __init__(self):
        self.model = Model3d()

    def read(self, path):
        logger.info(f'Reading from {path} file')
        with open(path, 'r') as file:
            self.model = Model3d()
            lines = file.readlines()
            for line in lines:
                self._parse_line(line)
        logger.info(f'Finish reading from {path} file')
        return self.model

    def _parse_line(self, line):
        word = line.split()
        if word[0] == 'v':
            self.model.add_point((float(word[1]), float(word[2]), float(word[3])))
        elif word[0] == 'vn':
            self.model.add_point_normal((float(word[1]), float(word[2]), float(word[3])))
        elif word[0] == 'f':
            face = []
            face_normal = []
            for i in range(1, len(word)):
                face.append(int(word[i].split('/')[0]))
            for i in range(1, len(word)):
                face_normal.append(int(word[i].split('/')[2]))
            self.model.add_face(tuple(face))
            self.model.add_faces_normal(tuple(face_normal))
            self.model.add_edge((face[0], face[1]))
            self.model.add_edge((face[1], face[2]))
            self.model.add_edge((face[2], face[0]))
        elif word[0] == 'vn':
            normal = []
            for i in range(1, len(word)):
                normal.append(int(word[i].split('/')[0]))
            self.model.add_point_normal(tuple(normal))


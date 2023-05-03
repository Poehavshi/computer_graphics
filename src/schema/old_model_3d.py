
class Model3dOLD:
    def __init__(self):
        self.points = None
        self.edges = None
        self.faces = None
        self.normals = None

        self.u0 = None
        self.v0 = None

    def from_file(self, path):
        with open(path, 'r') as file:
            self.points = []
            self.faces = []
            self.edges = []
            self.normals = []
            lines = file.readlines()
            for line in lines:
                self._parse_line(line)
        logger.info(f'Loaded {len(self.points)} points from {path} file')

    def _parse_line(self, line):
        if line.startswith('v '):
            self._parse_vertex(line)
        elif line.startswith('f '):
            self._parse_face(line)
        elif line.startswith("vn "):
            self._parse_normal(line)

    def _parse_face(self, line):
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

    def _parse_normal(self, line):
        x, y, z = line[3:].split(' ')
        self.normals.append((float(x), float(y), float(z)))

    def _parse_vertex(self, line):
        x, y, z = line[2:].split(' ')
        self.points.append((float(x), -float(y), float(z)))

    def render_on_image(self, matrix_with_image):
        rendered_image = matrix_with_image.copy()
        self.v0 = rendered_image.shape[0] // 2
        self.u0 = rendered_image.shape[1] // 2
        self.rotate(0, 0, 0)
        self._render_points(rendered_image)
        # self._render_edges(rendered_image)
        # self._render_faces(rendered_image)
        return rendered_image

    def _render_points(self, image):
        for i, point in enumerate(self.points):
            x, y, z = point
            x, y, z = map(int, projective_transform(x, y, z, self.ax, self.ay, self.u0, self.v0))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y, x] = (0, 0, 0)
        return image

    def _render_edges(self, image):
        for edge in self.edges:
            point1 = self.points[edge[0] - 1]
            point2 = self.points[edge[1] - 1]
            x1, y1, z1 = point1
            x2, y2, z2 = point2

            x1, y1, z1 = map(int, projective_transform(x1, y1, z1, self.ax, self.ay, self.u0, self.v0))
            x2, y2, z2 = map(int, projective_transform(x2, y2, z2, self.ax, self.ay, self.u0, self.v0))
            if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and \
                    0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]:
                draw_line_with_bresenham(x1, y1, x2, y2, image, (0, 0, 0))
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

    def _render_faces(self, image):
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

            if cos_angle_of_light > 0:
                render_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, image, color=color, z_buffer=z_buffer, l0=l0, l1=l1,
                                l2=l2)
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

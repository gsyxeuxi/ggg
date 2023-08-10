import numpy as np
import math

def coordinate_transform(x1, y1, x2, y2, x0, y0):
    cos_theta = (x2 - x1) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # print(cos_theta)

    sin_theta = (y2 - y1) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # print(math.degrees(math.asin(sin_theta)))

    flip_matrix = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]])
    translation_matrix = np.array([[1, 0, x0],
                                   [0, 1, y0],
                                   [0, 0, 1]])
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                [sin_theta, cos_theta, 0],
                                [0, 0, 1]])
    # transform_matrix = np.zeros((3, 3))

    transform_matrix = np.dot(np.dot(translation_matrix, rotation_matrix), flip_matrix)
    # transform_matrix = np.dot(translation_matrix, rotation_matrix)

    # print(transform_matrix)

    # inverse matrix
    inverse_matrix = np.linalg.inv(transform_matrix)

    # print(inverse_matrix)

    return inverse_matrix


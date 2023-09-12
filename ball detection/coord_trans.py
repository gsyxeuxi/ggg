import numpy as np
import math

def coordinate_transform(x1=156, y1=21, x2=396, y2=24, x0=268, y0=260):
    cos_theta = (x2 - x1) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    sin_theta = (y2 - y1) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # print(math.degrees(math.asin(sin_theta)))
    beta = 5/4 * math.pi
    
    flip_matrix = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]])
    translation_matrix = np.array([[1, 0, x0],
                                   [0, 1, y0],
                                   [0, 0, 1]])
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                [sin_theta, cos_theta, 0],
                                [0, 0, 1]])
    rotation_matrix_beta = np.array([[math.cos(beta), -math.sin(beta), 0],
                                [math.sin(beta), math.cos(beta), 0],
                                [0, 0, 1]])
    # transform_matrix = np.zeros((3, 3))

    # transform_matrix = np.dot(np.dot(translation_matrix, rotation_matrix), flip_matrix)
    transform_matrix = np.dot(np.dot(translation_matrix, rotation_matrix), rotation_matrix_beta)
    
    # inverse matrix
    inverse_matrix = np.linalg.inv(transform_matrix)
    # print(inverse_matrix)
    return inverse_matrix


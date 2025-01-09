import random

import numpy as np


def fit_plane_axis_aligned(points, threshold=0.05, max_iterations=300):
    # copied and modified from https://github.com/leomariga/pyRANSAC-3D
    # license: Apache License 2.0
    n_points = points.shape[0]
    best_equation = None
    best_inliers = np.array([], dtype=int)

    for _ in range(max_iterations):
        sample_ids = random.sample(range(n_points), 2)
        sample_points = points[sample_ids]

        vector_a = sample_points[1] - sample_points[0]
        vector_b = np.array([0.0, 0.0, 1.0])

        vector_c = np.cross(vector_a, vector_b)
        norm_c = np.linalg.norm(vector_c)
        if norm_c < 1e-12:
            continue

        vector_c /= norm_c
        d = -np.dot(vector_c, sample_points[1])
        plane_equation = np.array([vector_c[0], vector_c[1], vector_c[2], d])

        distances = (
            plane_equation[0] * points[:, 0]
            + plane_equation[1] * points[:, 1]
            + plane_equation[2] * points[:, 2]
            + plane_equation[3]
        ) / np.linalg.norm(plane_equation[:3])

        inlier_ids = np.where(np.abs(distances) <= threshold)[0]

        if len(inlier_ids) > len(best_inliers):
            best_equation = plane_equation
            best_inliers = inlier_ids

    return best_equation, best_inliers

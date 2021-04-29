import numpy as np


def undistort(p_d, camera_K, k1, k2, k3, p1, p2):
    p_dp = normalize(p_d, camera_K)
    p_up = p_dp
    for i in range(5):
        err = distort_normal(p_up, k1, k2, k3, p1, p2) - p_dp
        p_up = p_up - err
        # if (err < err_threshold):
        #     break
    p_u = denormalize(p_up, camera_K)
    return p_u


def normalize(p_d, camera_K):
    y_n = (p_d[1] - camera_K[1, 2]) / camera_K[1, 1]
    x_n = (p_d[0] - camera_K[0, 2]) / camera_K[0, 0]
    return np.array([x_n, y_n])


def denormalize(p_up, camera_K):
    x_p = camera_K[0, 0] * (p_up[0]) + camera_K[0, 2]
    y_p = camera_K[1, 1] * p_up[1] + camera_K[1, 2]
    return np.array([x_p, y_p])


def distort_normal(p_up, k1, k2, k3, p1, p2):
    r2 = p_up[0] ** 2 + p_up[1] ** 2
    radial_d = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_d = radial_d * p_up[0] + 2 * p1 * p_up[0] * p_up[1] + p2 * (r2 + 2 * p_up[0] ** 2)
    y_d = radial_d * p_up[1] + p1 * (r2 + p_up[1] ** 2) + 2 * p2 * p_up[0] * p_up[1]
    return np.array([x_d, y_d])
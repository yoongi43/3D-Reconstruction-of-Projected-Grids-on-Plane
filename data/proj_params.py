import numpy as np

"""Projector matrix"""
k1 = -0.08679
k2 = 0.21450
k3 = -0.00642
p1 = -0.00597
p2 = 0.0

K = np.array([[1535.01, 0, 218.09],
              [0, 1543.99, 604.23],
              [0, 0, 1]])

K_inv = np.linalg.inv(K)

R = np.array([[0.7354, 0.0294, -0.6770],
              [0.0454, 0.9947, 0.0926],
              [0.6761, -0.0988, 0.7302]])

R_inv = np.linalg.inv(R)

t = np.array([398.2093, -134.1070, 113.4529])
C = -R.T @ t

P = K @ R @ np.append(np.identity(3), -C.reshape(3,1), axis=1)
P_pinv = P.T @ np.linalg.inv(P @ P.T)






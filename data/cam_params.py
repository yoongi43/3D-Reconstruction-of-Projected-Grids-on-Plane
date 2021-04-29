import numpy as np
from scipy import io

k1 = -0.09924
k2 = 0.01384
k3 = 0.00164
p1 = 0.00081
p2 = 0.0

K = np.array([[2334.32, 0, 1528.16],
              [0, 2333.86, 971.93],
              [0, 0, 1]])
K_inv = np.linalg.inv(K)

rotations = io.loadmat('./data/rotations.mat')
translations = io.loadmat('./data/translations.mat')

R = rotations['Rc_1']
R_inv = np.linalg.inv(R)
t = translations['Tc_1']
C = -R.T@t

P = K @ R @ np.append(np.identity(3), -C.reshape(3,1), axis=1)
P_pinv = P.T @ np.linalg.inv(P @ P.T)



import cv2
import numpy as np
from scipy import io
from undistort import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from data.make_grids import *
from data import cam_params as cp
from data import proj_params as pp
from undistort import *


def main():

    Origin = 10
    interval = 30
    width = 800
    height = 600

    CONNECTIVITY = 4
    DRAW_CIRCLE_RADIUS = 4
    THRESHOLD = 180

    img_camera = cv2.imread('./data/Picture.jpg')  # Image taken by camera

    _, grids_proj_coords, num_grids_w, num_grids_h = make_grids(interval=interval, width=width, height=height,
                                                           save_path="./data/grids.jpg", save=False)
    num_grids = num_grids_h * num_grids_w

    # Projector matrix world coordinate로 바꿔야 함

    proj_R_world = pp.R @ cp.R
    proj_C = pp.C.reshape(3)
    proj_C_world = cp.R.T @ pp.C + cp.C
    proj_P_world = pp.K @ proj_R_world @ np.hstack([np.identity(3), -proj_C_world.reshape(3, 1)])
    proj_P_pinv_world = proj_P_world.T @ np.linalg.inv(proj_mtx_world @ proj_mtx_world.T)

    """ projector undistortion"""
    grids_proj_undistort = np.zeros((num_grids, 2))
    for i in range(num_grids):
        grids_proj_undistort[i] = undistort(grids_proj_coords[i], pp.K, pp.k1, pp.k2, pp.k3, pp.p1, pp.p2)











#################
num_dots_x = 27  # width 방향 점 개수
num_dots_y = 20  # height 방향 점 개수
num_dots = num_dots_y * num_dots_x

grids_proj = cv2.imread('./data/grids.jpg')  # 프로젝트 된 grid
img_camera = cv2.imread('./data/Picture.jpg')  # 카메라로 찍은 이미지
img_gray = cv2.cvtColor(img_camera, cv2.COLOR_BGR2GRAY)

height = img_gray.shape[0]
width = img_gray.shape[1]

# 기타 변수들
Origin = 10
interval = 30
CONNECTIVITY = 4
DRAW_CIRCLE_RADIUS = 4
THRESHOLD = 180

grids_proj = np.zeros((num_dots, 2))  # projected 된 grids 좌표

for i in range(num_dots_y):
    for j in range(num_dots_x):
        grids_proj[i * num_dots_x + j] = np.array([j, i]) * interval  # (x좌표, y좌표) grids_proj에 저장

"""Camera Matrix"""
k1_c = -0.09924
k2_c = 0.01384
k3_c = 0.00164
p1_c = 0.00081
p2_c = 0.0
distCoeff_c = np.float64([k1_c, k2_c, p1_c, p2_c])

camera_K_init = np.array([[2413.90, 0, 1459.5],
                           [0, 2413.90, 999.5],
                           [0, 0, 1]])
camera_K_opt = np.array([[2334.32, 0, 1528.16],
                         [0, 2333.86, 971.93],
                         [0, 0, 1]])
camera_K = camera_K_opt
camera_K_inv = np.linalg.inv(camera_K)

# Calibration시 카메라들의 rotation과 center
rotations = io.loadmat('./data/rotations.mat')
translations = io.loadmat('./data/translations.mat')

camera_R = rotations['Rc_1']
camera_R_inv = np.linalg.inv(camera_R)
camera_t = translations['Tc_1']
camera_C = -camera_R.T@camera_t

cam_mtx = camera_K @ camera_R @ np.append(np.identity(3), -camera_C.reshape(3,1), axis=1)
cam_mtx_pseudoinv = cam_mtx.T @ np.linalg.inv(cam_mtx @ cam_mtx.T)

"""Projector matrix"""
k1_p = -0.08679
k2_p = 0.21450
k3_p = -0.00642
p1_p = -0.00597
p2_p = 0.0
distCoeff_p = np.float64([k1_p, k2_p, p1_p, p2_p])

projector_K_init = np.array([[1328.92, 0, 399.5],
                            [0, 1328.92, 299.5],
                            [0, 0, 1]])
projector_K_opt = np.array([[1535.01, 0, 218.09],
                            [0, 1543.99, 604.23],
                            [0, 0, 1]])
projector_K_opt2 = np.array([[1625.01, 0, 399.5],
                            [0, 1625.01, 299.5],
                            [0, 0, 1]])
projector_K = projector_K_opt
projector_K_inv = np.linalg.inv(projector_K)
projector_R = np.array([[0.7354, 0.0294, -0.6770],
                        [0.0454, 0.9947, 0.0926],
                        [0.6761, -0.0988, 0.7302]])
projector_R_inv = np.linalg.inv(projector_R)

projector_t = np.array([398.2093, -134.1070, 113.4529])
projector_C = -projector_R.T@projector_t

proj_mtx = projector_K @ projector_R @ np.append(np.identity(3), -projector_C.reshape(3,1), axis=1)
proj_mtx_pseudoinv = proj_mtx.T @ np.linalg.inv(proj_mtx @ proj_mtx.T)

# Projector matrix world coordinate로 바꿔야 함
projector_R_world = projector_R @ camera_R
projector_C = projector_C.reshape(3)
camera_C = camera_C.reshape(3)
projector_C_world = camera_R.T @ projector_C + camera_C

# 조작하기>...!!
# projector_C_world = projector_C_world + np.array([0, -63, 0])


proj_mtx_world = projector_K @ projector_R_world @\
    np.append(np.identity(3), -projector_C_world.reshape(3,1), axis=1)
proj_mtx_world_pseudoinv = proj_mtx_world.T @ np.linalg.inv(proj_mtx_world @ proj_mtx_world.T)

"""z=0평면 reconstruction 해내기"""

"""프로젝터 undistortion"""
grids_proj_undistort = np.zeros((num_dots, 2))
for k in range(num_dots):
    grids_proj_undistort[k] = undistort(grids_proj[k], projector_K, k1_p, k2_p, k3_p, p1_p, p2_p)
print(grids_proj_undistort)

grids_proj_undistort = grids_proj



"""Thresholding"""

_, img_cam_bin = cv2.threshold(img_gray, THRESHOLD, 255, cv2.THRESH_BINARY)
img_connect = img_camera.copy()  # connected component 출력용

ret, labels, stats, grids_cam = cv2.connectedComponentsWithStats(img_cam_bin, CONNECTIVITY, cv2.CV_32S)
grids_cam = np.delete(grids_cam, 0, axis=0)  # 첫 번째 요소(label=0 들의 center) 제거하기. 전체의 center이므로
print("# of grids in camera image : {}".format(len(grids_cam)))
print("# of grids projected : {}".format(num_dots))

for grid in grids_cam:
    cv2.circle(img_connect, (int(grid[0]), int(grid[1])), DRAW_CIRCLE_RADIUS, (0, 0, 255), -1)

# grid 찍은거 보여주기
# cv2.imshow("Grids in Picture", cv2.resize(img_connect, (1200, 900)))
# cv2.imshow("Grids in Picture (binary)", cv2.resize(img_cam_bin, (1200, 900)))
# cv2.waitKey()
# cv2.destroyAllWindows()

# 꼭지점 4개 구하기
grids_sum = grids_cam.sum(axis=1)  # 각 grid 좌표 합
grids_diff = grids_cam[:, 1] - grids_cam[:, 0]  # 각 gird 좌표 차

UL_pt = grids_cam[grids_sum.argmin()]
UR_pt = grids_cam[grids_diff.argmin()]
LR_pt = grids_cam[grids_sum.argmax()]
LL_pt = grids_cam[grids_diff.argmax()]

four_pts = np.array([UL_pt, UR_pt, LR_pt, LL_pt])

# 점 4개 그려서 확인하기
img_four_circles = img_camera.copy()
for enum, pt in enumerate(four_pts):
    cv2.circle(img_four_circles, (int(pt[0]), int(pt[1])), 20, (0, 0, (enum+1)*60), -1)

# 꼭지점 찍은 것 보여주기
# cv2.imshow("Four grid", cv2.resize(img_four_circles, (1200, 900)))
# cv2.waitKey()
# cv2.destroyAllWindows()

"""Proj grid 와 Cam grid matching : Homography 이용"""

four_pts = four_pts.astype(np.float32)
rect_pt = np.array([[Origin, Origin],
                    [Origin + (num_dots_x - 1) * interval, Origin],
                    [Origin + (num_dots_x - 1) * interval, Origin + (num_dots_y - 1) * interval],
                    [Origin, Origin + (num_dots_y - 1) * interval]], dtype=np.float32)

Hom = cv2.getPerspectiveTransform(four_pts, rect_pt)

img_cam_rect = cv2.warpPerspective(img_camera, Hom, (int(800), int(600)))
img_cam_rect_gray = cv2.cvtColor(img_cam_rect, cv2.COLOR_BGR2GRAY)
_, img_cam_rect_bin = cv2.threshold(img_cam_rect_gray, THRESHOLD, 255, cv2.THRESH_BINARY)

# rectified 된 image 보여주기
# cv2.imshow("Rectified Picture", img_cam_rect)
# cv2.waitKey()
# cv2.destroyAllWindows()

img_cam_rect_connect = img_cam_rect.copy()
ret, labels, stats, grids_cam_rect = cv2.connectedComponentsWithStats(img_cam_rect_bin, CONNECTIVITY, cv2.CV_32S)
grids_cam_rect = np.delete(grids_cam_rect, 0, axis=0)
print("# of grids in camera image(rect) : ", len(grids_cam_rect))

# for grid in grids_cam_rect:
#     cv2.circle(img_cam_rect_connect, (int(grid[0]), int(grid[1])), DRAW_CIRCLE_RADIUS, (0, 0, 255), -1)
#     cv2.imshow("Matching with Original image", img_cam_rect_connect)
#     ord_k = cv2.waitKey(0)
#     if ord_k == ord('o'):
#         break
# cv2.destroyAllWindows()

"""grid order 맞추기"""

Hom_inv = np.linalg.inv(Hom)

grids_cam_rect_ordered = np.zeros((num_dots, 2))
grids_cam_ordered = np.zeros((num_dots, 2))
origin_pt = np.array([Origin, Origin])

for y in range(num_dots_y):
    for x in range(num_dots_x):
        close_pt = origin_pt + [x * interval, y * interval]
        close_pt_centered = grids_cam_rect - close_pt
        index_of_ordered_pt = np.linalg.norm(close_pt_centered, axis=1).argmin()  # 거리가 가장 작은 점 선택
        grids_cam_rect_ordered[y * num_dots_x + x] = grids_cam_rect[index_of_ordered_pt]

for k in range(num_dots):
    temp = Hom_inv @ np.append(grids_cam_rect_ordered[k], 1)
    temp = temp / temp[2]
    grids_cam_ordered[k] = np.delete(temp, 2)

img_ordered_grids = img_camera.copy()
# ordered grids 보여주기
# for grid in grids_cam_ordered:
#     cv2.circle(img_ordered_grids, (int(grid[0]), int(grid[1])), DRAW_CIRCLE_RADIUS, (0, 0, 255), -1)
#     cv2.imshow('Matched grids', cv2.resize(img_ordered_grids, (1200, 900)))
#     ord_k = cv2.waitKey(0)
#     if ord_k == ord('o'):
#         break
# cv2.destroyAllWindows()

"""카메라 grids undistort"""
grids_cam_undistort = np.zeros((num_dots, 2))

for k in range(num_dots):
    grids_cam_undistort[k] = undistort(grids_cam_ordered[k], camera_K, k1_c, k2_c, k3_c, p1_c, p2_c)
print(grids_cam_undistort)

# grids_cam_undistort = grids_cam_ordered

"""Mid-point method"""
rays = np.zeros((3, 3))  # 두o의 계수 (t1, t2, t3)저장
real_world_pts = np.zeros((num_dots, 3))

for k in range(num_dots):
    #P_camera^dagger @ x
    CPdx = cam_mtx_pseudoinv @ np.append(grids_cam_undistort[k], 1)
    u1 = np.array([CPdx[0] - CPdx[3] * camera_C[0],
                   CPdx[1] - CPdx[3] * camera_C[1],
                   CPdx[2] - CPdx[3] * camera_C[2]])
    #P_projector^dagger @ x
    PPdx = proj_mtx_world_pseudoinv @ np.append(grids_proj_undistort[k], 1)
    u2 = np.array([PPdx[0] - PPdx[3] * projector_C_world[0],
                   PPdx[1] - PPdx[3] * projector_C_world[1],
                   PPdx[2] - PPdx[3] * projector_C_world[2]])
    u3 = np.cross(u1, u2)
    rays[:, 0] = u1
    rays[:, 1] = -u2
    rays[:, 2] = u3

    vec_t = np.linalg.inv(rays) @ (projector_C_world - camera_C)

    real_world_pts[k] = ((camera_C + vec_t[0] * u1 + projector_C_world + vec_t[1] * u2) / 2)

    # 꼭지점 잇는 보조선 만들기 위한 변수들
    if k == 0:
        u_proj_first = u2
        T_proj_first = vec_t[1]
        u_cam_first = u1
        T_cam_first = vec_t[0]
    elif k == num_dots_x - 1:
        u_proj_second = u2
        T_proj_second = vec_t[1]
        u_cam_second = u1
        T_cam_second = vec_t[0]
    elif k == num_dots_x * (num_dots_y - 1):
        u_proj_third = u2
        T_proj_third = vec_t[1]
        u_cam_third = u1
        T_cam_third = vec_t[0]
    elif k == num_dots - 1:
        u_proj_fourth = u2
        T_proj_fourth = vec_t[1]
        u_cam_fourth = u1
        T_cam_fourth = vec_t[0]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

X = real_world_pts[:, 0]
Y = real_world_pts[:, 1]
Z = real_world_pts[:, 2]

ax.scatter(X, Y, Z, s=1)
ax.scatter(projector_C_world[0], projector_C_world[1], projector_C_world[2],
           c='orange', s=10, label='proj')
ax.scatter(camera_C[0], camera_C[1], camera_C[2],
           c='green', s=10, label='cam')

"""보조선 그리기"""
# 가장자리 4개 점 까지 선 그리기(프로젝터)
ax.plot([projector_C_world[0], projector_C_world[0] + T_proj_first * u_proj_first[0]],
        [projector_C_world[1], projector_C_world[1] + T_proj_first * u_proj_first[1]],
        [projector_C_world[2], projector_C_world[2] + T_proj_first * u_proj_first[2]])
ax.plot([projector_C_world[0], projector_C_world[0] + T_proj_second * u_proj_second[0]],
        [projector_C_world[1], projector_C_world[1] + T_proj_second * u_proj_second[1]],
        [projector_C_world[2], projector_C_world[2] + T_proj_second * u_proj_second[2]])
ax.plot([projector_C_world[0], projector_C_world[0] + T_proj_third * u_proj_third[0]],
        [projector_C_world[1], projector_C_world[1] + T_proj_third * u_proj_third[1]],
        [projector_C_world[2], projector_C_world[2] + T_proj_third * u_proj_third[2]])
ax.plot([projector_C_world[0], projector_C_world[0] + T_proj_fourth * u_proj_fourth[0]],
        [projector_C_world[1], projector_C_world[1] + T_proj_fourth * u_proj_fourth[1]],
        [projector_C_world[2], projector_C_world[2] + T_proj_fourth * u_proj_fourth[2]])
# principal axis(프로젝터)
z_axis = np.array([0, 0, 1])

# ax.plot([projector_C_world[0], projector_C_world[0] + 400 * proj_mtx_world[2, 0]],
#         [projector_C_world[1], projector_C_world[1] + 400 * proj_mtx_world[2, 1]],
#         [projector_C_world[2], projector_C_world[2] + 400 * proj_mtx_world[2, 2]],
#         linestyle='--')

# 가장자리 4개 선 (카메라)
ax.plot([camera_C[0], camera_C[0] + T_cam_first * u_cam_first[0]],
        [camera_C[1], camera_C[1] + T_cam_first * u_cam_first[1]],
        [camera_C[2], camera_C[2] + T_cam_first * u_cam_first[2]])
ax.plot([camera_C[0], camera_C[0] + T_cam_second * u_cam_second[0]],
        [camera_C[1], camera_C[1] + T_cam_second * u_cam_second[1]],
        [camera_C[2], camera_C[2] + T_cam_second * u_cam_second[2]])
ax.plot([camera_C[0], camera_C[0] + T_cam_third * u_cam_third[0]],
        [camera_C[1], camera_C[1] + T_cam_third * u_cam_third[1]],
        [camera_C[2], camera_C[2] + T_cam_third * u_cam_third[2]])
ax.plot([camera_C[0], camera_C[0] + T_cam_fourth * u_cam_fourth[0]],
        [camera_C[1], camera_C[1] + T_cam_fourth * u_cam_fourth[1]],
        [camera_C[2], camera_C[2] + T_cam_fourth * u_cam_fourth[2]])
# principal axis (카메라)
# ax.plot([camera_C[0], camera_C[0] + 400 * cam_mtx[2, 0]],
#         [camera_C[1], camera_C[1] + 400 * cam_mtx[2, 1]],
#         [camera_C[2], camera_C[2] + 400 * cam_mtx[2, 2]],
#         linestyle='--')
ax.legend()
plt.suptitle('3D reconstruction of grids', fontsize=15)
plt.show()

fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')

for k in range(num_dots_y):
    ax3.scatter(X[k * num_dots_x:(k + 1) * num_dots_x], Y[k * num_dots_x:(k + 1) * num_dots_x],
                Z[k * num_dots_x:(k + 1) * num_dots_x])

ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.suptitle('Expanded scale of Z-axis')
plt.show()
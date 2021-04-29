from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from data.make_grids import *
from data import cam_params as cp
from data import proj_params as pp
from undistort import *
from find_cam_grids import find_cam_grids


def main():

    Origin = 10
    interval = 30
    width = 800
    height = 600

    # CONNECTIVITY = 4
    # DRAW_CIRCLE_RADIUS = 4
    THRESHOLD = 180

    img_camera = cv2.imread('./data/Picture.jpg')  # Image taken by camera

    _, grids_proj_coords, num_grids_w, num_grids_h = make_grids(interval=interval, width=width, height=height,
                                                                save_path="./data/grids.jpg", save=False)
    num_grids = num_grids_h * num_grids_w

    # Projector matrix world coordinate로 바꿔야 함

    camera_C = cp.C.reshape(3)
    proj_R_world = pp.R @ cp.R
    proj_C_world = cp.R.T @ pp.C + camera_C
    proj_P_world = pp.K @ proj_R_world @ np.hstack([np.identity(3), -proj_C_world.reshape(3, 1)])
    proj_P_pinv_world = proj_P_world.T @ np.linalg.inv(proj_P_world @ proj_P_world.T)

    """ projector undistortion"""
    grids_proj_undistort = np.zeros((num_grids, 2))
    for i in range(num_grids):
        grids_proj_undistort[i] = undistort(grids_proj_coords[i], pp.K, pp.k1, pp.k2, pp.k3, pp.p1, pp.p2)

    grids_cam_undistort = find_cam_grids(img_camera, num_w=num_grids_w, num_h=num_grids_h, interval=interval,
                                         THRESHOLD=THRESHOLD, show_grid_img=False)

    """Mid-point method"""
    rays = np.zeros((3, 3))  # 두o의 계수 (t1, t2, t3)저장
    real_world_pts = np.zeros((num_grids, 3))

    for k in range(num_grids):
        # P_camera^dagger @ x
        CPdx = cp.P_pinv @ np.append(grids_cam_undistort[k], 1)
        u1 = np.array([CPdx[0] - CPdx[3] * camera_C[0],
                       CPdx[1] - CPdx[3] * camera_C[1],
                       CPdx[2] - CPdx[3] * camera_C[2]])
        # P_projector^dagger @ x
        PPdx = proj_P_pinv_world @ np.append(grids_proj_undistort[k], 1)
        u2 = np.array([PPdx[0] - PPdx[3] * proj_C_world[0],
                       PPdx[1] - PPdx[3] * proj_C_world[1],
                       PPdx[2] - PPdx[3] * proj_C_world[2]])

        u3 = np.cross(u1, u2)
        rays[:, 0] = u1
        rays[:, 1] = -u2
        rays[:, 2] = u3

        vec_t = np.linalg.inv(rays) @ (proj_C_world - camera_C)

        real_world_pts[k] = ((camera_C + vec_t[0] * u1 + proj_C_world + vec_t[1] * u2) / 2)

        # 꼭지점 잇는 보조선 만들기 위한 변수들
        if k == 0:
            u_proj_first = u2
            T_proj_first = vec_t[1]
            u_cam_first = u1
            T_cam_first = vec_t[0]
        elif k == num_grids_w - 1:
            u_proj_second = u2
            T_proj_second = vec_t[1]
            u_cam_second = u1
            T_cam_second = vec_t[0]
        elif k == num_grids_w * (num_grids_h - 1):
            u_proj_third = u2
            T_proj_third = vec_t[1]
            u_cam_third = u1
            T_cam_third = vec_t[0]
        elif k == num_grids - 1:
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
    ax.scatter(proj_C_world[0], proj_C_world[1], proj_C_world[2],
               c='orange', s=10, label='proj')
    ax.scatter(camera_C[0], camera_C[1], camera_C[2],
               c='green', s=10, label='cam')

    """보조선 그리기"""
    # 가장자리 4개 점 까지 선 그리기(프로젝터)
    ax.plot([proj_C_world[0], proj_C_world[0] + T_proj_first * u_proj_first[0]],
            [proj_C_world[1], proj_C_world[1] + T_proj_first * u_proj_first[1]],
            [proj_C_world[2], proj_C_world[2] + T_proj_first * u_proj_first[2]])
    ax.plot([proj_C_world[0], proj_C_world[0] + T_proj_second * u_proj_second[0]],
            [proj_C_world[1], proj_C_world[1] + T_proj_second * u_proj_second[1]],
            [proj_C_world[2], proj_C_world[2] + T_proj_second * u_proj_second[2]])
    ax.plot([proj_C_world[0], proj_C_world[0] + T_proj_third * u_proj_third[0]],
            [proj_C_world[1], proj_C_world[1] + T_proj_third * u_proj_third[1]],
            [proj_C_world[2], proj_C_world[2] + T_proj_third * u_proj_third[2]])
    ax.plot([proj_C_world[0], proj_C_world[0] + T_proj_fourth * u_proj_fourth[0]],
            [proj_C_world[1], proj_C_world[1] + T_proj_fourth * u_proj_fourth[1]],
            [proj_C_world[2], proj_C_world[2] + T_proj_fourth * u_proj_fourth[2]])
    # principal axis(프로젝터)
    z_axis = np.array([0, 0, 1])

    # ax.plot([proj_C_world[0], proj_C_world[0] + 400 * proj_mtx_world[2, 0]],
    #         [proj_C_world[1], proj_C_world[1] + 400 * proj_mtx_world[2, 1]],
    #         [proj_C_world[2], proj_C_world[2] + 400 * proj_mtx_world[2, 2]],
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

    for k in range(num_grids_h):
        ax3.scatter(X[k * num_grids_w:(k + 1) * num_grids_w], Y[k * num_grids_w:(k + 1) * num_grids_w],
                    Z[k * num_grids_w:(k + 1) * num_grids_w])

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.suptitle('Expanded scale of Z-axis')
    plt.show()


if __name__ == "__main__":
    main()
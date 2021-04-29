import numpy as np
import cv2
from undistort import *
from data import cam_params as cp


def find_cam_grids(img, num_w, num_h, interval, THRESHOLD, CONNECTIVITY=4, show_grid_img = False, Origin = 10, DRAW_CIRCLE_RADIUS = 4):
    """

    :param img: camera image
    :param num_w: x_number of grids
    :param num_h: y_number of grids
    :param interval: interval of grids
    :param THRESHOLD: Threshold of binary image
    :param CONNECTIVITY: Connectivity of cv2.connected components
    :param show_grid_img: Whether to show how to find and order grids
    :param Origin: Parameter where to send upper left grid of camera image
    :param DRAW_CIRCLE_RADIUS: Radius of red dots drawn
    :return: Undistorted ordered grids of camera image
    """

    if len(img.shape) != 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    _, img_bin = cv2.threshold(img_gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    ret, labels, stats, grids_cam = cv2.connectedComponentsWithStats(img_bin, CONNECTIVITY, cv2.CV_32S)
    grids_cam = np.delete(grids_cam, 0, axis=0)

    print("# of grids in camera image : {}".format(len(grids_cam)))

    # 꼭지점 4개 구하기
    grids_sum = grids_cam.sum(axis=1)  # 각 grid 좌표 합
    grids_diff = grids_cam[:, 1] - grids_cam[:, 0]  # 각 gird 좌표 차

    UL_pt = grids_cam[grids_sum.argmin()]
    UR_pt = grids_cam[grids_diff.argmin()]
    LR_pt = grids_cam[grids_sum.argmax()]
    LL_pt = grids_cam[grids_diff.argmax()]

    four_pts = np.array([UL_pt, UR_pt, LR_pt, LL_pt])

    four_pts = four_pts.astype(np.float32)
    rect_pt = np.array([[Origin, Origin],
                        [Origin + (num_w - 1) * interval, Origin],
                        [Origin + (num_w - 1) * interval, Origin + (num_h - 1) * interval],
                        [Origin, Origin + (num_h - 1) * interval]], dtype=np.float32)

    Hom = cv2.getPerspectiveTransform(four_pts, rect_pt)

    img_cam_rect = cv2.warpPerspective(img, Hom, (int(800), int(600)))
    img_cam_rect_gray = cv2.cvtColor(img_cam_rect, cv2.COLOR_BGR2GRAY)
    _, img_cam_rect_bin = cv2.threshold(img_cam_rect_gray, THRESHOLD, 255, cv2.THRESH_BINARY)

    ret, labels, stats, grids_cam_rect = cv2.connectedComponentsWithStats(img_cam_rect_bin, CONNECTIVITY, cv2.CV_32S)
    grids_cam_rect = np.delete(grids_cam_rect, 0, axis=0)
    print("# of grids in camera image(rect) : ", len(grids_cam_rect))

    """grid order 맞추기"""
    Hom_inv = np.linalg.inv(Hom)
    num_grids = num_w * num_h

    grids_cam_rect_ordered = np.zeros((num_grids, 2))
    grids_cam_ordered = np.zeros((num_grids, 2))
    origin_pt = np.array([Origin, Origin])

    for y in range(num_h):
        for x in range(num_w):
            close_pt = origin_pt + [x * interval, y * interval]
            close_pt_centered = grids_cam_rect - close_pt
            index_of_ordered_pt = np.linalg.norm(close_pt_centered, axis=1).argmin()  # 거리가 가장 작은 점 선택
            grids_cam_rect_ordered[y * num_w + x] = grids_cam_rect[index_of_ordered_pt]

    for k in range(num_grids):
        temp = Hom_inv @ np.append(grids_cam_rect_ordered[k], 1)
        temp = temp / temp[2]
        grids_cam_ordered[k] = np.delete(temp, 2)

    """카메라 grids undistort"""
    grids_cam_undistort = np.zeros((num_grids, 2))

    for k in range(num_grids):
        grids_cam_undistort[k] = undistort(grids_cam_ordered[k], cp.K, cp.k1, cp.k2, cp.k3, cp.p1, cp.p2)\

    if show_grid_img:

        img_connect = img.copy()
        for grid in grids_cam:
            cv2.circle(img_connect, (int(grid[0]), int(grid[1])), DRAW_CIRCLE_RADIUS, (0, 0, 255), -1)
        cv2.imshow("Detected grids", cv2.resize(img_connect, (1200, 900)))
        cv2.waitKey()
        cv2.destroyAllWindows()

        img_four_circles = img.copy()
        for enum, pt in enumerate(four_pts):
            cv2.circle(img_four_circles, (int(pt[0]), int(pt[1])), 20, (0, 0, (enum + 1) * 60), -1)
        cv2.imshow("4 vertices", cv2.resize(img_four_circles, (1200, 900)))
        cv2.waitKey()
        cv2.destroyAllWindows()

        img_cam_rect_connect = img_cam_rect.copy()
        for grid in grids_cam_rect:
            cv2.circle(img_cam_rect_connect, (int(grid[0]), int(grid[1])), DRAW_CIRCLE_RADIUS, (0, 0, 255), -1)
            cv2.imshow("Matching with Original image (exit: type \"o\")", cv2.resize(img_cam_rect_connect, (1200, 900)))
            ord_k = cv2.waitKey(0)
            if ord_k == ord('o'):
                break
        cv2.destroyAllWindows()
        
        # ordered grids 보여주기
        img_ordered_grids = img.copy()
        for grid in grids_cam_ordered:
            cv2.circle(img_ordered_grids, (int(grid[0]), int(grid[1])), DRAW_CIRCLE_RADIUS, (0, 0, 255), -1)
            cv2.imshow('Matched grids (exit : type \"o\")', cv2.resize(img_ordered_grids, (1200, 900)))
            ord_k = cv2.waitKey(0)
            if ord_k == ord('o'):
                break
        cv2.destroyAllWindows()

    return grids_cam_undistort
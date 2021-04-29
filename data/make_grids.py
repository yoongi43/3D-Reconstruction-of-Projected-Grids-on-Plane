import numpy as np
import cv2


def make_grids(interval, width, height, save_path, save=True, RGB_list = [0, 255, 0]):
    """
    Create image with grids
    :param interval: interval of grids
    :param width: width of image
    :param height: height of image
    :param RGB_list: RGB of grids
    :return: grid image
    """

    grids = np.zeros((height, width, 3), dtype=np.uint8)
    R, G, B = RGB_list
    w_range = np.arange(0, width, interval)
    h_range = np.arange(0, height, interval)
    num_w = len(w_range)
    num_h = len(h_range)

    for h in h_range:
        for w in w_range:
            cv2.circle(grids, (w, h), 1, (int(B), int(G), int(R)), -1)

    ww, hh = np.meshgrid(w_range, h_range)
    grids_coords = np.vstack([ww.flatten(), hh.flatten()]).T

    if save:
        cv2.imwrite(save_path, grids)

    return grids, grids_coords, len(w_range), len(h_range)


if __name__ == "__main__":
    grids = make_grids(30, 800, 600, save_path="./grids.jpg", save=False)

    # cv2.imshow("grids", grids)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
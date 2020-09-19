import os
from functools import reduce

import numpy as np
import pandas as pd
import cv2 as cv
from scipy import spatial
import matplotlib.pyplot as plt
from skimage.feature import haar_like_feature

from mondrian.visualization import table_as_image, find_table_elements, lbp_describe, parse_cell
from skimage.transform import integral_image
from sklearn.preprocessing import normalize

def find_layout(file_path, cluster_edges=None, delimiter=",", keyp_set=None, partitioning=True, print=False):
    img = table_as_image(file_path, delimiter=delimiter, color=True, cell_length=False)  # TODO: Dynamic delimiter
    # elements_found, _, _ = find_table_elements(img, partitioning=partitioning)

    # boundaries = [[*x["top_lx"], *x["bot_rx"]] for x in cluster_edges]
    # boundaries = [[*map(int, b)] for b in boundaries]
    # boundaries = sorted(boundaries)
    # feature_vect = lbp_describe(img)

    # feature_types = ['type-2-x', 'type-2-y']
    # img2 = cv.resize(img, dsize=(16,16))
    # img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # feature_vect = extract_feature_image(img2, feature_types)

    hist = cv.calcHist([img], [0, 1, 2], None, [64, 64, 64],
                       [0, 256, 0, 256, 0, 256])
    feature_vect = cv.normalize(hist, hist).flatten()
    if print:
        fig = plt.figure()
        plt.title(os.path.split(file_path)[1][:20])
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(feature_vect)
        plt.show()
    return feature_vect


def rectangle_similarity(rect_a, rect_b):
    top_a, bot_a = rect_a["top_lx"], rect_a["bot_rx"]
    top_b, bot_b = rect_b["top_lx"], rect_b["bot_rx"]
    if top_a == top_b and bot_a == bot_b:
        return 1
    else:
        return 0


def layout_similarity(layout_a, layout_b):
    dist = spatial.distance.euclidean(layout_a, layout_b)
    return 1 - dist


def compare(a, b):
    print(a, b)
    if a == b:
        return a
    else:
        pars_a = parse_cell(a)
        pars_b = parse_cell(b)
        if (pars_a != pars_b).all:
            return str([0, 125, 125])
        else:
            return str(pars_a)


# http://stackoverflow.com/q/3844948/
def shrink(cells):
    # cell_array = [str(parse_cell(c,color=True)) for c in cells]
    cell_array = cells
    sim = ((len(set(cell_array))-1) / len(cells))  # 1 is complete difference => should be white
    return 1 - sim
    # if set(cells) == {"nan"}: #TODO Change representation for missing values
    #     return False
    # lst = cells.tolist()
    # return not lst or lst.count(lst[0]) == len(lst)


def find_structural_mask(files):
    file_matrices = []
    for f in files:
        file_matrices.append(pd.read_csv(f, quotechar='"', skipinitialspace=True, header=None).to_numpy(str))

    target_rows = max([np.shape(x)[0] for x in file_matrices])
    target_cols = max([np.shape(x)[1] for x in file_matrices])

    for idx, m in enumerate(file_matrices):
        desired_rows = np.abs(np.shape(m)[0] - target_rows)
        desired_cols = np.abs(np.shape(m)[1] - target_cols)

        file_matrices[idx] = np.pad(m, [(0, desired_rows), (desired_cols, 0)], constant_values="PAD")

    file_matrices = np.asarray(file_matrices)
    # TODO change comparison with something not a==b
    # mask = np.apply_along_axis(lambda x: reduce(lambda a, b: compare(a, b), x), 0, file_matrices)  # python magic syntax
    mask = np.apply_along_axis(lambda x: shrink(x), 0, file_matrices)

    return mask

#DEPRECATED
# def extract_feature_image(img, feature_type, feature_coord=None):
#     """Extract the haar feature for the current image"""
#     ii = integral_image(img)
#     return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
#                              feature_type=feature_type,
#                              feature_coord=feature_coord)
#
#
# def get_keyp_set(file_path):
#     img = table_as_image(file_path, delimiter=",", color=True)
#     minHessian = 400
#     detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
#     keypoints = detector.detect(img)
#     # -- Draw keypoints
#     img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     cv.drawKeypoints(img, keypoints, img_keypoints)
#     # plt.imshow('SURF Keypoints', img_keypoints)
#     # plt.show()
#     print(keypoints)
#     return keypoints

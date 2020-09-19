import numpy as np


def hist_intersection(h1,h2):
    s = 0
    for idx in range(len(h1)):
        s += min(h1[idx], h2[idx])

    return s


def custom_distance(X, Y, file_area, alpha=1, beta=0, gamma=0.1):
    # Workaround
    X = X[0]
    Y = Y[0]
    eps = np.finfo(np.float32).eps

    # how to make sure ratio is always smaller/bigger
    area_ratio = X["area"] / Y["area"]
    if area_ratio > 1:
        area_ratio = 1 / area_ratio

    # If geo_dist=0 => they are equal
    geo_dist = np.linalg.norm(np.array(X["top_lx"]) - np.array(Y["top_lx"]))

    # If area_dist = 0 => they should be grouped together
    area_dist = np.abs(np.log1p(eps - area_ratio))

    # If width is the same they should be grouped together
    h_alignment = np.abs(X["bot_rx"][0] - Y["bot_rx"][0]) + np.abs(X["top_lx"][0] - Y["top_lx"][0])
    v_alignment = np.abs(X["bot_rx"][1] - Y["bot_rx"][1]) + np.abs(X["top_lx"][1] - Y["top_lx"][1])

    return (alpha * geo_dist) + (beta * area_dist) + (gamma * h_alignment) + (gamma * v_alignment)


def get_center_width_height(obj):
    top_lx = obj["top_lx"]
    bot_rx = obj["bot_rx"]

    width = bot_rx[0] - top_lx[0] + 1
    height = bot_rx[1] - top_lx[1] + 1
    center_x = top_lx[0] + (width / 2)
    center_y = top_lx[1] + (height / 2)

    return center_x, center_y, width, height


def dist_closest(obj_a, obj_b):
    a_top_x, a_top_y = obj_a[0], obj_a[1]  # obj_a["top_lx"]
    a_bot_x, a_bot_y = obj_a[2], obj_a[3]  # obj_b["bot_rx"]

    b_top_x, b_top_y = obj_b[0], obj_b[1]  # obj_b["top_lx"]
    b_bot_x, b_bot_y = obj_b[2], obj_b[3]  # obj_b["bot_rx"]

    left = b_bot_x < a_top_x # b is left
    right = a_bot_x < b_top_x #b is right
    top = b_bot_y < a_top_y #b is top
    bottom = a_bot_y < b_top_y  # b is bot

    if bottom and left:
        return np.linalg.norm(np.array([a_top_x, a_bot_y]) - np.array([b_bot_x+1, b_top_y-1]))
    elif bottom and right:
        return np.linalg.norm(np.array([a_bot_x, a_bot_y]) - np.array([b_top_x-1, b_top_y-1]))
    elif top and left:
        return np.linalg.norm(np.array([a_top_x-1, a_top_y-1]) - np.array([b_bot_x, b_bot_y]))
    elif top and right:
        return np.linalg.norm(np.array([a_bot_x+1, a_top_y-1]) - np.array([b_top_x, b_bot_y]))

    elif left:
        return a_top_x - b_bot_x -1
    elif right:
        return b_top_x - a_bot_x -1
    elif top:
        return a_top_y - b_bot_y -1
    elif bottom:
        return b_top_y - a_bot_y -1
    else:  # rectangles intersect
        return 0.


def is_connected(obj_a, obj_b):
    if np.array_equal(obj_a, obj_b):
        return 0

    h_gap = np.abs(obj_a[4] - obj_b[4]) - obj_a[6] * 0.5 - obj_b[6] * 0.5;
    v_gap = np.abs(obj_a[5] - obj_b[5]) - obj_a[7] * 0.5 - obj_b[7] * 0.5

    if h_gap < 0 and v_gap < 0:
        return 0#-1
    elif (h_gap == 0 and v_gap <= 0) or (v_gap == 0 and h_gap < 0):
        return 0
    else:
        return 1


def rectangle_as_array(obj):
    """To be used in order to paralallelize the distance measure"""
    x_0 = obj["top_lx"][0]
    x_1 = obj["top_lx"][1]
    x_2 = obj["bot_rx"][0]
    x_3 = obj["bot_rx"][1]
    x_4, x_5, x_6, x_7 = get_center_width_height(obj)
    x_8 = obj["area"]
    try:
        x_9 = obj["lbp"]
        return np.asarray([x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9])

    except:
        return np.asarray([x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8])


def parallel_distance(X, Y, file_width, file_height, alpha, beta, gamma, lbp = False, print_metric=False):
    """
    Custom distance metric that takes into account the euclidean distance as well as the geometric shape of elements.
    The closer their area are, the more distant two elements are.

    Parameters:
    -----------
    X,Y: bounding boxes expressed as a numpy array (to be compatible with numpy pairwise distance)

    Returns:
    ---------
    distance: float
    """
    eps = np.finfo(np.float32).eps

    # how to make sure ratio is always smaller/bigger
    area_ratio = X[8] / Y[8]
    if area_ratio > 1:
        area_ratio = 1 / area_ratio

    file_area = file_height * file_width

    # If geo_dist=0 => they are equal
    geo_dist = np.linalg.norm(np.array([X[0], X[1]]) - np.array([Y[0], Y[1]])) #/ file_area
    geo_dist += is_connected(X,Y)

    # geo_dist must be closest!
    geo_dist = dist_closest(X, Y) * is_connected(X, Y) #/ file_area
    if is_connected(X, Y) == -1:
        print("X:", X)
        print("Y:", Y)
        print("intersecting elements, geo_dist", geo_dist)

    # If area_similarity = 0 => they should be grouped together
    # area_similarity = np.abs(np.log1p(eps - area_ratio))
    area_similarity = (1-area_ratio)

    # If width is the same they should be grouped together
    h_alignment = (np.abs(X[2] - Y[2]) + np.abs(X[0] - Y[0])) / max(X[6], Y[6]) #the greater width
    v_alignment = (np.abs(X[3] - Y[3]) + np.abs(X[1] - Y[1])) / max(X[7], Y[7]) #the greater height
    alignment_distance = min(h_alignment, v_alignment)
    # combined = (alpha * geo_dist) + (beta * area_similarity) + (gamma * h_alignment) + (gamma * v_alignment)
    combined = (alpha * geo_dist) + (beta * area_similarity) + (gamma * alignment_distance)

    if lbp:
        lbp = hist_intersection(X[9], Y[9])
        combined += 1-lbp

    if print_metric:
        print("Elems:", [X[0], X[1]], [X[2], X[3]], "<->", [Y[0], Y[1]], [Y[2], Y[3]])
        print("\tgeo_dist:", alpha * geo_dist)
        print("\tarea_similarity", beta * area_similarity)
        print("\th_alignment", h_alignment, "v_alignment", v_alignment)
        print("\talignment distance", gamma * alignment_distance)
        print("\tlbp similarity",lbp)
        print("\t\tcombined", combined)
    return combined

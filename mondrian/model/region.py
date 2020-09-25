import itertools

import numpy as np
import cv2 as cv

DIRECTION_NONE = 0
DIRECTION_H = 1
DIRECTION_V = 2
DIRECTION_O = 3

def parallel_region_sim(r1,r2):
    return np.abs(cv.compareHist(r1, r2, method = cv.HISTCMP_CORREL))

def region_similarity(region_a, region_b):
    w_a, h_a, hist_a = region_a.width, region_a.height, region_a.color_hist
    w_b, h_b, hist_b = region_b.width, region_b.height, region_b.color_hist
    hist_correl = np.abs(cv.compareHist(hist_a, hist_b, method=cv.HISTCMP_CORREL))
    return hist_correl

def region_adjacency(region_a, region_b):
    """
    Returns 0 if the regions are either the same or not adjacent.
    Returns N if the regions are on top of each other, for N pixels
    Returns -N if the regions are left/right of each other for N pixels
    :param region_a:
    :param region_b:
    :return:
    """
    if region_a.top_lx == region_b.top_lx and region_a.bot_rx == region_b.bot_rx:
        return [None, 0]

    a_x0, a_y0, a_x1, a_y1 = *region_a.top_lx, *region_a.bot_rx
    b_x0, b_y0, b_x1, b_y1 = *region_b.top_lx, *region_b.bot_rx

    a_top = list(zip(range(a_x0, a_x1 + 1), [a_y0] * len(range(a_x0, a_x1 + 1))))
    a_bot = list(zip(range(a_x0, a_x1 + 1), [a_y1] * len(range(a_x0, a_x1 + 1))))
    a_left = list(zip([a_x0] * len(range(a_y0, a_y1 + 1)), range(a_y0, a_y1 + 1)))
    a_right = list(zip([a_x1] * len(range(a_y0, a_y1 + 1)), range(a_y0, a_y1 + 1)))

    b_top = list(zip(range(b_x0, b_x1 + 1), [b_y0] * len(range(b_x0, b_x1 + 1))))
    b_bot = list(zip(range(b_x0, b_x1 + 1), [b_y1] * len(range(b_x0, b_x1 + 1))))
    b_left = list(zip([b_x0] * len(range(b_y0, b_y1 + 1)), range(b_y0, b_y1 + 1)))
    b_right = list(zip([b_x1] * len(range(b_y0, b_y1 + 1)), range(b_y0, b_y1 + 1)))

    cells = [cell for cell in a_top if (cell[0], cell[1] - 1) in b_bot]
    if len(cells) > 0:
        return [DIRECTION_H, len(cells)]
    cells = [cell for cell in b_top if (cell[0], cell[1] - 1) in a_bot]
    if len(cells) > 0:
        return [DIRECTION_H, len(cells)]

    cells = [cell for cell in a_left if (cell[0] - 1, cell[1]) in b_right]
    if len(cells) > 0:
        return [DIRECTION_V, len(cells)]
    cells = [cell for cell in b_left if (cell[0] - 1, cell[1]) in a_right]
    if len(cells) > 0:
        return [DIRECTION_V, len(cells)]

    # check overlap https://stackoverflow.com/a/27162334
    dx = min(a_x1, b_x1) - max(a_x0, b_x0) + 1
    dy = min(a_y1, b_y1) - max(a_y0, b_y0) + 1
    if (dx >= 0) and (dy >= 0):
        return [DIRECTION_O, dx * dy]

    return [None, 0]


def region_distance(region_a, region_b):
    """
        returns ["direction", n_shared_pixels´,distance]
        direction can be
            None if they are the same
            DIRECTION_O if they overlap
            DIRECTION_H if they share x axis
            DIRECTION_V if they share Y axis
            DIRECTION_NONE if they don't share any axis
    """

    if region_a.top_lx == region_b.top_lx and region_a.bot_rx == region_b.bot_rx:
        return {"direction": None, "weight": 0, "distance": 0}

    a_x0, a_y0, a_x1, a_y1 = *region_a.top_lx, *region_a.bot_rx
    b_x0, b_y0, b_x1, b_y1 = *region_b.top_lx, *region_b.bot_rx

    dx = min(a_x1, b_x1) - max(a_x0, b_x0)
    dy = min(a_y1, b_y1) - max(a_y0, b_y0)

    # Overlap
    if (dx >= 0) and (dy >= 0):
        return {"direction":DIRECTION_O, "weight":(dx+1) * (dy+1), "distance":0}
    # H overlap
    elif (dx >= 0) and (dy < 0):
        return {"direction":DIRECTION_H, "weight":dx+1, "distance":np.abs(dy+1)}
    # V overlap
    elif (dx < 0) and (dy >= 0):
        return {"direction": DIRECTION_V, "weight": dy+1, "distance": np.abs(dx+1)}
    # non overlap
    else:
        return {"direction":DIRECTION_NONE, "weight":0, "distance":np.sqrt((dx+1) ** 2 + (dy+1) ** 2)}


def intersecting(line_a, line_b):
    x0_a, y0_a, x1_a, y1_a, _ = line_a
    x0_b, y0_b, x1_b, y1_b, _ = line_b

    a_pts = set(itertools.product(range(x0_a, x1_a + 1), range(y0_a, y1_a + 1)))
    b_pts = set(itertools.product(range(x0_b, x1_b + 1), range(y0_b, y1_b + 1)))

    return int(len(a_pts.intersection(b_pts)) > 0)


def dominating(region_a, region_b):
    """ Returns the dominated region (the subset) if there's intersection between the two
    """
    x0_a, y0_a, x1_a, y1_a = region_a
    x0_b, y0_b, x1_b, y1_b = region_b

    x = max(x0_a, x0_b)
    y = max(y0_a, y0_b)
    w = min(x0_a + x1_a, x0_b + x1_b) - x
    h = min(y0_a + y1_a, y0_b + y1_b) - y

    if (x, y, w, h) == (x0_a, y0_a, x1_a, y1_a):
        return region_b
    elif (x, y, w, h) == (x0_b, y0_b, x1_b, y1_b):
        return region_a

    return None


def rectangles_from_lines(lines):
    top_x = min([l[1] for l in lines if l[0] == "v"])
    top_y = min([l[1] for l in lines if l[0] == "h"])
    bot_x = max([l[1] for l in lines if l[0] == "v"]) - 1
    bot_y = max([l[1] for l in lines if l[0] == "h"]) - 1

    return (top_x, top_y, bot_x, bot_y)


class Region:

    def __init__(self, filename, points_tuple=None, top_lx=None, bot_rx=None, img=None, color_hist=None, type=None,
                 **kwargs):
        self.filename = filename
        if top_lx is not None:
            self.top_lx = [int(x) for x in top_lx]
            self.bot_rx = [int(x) for x in bot_rx]
        else:
            self.top_lx = [int(points_tuple[0]), int(points_tuple[1])]
            self.bot_rx = [int(points_tuple[2]), int(points_tuple[3])]

        x0, y0, x1, y1 = *self.top_lx, *self.bot_rx
        self.height, self.width = y1 - y0 + 1, x1 - x0 + 1

        if color_hist is None:
            content = img[y0:y1 + 1, x0:x1 + 1, :]
            b_hist = cv.calcHist([content], [0], None, [64], (0,256))
            b_hist = cv.normalize(b_hist, b_hist)
            g_hist = cv.calcHist([content], [1], None, [64], (0,256))
            g_hist = cv.normalize(g_hist, g_hist)
            r_hist = cv.calcHist([content], [2], None, [64], (0,256))
            r_hist = cv.normalize(r_hist, r_hist)
            self.color_hist = np.vstack([b_hist,g_hist,r_hist]).flatten()
        else:
            self.color_hist = color_hist

        self.type = type
        self.hashname = filename + "_" + str(top_lx) + "-" + str(bot_rx)

    @classmethod
    def fromdict(cls, saved_dict, img):
        return cls(**saved_dict, img=img)

    def asdict(self):
        return {k: v for k, v in vars(self).items() if k != "color_hist"}

    def __str__(self):
        out = ""
        out += str(self.top_lx)
        out += "\n"
        out += str(self.bot_rx)
        return out

    def asarray(self):
        """To be used in order to paralallelize the distance measure"""
        x_0 = self.top_lx[0]
        y_0 = self.top_lx[1]
        x_1 = self.bot_rx[0]
        y_1 = self.bot_rx[1]
        width = self.bot_rx[0] - self.top_lx[0] + 1
        height = self.bot_rx[1] - self.top_lx[1] + 1
        x_c = self.top_lx[0] + (width / 2)
        y_c = self.top_lx[1] + (height / 2)

        return np.asarray([x_0, y_0, x_1, y_1, x_c, y_c, width, height, width * height])

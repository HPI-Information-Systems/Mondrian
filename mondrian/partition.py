# The functions in this file can be used to partition connected components in a more efficient way than the one used in model.mondrian.find_regions()
# Conceptually, the results are equivalent to those calculated with mondrian.find_regions()

import numpy as np
from functools import reduce


def find_external_lines(contour):
    if len(contour) == 1:  # case of single pixels
        pixel = contour[0, 0]
        h_line = [pixel[1], pixel[0], pixel[0]]
        v_line = [pixel[0], pixel[1], pixel[1]]
        external_lines = [[h_line], [h_line], [v_line], [v_line]]
        return external_lines, []

    top = set()
    bot = set()
    left = set()
    right = set()

    concave_vertices = []
    line_edges = []

    # The first for cycle is to get outer borders and concave vertices
    i = 0
    orientation = []

    while i < len(contour):
        j = (i + 1) % len(contour)

        [x0, y0] = contour[i, 0]
        [x1, y1] = contour[j, 0]

        y_top, y_bot = sorted([y0, y1])
        x_left, x_right = sorted([x0, x1])

        # deal with diagonal lines
        if np.abs(x1 - x0) == np.abs(y1 - y0) and np.abs(x1 - x0) > 1:
            x_add = np.sign(x1 - x0)
            y_add = np.sign(y1 - y0)

            to_add = np.array([[(x0 + x_add), (y0 + y_add)]])
            contour = np.insert(contour, i + 1, to_add, axis=0)
            continue  # don't increment i

        if x0 == x1:
            if y0 > y1:
                right.add((x0, y_top, y_bot))
                orientation.append("right")
                line_edges.append([[x0, y0], [x1, y1], "right", i])
            else:
                left.add((x0, y_top, y_bot))
                orientation.append("left")
                line_edges.append([[x0, y0], [x1, y1], "left", i])

        elif y0 == y1:
            if x0 > x1:
                top.add((y0, x_left, x_right))
                orientation.append("top")
                line_edges.append([[x0, y0], [x1, y1], "top", i])

            else:
                bot.add((y0, x_left, x_right))
                orientation.append("bot")
                line_edges.append([[x0, y0], [x1, y1], "bot", i])

        if (x0 == x1 - 1) and (y0 == y1 - 1):  # --
            orientation.append(90)
            concave_vertices.append([[x0, y0], [x1, y1], 90, i])

        elif (x0 == x1 + 1) and (y0 == y1 - 1):  # +-
            orientation.append(180)
            concave_vertices.append([[x0, y0], [x1, y1], 180, i])

        elif (x0 == x1 + 1) and (y0 == y1 + 1):  # ++
            orientation.append(270)
            concave_vertices.append([[x0, y0], [x1, y1], 270, i])

        elif (x0 == x1 - 1) and (y0 == y1 + 1):  # -+
            orientation.append(360)
            concave_vertices.append([[x0, y0], [x1, y1], 360, i])

        i += 1

    for i in range(len(contour)):
        j = (i + 1) % len(contour)
        [x0, y0] = contour[i, 0]
        [x1, y1] = contour[j, 0]

    # pair is a pair of sequent points
    for pair in line_edges:

        [x0, y0], [x1, y1] = pair[0], pair[1]

        j = (pair[3] - 1)
        if j < 0:
            j = len(orientation) - 1

        k = (pair[3] + 1) % len(orientation)

        prev_orientation = orientation[j]
        next_orientation = orientation[k]

        if pair[2] == "bot":
            if next_orientation == "top":
                right.add((x1, y1, y1))
                if prev_orientation == "top":
                    left.add((x0, y0, y0))

            continue

        elif pair[2] == "top":
            if next_orientation == "bot":
                left.add((x1, y1, y1))
                if prev_orientation == "bot":
                    right.add((x0, y0, y0))
            continue

        elif pair[2] == "left":
            if next_orientation == "right":
                bot.add((y1, x1, x1))
                if prev_orientation == "right":
                    top.add((y0, x0, x0))

            continue

        elif pair[2] == "right":
            if next_orientation == "left":
                top.add((y1, x1, x1))
                if prev_orientation == "left":
                    bot.add((y0, x0, x0))
            continue

    for pair in concave_vertices:
        # c[0], c[1] coordinates of two points, c[2] orientation, c[3] original index of pair
        [x0, y0], [x1, y1] = pair[0], pair[1]

        j = (pair[3] - 1)
        if j < 0:
            j = len(orientation) - 1
        k = (pair[3] + 1) % len(orientation)

        previous_orientation = orientation[j]
        next_orientation = orientation[k]

        if pair[2] == 90:
            if previous_orientation in {"left", 270}:
                bot.add((y0, x0, x0))
            elif previous_orientation in {"top", 90}:
                bot.add((y0, x0, x0))
                left.add((x0, y0, y0))
            elif previous_orientation == 180:
                top.add((y0, x0, x0))
                left.add((x0, y0, y0))
                bot.add((y0, x0, x0))

            if next_orientation in {"bot", 180}:
                left.add((x1, y1, y1))
            elif next_orientation in {"right", 90}:
                left.add((x1, y1, y1))
                bot.add((y1, x1, x1))
            elif next_orientation == 360:
                left.add((x1, y1, y1))
                bot.add((y1, x1, x1))
                right.add((x1, y1, y1))

            if 270 in (previous_orientation, next_orientation):
                if previous_orientation == 270:
                    top.add((y0, x0, x0))
                    left.add((x0, y0, y0))
                    bot.add((y0, x0, x0))

                right.add((x0, y0, y0))

            continue

        elif pair[2] == 180:
            if previous_orientation in {"top", 90}:
                left.add((x0, y0, y0))
            elif previous_orientation in {"right", 180}:
                left.add((x0, y0, y0))
                top.add((y0, x0, x0))
            elif previous_orientation == 270:
                left.add((x0, y0, y0))
                top.add((y0, x0, x0))
                right.add((x0, y0, y0))

            if next_orientation in {"left", 270}:
                top.add((y1, x1, x1))
            elif next_orientation in {"bot", 180}:
                top.add((y1, x1, x1))
                left.add((x1, y1, y1))
            elif next_orientation == 90:
                top.add((y1, x1, x1))
                left.add((x1, y1, y1))
                bot.add((y1, x1, x1))

            if 360 in (previous_orientation, next_orientation):
                if previous_orientation == 360:
                    top.add((y0, x0, x0))
                    left.add((x0, y0, y0))
                    bot.add((y0, x0, x0))
                    right.add((x0, y0, y0))
            continue

        elif pair[2] == 270:
            if previous_orientation in {"right", 180}:
                top.add((y0, x0, x0))
            elif previous_orientation in {"bot", 270}:
                top.add((y0, x0, x0))
                right.add((x0, y0, y0))
            elif previous_orientation == 360:
                top.add((y0, x0, x0))
                right.add((x0, y0, y0))
                bot.add((y0, x0, x0))

            if next_orientation in {"top", 360}:
                right.add((x1, y1, y1))
            elif next_orientation in {"left", 270}:
                right.add((x1, y1, y1))
                top.add((y1, x1, x1))
            elif next_orientation == 180:
                right.add((x1, y1, y1))
                top.add((y1, x1, x1))
                left.add((x1, y1, y1))

            if 90 in (previous_orientation, next_orientation):
                if previous_orientation == 90:
                    bot.add((y0, x0, x0))
                    left.add((x0, y0, y0))
                    right.add((x0, y0, y0))

                top.add((y0, x0, x0))

            continue

        elif pair[2] == 360:
            if previous_orientation in {"bot", 270}:
                right.add((x0, y0, y0))
            elif previous_orientation in {"left", 360}:
                right.add((x0, y0, y0))
                bot.add((y0, x0, x0))
            elif previous_orientation == 90:
                right.add((x0, y0, y0))
                bot.add((y0, x0, x0))
                left.add((x0, y0, y0))

            if next_orientation in {"right", 90}:
                bot.add((y1, x1, x1))
            elif next_orientation in {"top", 360}:
                bot.add((y1, x1, x1))
                right.add((x1, y1, y1))
            elif next_orientation == 270:
                bot.add((y1, x1, x1))
                right.add((x1, y1, y1))
                top.add((y1, x1, x1))

            if 180 in (previous_orientation, next_orientation):
                if previous_orientation == 180:
                    bot.add((y0, x0, x0))
                    left.add((x0, y0, y0))
                    top.add((y0, x0, x0))
                    right.add((x0, y0, y0))

            continue

    return [top, bot, left, right], concave_vertices


def compare_lines(line_a, line_b):
    """
        one line is a triple (key, start, end)
        returns -1 if the two are not mergeable
    """

    if line_a[0] != line_b[0]:
        return -1

    line_min, line_max = sorted([line_a, line_b])

    # if the greater start point is <= than the minor end point
    if line_max[1] <= line_min[2]:
        if (line_max[2] <= line_min[2]):
            return tuple(line_min)
        else:
            return tuple([line_min[0], line_min[1], line_max[2]])
    else:
        return -1


def merge_lines(lines_set):
    lines = sorted(list(lines_set))

    i = 0
    while i < len(lines) - 1:

        l = compare_lines(lines[i], lines[i + 1])
        if l is not -1:
            tmp = lines[:i]
            tmp.append(l)
            lines = tmp + lines[i + 2:]
            continue
        i += 1

    return set(lines)


def split_intersections(array, b1, b2):
    """
        To be used with h_lines such that b1-left, b2-right
                   with v_lines such that b1-top, b2-bot
    """
    array = list(array)
    b1 = list(b1)
    b2 = list(b2)

    i = 0
    while i < len(array):
        a = array[i]

        if a[1] != a[2]:
            for b in b1:
                if a[1] < b[0] <= a[2] and b[1] <= a[0] <= b[2]:  # further check if second if is good

                    array.pop(i)
                    array.append((a[0], a[1], b[0] - 1))
                    array.append((a[0], b[0], a[2]))
                    i -= 1
                    break

            else:  # if no v_left intersection, check for v_right
                for b in b2:
                    if a[1] <= b[0] < a[2] and b[1] <= a[0] <= b[2]:
                        array.pop(i)
                        array.append((a[0], a[1], b[0]))
                        array.append((a[0], b[0] + 1, a[2]))
                        i -= 1
                        break
        i += 1
    return set(array)


def find_virtual_lines(ext_lines, hole_lines, concave_vertices):
    sides = reduce(lambda count, l: count + len(l), ext_lines, 0)

    if sides == 4:  # case of simple rectangles
        return []

    virtual_top = set()
    virtual_bot = set()
    virtual_left = set()
    virtual_right = set()

    if hole_lines != -1:
        ext_lines = list(map(lambda x: x[0] | x[1], zip(ext_lines, hole_lines)))

    ext_hor = set([h + ("top",) for h in ext_lines[0]]) | set([h + ("bot",) for h in ext_lines[1]])
    ext_ver = set([v + ("left",) for v in ext_lines[2]]) | set([v + ("right",) for v in ext_lines[3]])

    for c in concave_vertices:

        [x0, y0], [x1, y1] = c[0], c[1]

        if c[2] == 90:  # --

            y_t = find_next_parallel(x0, y0, ext_hor, concave_vertices, "top")
            virtual_right.add((x0, y_t, y0))

            y_t = find_next_parallel(x1, y1, ext_hor, concave_vertices, "top")
            virtual_left.add((x1, y_t, y1))

            x_r = find_next_meridian(x0, y0, ext_ver, concave_vertices, "right")
            virtual_bot.add((y0, x0, x_r))

            x_r = find_next_meridian(x1, y1, ext_ver, concave_vertices, "right")
            virtual_top.add((y1, x1, x_r))

        elif c[2] == 180:  # +-

            y_b = find_next_parallel(x1, y1, ext_hor, concave_vertices, "bot")
            virtual_right.add((x1, y1, y_b))

            y_b = find_next_parallel(x0, y0, ext_hor, concave_vertices, "bot")
            virtual_left.add((x0, y0, y_b))

            x_r = find_next_meridian(x0, y0, ext_ver, concave_vertices, "right")
            virtual_bot.add((y0, x0, x_r))

            x_r = find_next_meridian(x1, y1, ext_ver, concave_vertices, "right")
            virtual_top.add((y1, x1, x_r))

        elif c[2] == 270:  # ++

            y_b = find_next_parallel(x0, y0, ext_hor, concave_vertices, "bot")
            virtual_left.add((x0, y0, y_b))

            y_b = find_next_parallel(x1, y1, ext_hor, concave_vertices, "bot")
            virtual_right.add((x1, y1, y_b))

            x_l = find_next_meridian(x0, y0, ext_ver, concave_vertices, "left")
            virtual_top.add((y0, x_l, x0))

            x_l = find_next_meridian(x1, y1, ext_ver, concave_vertices, "left")
            virtual_bot.add((y1, x_l, x1))

        elif c[2] == 360:  # -+

            y_t = find_next_parallel(x1, y1, ext_hor, concave_vertices, "top")
            virtual_left.add((x1, y_t, y1))

            x_l = find_next_meridian(x0, y0, ext_ver, concave_vertices, "left")
            virtual_top.add((y0, x_l, x0))

            x_l = find_next_meridian(x1, y1, ext_ver, concave_vertices, "left")
            virtual_bot.add((y1, x_l, x1))

    return [virtual_top, virtual_bot, virtual_left, virtual_right]


def find_next_parallel(x, y, h_lines, concave_vertices, direction):
    try:
        lines = [h for h in h_lines if h[1] <= x <= h[2]]
        if direction == "top":
            l = max([l for l in lines if l[0] < y or (l[0] == y and l[3] == direction)], key=lambda h: h[0])
        else:
            l = min([l for l in lines if l[0] > y or (l[0] == y and l[3] == direction)], key=lambda h: h[0])
        ret = l[0]

    except:
        if direction == "top":
            ret = [max({c[0][1], c[1][1]}) for c in concave_vertices if
                   {c[0][0], c[1][0]} & {x, x} and y > c[0][1]][0]
        else:
            ret = [min({c[0][1], c[1][1]}) for c in concave_vertices if
                   {c[0][0], c[1][0]} & {x, x} and y < c[0][1]][0]

    return ret


def find_next_meridian(x, y, v_lines, concave_vertices, direction):
    try:
        lines = [v for v in v_lines if v[1] <= y <= v[2]]

        if direction == "left":
            l = max([l for l in lines if l[0] < x or (l[0] == x and l[3] == direction)], key=lambda v: v[0])

        else:
            l = min([l for l in lines if (l[0] > x) or (l[0] == x and l[3] == direction)], key=lambda v: v[0])

        ret = l[0]

    except:

        if direction == "left":
            ret = [max({c[0][0], c[1][0]}) for c in concave_vertices if
                   {c[0][1], c[1][1]} & {y, y} and x > c[1][0]][0]
        else:
            ret = [min({c[0][0], c[1][0]}) for c in concave_vertices if
                   {c[0][1], c[1][1]} & {y, y} and x < c[1][0]][0]
    return ret

def partition_contour(external_lines, virtual_lines, hole_lines):
    elements_found = []

    sides = reduce(lambda count, l: count + len(l), external_lines, 0)

    if sides == 4:  # case of simple rectangles
        top = list(external_lines[0])[0]
        start = (top[1], top[0])

        right = list(external_lines[3])[0]
        end = (right[0], right[2])

        elements_found.append([start, end])
        return elements_found

    if hole_lines == -1:
        hole_lines = [set()] * 4
    top, bot, left, right = list(map(lambda x, y, z: x | y | z, external_lines, virtual_lines, hole_lines))

    top = merge_lines(top)
    bot = merge_lines(bot)
    left = merge_lines(left)
    right = merge_lines(right)

    top = split_intersections(top, left, right)
    bot = split_intersections(bot, left, right)
    left = split_intersections(left, top, bot)
    right = split_intersections(right, top, bot)

    top_points = set()
    bot_points = set()

    for h in bot:
        bot_points.add((h[2], h[0]))
    for h in top:
        top_points.add((h[1], h[0]))

    for v in left:
        top_points.add((v[0], v[1]))
    for v in right:
        bot_points.add((v[0], v[2]))

    if len(top_points) != len(bot_points):
        print("len top", len(top_points), "len bot", len(bot_points), "\n")
        print("top points", sorted(list(top_points)))
        print("bot points", sorted(list(bot_points)), "\n")

    top_points = sorted(list(top_points), key=lambda x: (x[0], x[1]))
    bot_points = sorted(list(bot_points), key=lambda x: (x[0], x[1]))

    for t in top_points:
        bot_limit = min([x for x in bot if t[0] in range(x[1], x[2] + 1) and x[0] >= t[1]])[0]
        right_limit = min([x for x in right if t[1] in range(x[1], x[2] + 1) and x[0] >= t[0]])[0]

        list1 = [b for b in bot_points if b[0] <= right_limit and b[1] <= bot_limit]
        b = [b for b in list1 if b[0] >= t[0] and b[1] >= t[1]][0]

        bot_points.remove(b)
        elements_found.append([t, b])

    lines = [top, bot, left, right]
    return elements_found

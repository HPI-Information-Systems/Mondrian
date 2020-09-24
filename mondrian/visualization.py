import csv
import os
import pickle
import random
import PIL
import numpy as np
from pathlib import Path
from PIL import Image, ImageFont
from skimage import feature

import cv2 as cv
import matplotlib.pyplot as plt
from PIL import ImageDraw

from . import colors as colors
from .parsing import parse_cell

from .partition import find_external_lines, partition_contour, find_virtual_lines


def lbp_describe(image, numPoints=8, radius=1, eps=1e-7):
    """
    This function computes the Local Binary Pattern representation
    of the image, and then use the LBP representation
    to build the histogram of patterns
    """
    lbp = feature.local_binary_pattern(image, numPoints,
                                       radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))

    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def table_as_image(path, delimiter=',', color=False, cell_length=False):
    """
    This function imports a csv file, with a custom delimiter, and renders it into a binary array with white pixels where cells are full, and black pixels where they are empty.
    Definition of empty cell is everything that does not contain at least an alphanumeric/special character. (e.g. no only whitespace characters)
    If cell_length = True, then each cell is rendered with a number of pixels proportionate to its length

    Params:
    -------
    path: string
        The path of the csv file
    delimiter: string
        The delimiter character
    cell_length: boolean
        If every cell should be parsed with a number of pixels equivalent to its length

    Returns:
    --------
    img: numpy.array
        A binary array of shape (width, lenght, 3)
    """

    img = []
    last_nonempty = 0
    with open(path, 'r', encoding="UTF-8") as csvfile:
        tablereader = csv.reader(csvfile, delimiter=delimiter)
        max_size = 0
        for idx, line in enumerate(tablereader):
            if line != [''] * len(line):
                last_nonempty = idx

            result = [parse_cell(val, color) for val in line]
            if cell_length:
                result = [r for idx, r in enumerate(result) for _ in range(len(line[idx]))]

            if len(result) > max_size:
                max_size = len(result)

            img.append(result)

    img = img[:last_nonempty + 1][:]

    for idx, line in enumerate(img):
        line += [[255, 255, 255]] * (max_size - len(line))

    img = np.asarray(img, dtype="uint8")

    return img


def find_table_elements(img, partitioning=False, lbp=False, save_path=None):
    ''' This function finds the components of a table, whether connected or after partitioning.
    :param img: np.array
    :param partitioning: bool
    :param lbp: bool
    :param save_path:
        The path where the file should be saved to, including filename
    :return:
        elements_found: sequence
        contours: sequence
        hierarchy: sequence
    '''

    save = False
    if save_path:
        try:
            elements_found, contours, hierarchy = pickle.load(open(save_path, "rb"))
            return elements_found, contours, hierarchy
        except:
            save = True

    img = 255 - cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, None, 1)

    elements_found = []
    for idx, c in enumerate(contours):
        label = idx
        inner_area = cv.contourArea(c)
        x, y, width, height = cv.boundingRect(c)
        x0, x1, y0, y1 = x, x + width - 1, y, y + height - 1
        c_x = x + np.floor(width / 2)
        c_y = y + np.floor(height / 2)
        area = width * height

        if area == 0:
            area = inner_area
        extent = inner_area / area

        diagonal = np.sqrt(width ** 2 + height ** 2)

        aspect_ratio = float(width) / height

        e = {
            "label": label,
            "top_lx": [x0, y0],
            "bot_rx": [x1, y1],
            "center": [c_x, c_y],
            "aspect_ratio": aspect_ratio,
            "width": width,
            "height": height,
            "area": area,
            "inner_area": inner_area,
            "extent": extent,
            "diagonal": diagonal
        }

        if lbp:
            region = Image.fromarray(img).crop((x, y0, x1 + 1, y1 + 1))
            region = np.asarray(region)
            h = lbp_describe(region, numPoints=8, radius=8)
            e["lbp"] = h

        elements_found.append(e)

    if partitioning:
        elements_found = []
        for c in contours:
            external_lines, vertices = find_external_lines(c)
            hole_lines = -1
            virtual_lines = find_virtual_lines(external_lines, hole_lines, vertices)
            rectangles = partition_contour(external_lines, virtual_lines, hole_lines)
            for r in rectangles:
                h, w = np.subtract(r[1], r[0])
                area = np.multiply.reduce((h + 1, w + 1))
                elements_found.append({
                    "top_lx": list(r[0]),
                    "bot_rx": list(r[1]),
                    "area": area,
                })

    if save:
        Path(os.path.split(save_path)[0]).mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump([elements_found, contours, hierarchy], f)

    return elements_found, contours, hierarchy


def partition_contours(list_contours):
    elements_found = []

    for contour in list_contours:

        top_points = []
        bot_points = []
        rectangles = []

        if np.size(contour) == 1:  # case of single pixels
            rectangles.append([contour[0], contour[0]])
            continue

        for i in range(1, np.size(contour) + 1):
            print(contour[i])
            [x0, y0] = contour[i - 1]
            [x1, y1] = contour[i]

            x_values = np.asarray([x[0] for x in contour])

            left_x, right_x = sorted([x0, x1])

            next_right = np.min([x for x in x_values - right_x if x > 0]) + right_x
            next_left = np.max([x for x in x_values - left_x if x < 0]) + left_x

            if y0 == y1:
                if x0 > x1:
                    top_points.append([x1, y1])
                else:
                    bot_points.append([x1, y1])

            if x0 == x1:
                if y0 > y1:
                    bot_points.append([x0, y0])
                else:
                    top_points.append([x0, y0])

            if y0 == y1 + 1:
                top_points.append([next_left, y0])
                if x0 == x1 + 1:
                    bot_points.append([x1, y1])
                elif x0 == x1 - 1:
                    bot_points.append([x0, y1])

            if y0 == y1 - 1:
                top_points.append([next_right, y0])
                if x0 == x1 + 1:
                    top_points.append([x1, y1])
                elif x0 == x1 - 1:
                    top_points.append([x0, y1])

        top_points = sorted(top_points, key=lambda x: (x[0], x[1]))
        bot_points = sorted(bot_points, key=lambda x: (x[0], x[1]))

        rectangles = zip(top_points, bot_points)

        elements_found.append(rectangles)

    return elements_found


def draw_contours(img, contours, labels=None):
    """
    A wrapper function to draw colored contours in case clustering labels are present. 
    For convenience, it can also be used to draw b/w contours if no labels present.
    
    Parameters
    ----------
    img : numpy matrix 
        Binary input image
    contours: List of lists 
        Element contours as returned by cv.findContours
    labels: Integer list 
        A label for each element of the contour list, as returned by a scipy clustering object
    
    Returns
    ---------
    cnt: numpy matrix
        Input image with highlighted borders
    
    """

    cnt = img.copy()

    if labels is not None:
        for i in range(0, np.size(contours)):
            cluster = labels[i]
            random.seed(cluster ** 2)  # differentiate more with the square
            color = random.randint(0, 255)
            cv.drawContours(cnt, contours, i, (color, 0, 0), 1)
    else:
        cv.drawContours(cnt, contours, -1, (125, 0, 0), 1)

    return cnt


def print_contours(img, contours, labels=None, n_lines=100, triple=False):
    """
    Wrapper function because I'm bored of using the same code every time to plot contours.
    
    Parameters
    ----------
    img, contours, labels: numpy matrix, list of lists, list of integers
        The input parameters for draw_contours
        
    n_lines: integer
        The number of lines of the original image to print out
    
    triple: boolean
        Wheter to plot a triple plot or not (original, borders, clustered)
    """

    cnt = draw_contours(img, contours, labels)

    print("Subset of the image, ", n_lines, "lines")

    if not triple:
        plt.figure(figsize=(10, 25))
        plt.subplot(1, 2, 1)
        plt.imshow(img[:n_lines, :, 0], cmap="Greys")

        plt.subplot(1, 2, 2)
        plt.imshow(cnt[:n_lines, :, 0], cmap="nipy_spectral")
        plt.pause(1)

    else:
        brd = draw_contours(img, contours)

        plt.figure(figsize=(15, 25))
        plt.subplot(1, 3, 1)
        plt.imshow(img[:n_lines, :, 0], cmap="Greys")

        plt.subplot(1, 3, 2)
        plt.imshow(brd[:n_lines, :, 0], cmap="nipy_spectral")

        plt.subplot(1, 3, 3)
        plt.imshow(cnt[:n_lines, :, 0], cmap="nipy_spectral")
        plt.pause(1)


def save_colored_image(img, path):
    # to_save is a uint8 array
    to_save = (img ** 2 - img)

    to_save = np.concatenate((to_save, to_save, to_save), axis=2)
    mask = (to_save != 0)

    to_save[:, :, 0] %= 180
    to_save[:, :, 1] = 255
    to_save[:, :, 2] = 255
    to_save *= mask

    to_save = cv.cvtColor(to_save, cv.COLOR_HSV2RGB)

    cv.imwrite(path, to_save)


def save_image(img, path, grid=False):
    im = Image.fromarray(img)
    FACTOR = 20
    im = im.resize(tuple(np.array(im.size) * FACTOR), resample=PIL.Image.NEAREST)
    if grid:
        rows, cols = np.shape(img)[0:2]
        height, width = im.height, im.width
        line_width = int(FACTOR * 0.05)
        x_points = np.asarray(range(cols + 1)) * FACTOR
        y_points = np.asarray(range(rows + 1)) * FACTOR
        v_lines = [(x - line_width, 0, height) for x in x_points]
        h_lines = [(y - line_width, 0, width) for y in y_points]
        draw = ImageDraw.Draw(im)
        draw_lines(draw, [h_lines, h_lines, v_lines, v_lines], color=tuple(colors.RGB_BLACK), width=line_width)

    Path(os.path.split(path)[0]).mkdir(parents=True, exist_ok=True)
    im.save(path)


def draw_lines(draw, lines, color=(255, 0, 0), width=5, factor=1):
    """
    Given a draw object on an image, it adds to it the lines specified in the lines array

    :param draw: a PIL ImageDraw object on an RGB Image
    :param lines: a list of 4 lists containing lines in the order (top, bot, left, right)
    :param color: color of the lines to be drawn
    :param width: width of the lines to be drawn
    :param factor: the scaling factor if the image has been rescaled
    :return the draw object
    """
    top, bot, left, right = lines

    for h in top:
        draw.line((h[1] * factor, h[0] * factor) + ((h[2] + 1) * factor, h[0] * factor), fill=color, width=width)

    for h in bot:
        draw.line((h[1] * factor, (h[0] + 1) * factor) + ((h[2] + 1) * factor, (h[0] + 1) * factor), fill=color,
                  width=width)

    for v in left:
        draw.line((v[0] * factor, v[1] * factor) + (v[0] * factor, (v[2] + 1) * factor), fill=color, width=width)

    for v in right:
        draw.line(((v[0] + 1) * factor, v[1] * factor) + ((v[0] + 1) * factor, (v[2] + 1) * factor), fill=color,
                  width=width)

    return draw


def draw_grid(draw, img_size, color, factor=1, line_width=1, text=False, text_color=None):
    h, w, _ = img_size

    for x in range(w):
        draw.line([x * factor, 0, x * factor, h * factor], fill=color, width=line_width)
    for y in range(h):
        draw.line([0, y * factor, w * factor, y * factor], fill=color, width=line_width)

    if text:
        font = ImageFont.truetype("arial.ttf", int(factor / 3))

        for i, x in enumerate(range(w)):
            for j, y in enumerate(range(h)):
                draw.text([x * factor + int(factor / 6), y * factor + int(factor / 6)], text=str(i) + "," + str(j), fill=text_color, font=font)


def draw_rectangles(draw, rectangles, labels=None, factor=1, color=None, out_color=None, alpha=255, width=0):
    """
    Add rectangles on a draw object

    :param color:
    :param draw: a PIL ImageDraw object on an RGB Image
    :param rectangles: a list of rectangles in the form ( top_rx_corner, bot_lx_corner)
    :param labels: if labels are present, each rectangle with same label has same color
    :param factor: the scaling factor if the image has been rescaled
    :param alpha: the transparency of the fill (255 is opaque)
    :param width: the thickness of the border line
    :return: the draw object
    """

    color_list = []
    for i, r in enumerate(rectangles):
        rect = ((r["top_lx"][0] * factor, r["top_lx"][1] * factor),
                ((r["bot_rx"][0] + 1) * factor, (r["bot_rx"][1] + 1) * factor))
        if labels is not None:
            random.seed(labels[i])  # same label, same color
        if color is None:
            draw_color = (*colors.rgb_region_colors[i % len(colors.rgb_region_colors)], alpha)
        else:
            draw_color = (*color, alpha)
        outline = (*out_color, alpha) if out_color is not None else draw_color

        draw.rectangle(rect, fill=draw_color, outline=outline, width=width)

        color_list.append(list(draw_color))

    return draw, color_list


def print_images(fname, img, predicted_labels, predicted_cluster_edges, target_cluster_edges):
    im = Image.fromarray(img)
    FACTOR = 20
    im = im.resize(tuple(np.array(im.size) * FACTOR))
    im = im.convert("RGB")
    gold_im = im.copy()
    draw = ImageDraw.Draw(im, "RGBA")
    gold_draw = ImageDraw.Draw(gold_im, "RGBA")

    draw = ImageDraw.Draw(im)
    images = [im, gold_im]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + 50
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + 50
    new_im.save('res/' + fname + '.jpg')

import cv2 as cv
import numpy as np

# https://www.imgonline.com.ua/eng/color-palette.php
from webcolors import hex_to_rgb

RGB_BLACK = np.asarray([0, 0, 0])
RGB_WHITE = np.asarray([255, 255, 255])
RGB_RED = np.asarray([255, 0, 0])
RGB_MAROON = np.asarray([128, 0, 0])
RGB_SALMON = np.asarray([255, 128, 128])
RGB_TOMATO = np.asarray([255, 75, 75])
RGB_BLUE = np.asarray([0, 0, 255])
RGB_LIME = np.asarray([0, 255, 0])
RGB_GREEN = np.asarray([0, 128, 0])
RGB_AQUA = np.asarray([0, 255, 255])

HSV_BLACK = cv.cvtColor(np.uint8([[RGB_BLACK]]), cv.COLOR_RGB2HSV)
HSV_WHITE = cv.cvtColor(np.uint8([[RGB_WHITE]]), cv.COLOR_RGB2HSV)
HSV_RED = cv.cvtColor(np.uint8([[RGB_RED]]), cv.COLOR_RGB2HSV)
HSV_GREEN = cv.cvtColor(np.uint8([[RGB_GREEN]]), cv.COLOR_RGB2HSV)
HSV_BLUE = cv.cvtColor(np.uint8([[RGB_BLUE]]), cv.COLOR_RGB2HSV)

hex_region_colors = ['#51933a',
                     '#9c5c2c',
                     '#6686e2',
                     '#9b7f30',
                     '#317341',
                     '#d7407e',
                     '#a9b365',
                     '#825bcd',
                     '#c58cd6',
                     '#91ba35',
                     '#42c0c7',
                     '#657028',
                     '#d1522b',
                     '#c853b8',
                     '#5dc99c',
                     '#e1936a',
                     '#b35659',
                     '#e385a5',
                     '#54c25e',
                     '#6160a3',
                     '#9c4b7b',
                     '#419a76',
                     '#5f9fd3',
                     '#d34250',
                     '#d7a135']

rgb_region_colors = [hex_to_rgb(c)[0:3] for c in reversed(hex_region_colors)]


class term_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\u001b[32;1m'
    WARNING = '\u001b[31;1m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import warnings
from dateutil.parser._parser import UnknownTimezoneWarning
warnings.simplefilter(action='ignore', category=UnknownTimezoneWarning)

import datetime
from backports.datetime_fromisoformat import MonkeyPatch
MonkeyPatch.patch_fromisoformat()

import re
from enum import Enum

import mondrian.colors as colors
import numpy as np
import dateutil.parser


class CellType():
    EMPTY = colors.RGB_WHITE
    NON_EMPTY = colors.RGB_BLACK
    INTEGER = colors.RGB_AQUA
    FLOAT = colors.RGB_BLUE
    TIME = colors.RGB_LIME
    DATE = colors.RGB_GREEN
    STRING_UPPER = colors.RGB_MAROON
    STRING_LOWER = colors.RGB_SALMON
    STRING_TITLE = colors.RGB_TOMATO
    STRING_GENERIC = colors.RGB_RED


class customDateParserInfo(dateutil.parser.parserinfo):
    JUMP = [' ', '.', ',', ';', '-', '/', "'"]


def parse_cell(val, color=False):
    # if cell_length else CellType.EMPTY

    if not val.split() or val.isspace():
        return CellType.EMPTY

    if not color:
        return CellType.NON_EMPTY

    comma_split = val.split(",")
    # it's a number like 1,123 or something
    if any([len(x) == 3 for x in comma_split]) and not any([x.isalpha() for x in val]):
        val = re.sub(",", "", val)  # potentially, the float like 1,234 becomes an integer 1234

    elif len(comma_split) == 2 and not any([x.isalpha() for x in val]):
        val = re.sub(",", ".", val)

    val = val.lstrip().rstrip()

    try:
        int(val)
        return CellType.INTEGER
    except ValueError:
        pass
    try:
        float(val)
        return CellType.FLOAT
    except ValueError:
        pass
    try:
        datetime.time.fromisoformat(val)
        return CellType.TIME
    except ValueError:
        pass
    try:
        dateutil.parser.parse(val, parserinfo=customDateParserInfo())
        return CellType.DATE
    except ValueError:
        pass
    except TypeError:
        # print("\tTypeError parsing cell as date:", val)
        pass

    if val.isupper():
        return CellType.STRING_UPPER
    elif val.islower():
        return CellType.STRING_LOWER
    elif val.istitle():
        return CellType.STRING_TITLE

    return CellType.STRING_GENERIC

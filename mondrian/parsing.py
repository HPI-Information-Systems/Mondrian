import datetime
from backports.datetime_fromisoformat import MonkeyPatch
import re
from . import colors as colors
import dateutil.parser

MonkeyPatch.patch_fromisoformat()


class CellType:
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
    if not val.split() or val.isspace():
        return CellType.EMPTY

    if not color:
        return CellType.NON_EMPTY

    comma_split = val.split(",")
    # can be a number like 1,123 or something
    if any([len(x) == 3 for x in comma_split]) and not any([x.isalpha() for x in val]):
        val = re.sub(",", "", val)
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
        pass

    if val.isupper():
        return CellType.STRING_UPPER
    elif val.islower():
        return CellType.STRING_LOWER
    elif val.istitle():
        return CellType.STRING_TITLE

    return CellType.STRING_GENERIC

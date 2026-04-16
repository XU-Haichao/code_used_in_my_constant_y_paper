"""MNRAS journal plotting style configuration."""

import matplotlib.pyplot as plt

SINGLE_COLUMN_WIDTH = 3.5
SINGLE_COLUMN_HEIGHT = 3.0
DOUBLE_COLUMN_WIDTH = 7.2
DOUBLE_COLUMN_HEIGHT = 3.0

# Backward-compatible aliases for existing single-column scripts.
fig_width = SINGLE_COLUMN_WIDTH
fig_height = SINGLE_COLUMN_HEIGHT


def get_single_column_size(row_height_scale=1.0):
    """Return the single-column figure size.

    Parameters
    ----------
    row_height_scale : float, optional
        Multiplier applied to the default figure height.
    """

    return SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_HEIGHT * row_height_scale


def get_double_column_size(row_height_scale=1.0):
    """Return the double-column figure size.

    Parameters
    ----------
    row_height_scale : float, optional
        Multiplier applied to the default figure height. Use values
        larger than 1 to increase the row height for taller panels.
    """

    return DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_HEIGHT * row_height_scale


def get_figure_size(columns=1, row_height_scale=1.0):
    """Return an MNRAS-style figure size for one or two columns."""

    if columns == 1:
        return get_single_column_size(row_height_scale=row_height_scale)
    if columns == 2:
        return get_double_column_size(row_height_scale=row_height_scale)
    raise ValueError("columns must be 1 or 2")


def apply_mnras_style():
    """Apply MNRAS-compliant matplotlib style settings."""

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["axes.linewidth"] = 1.0

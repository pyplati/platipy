import math

import numpy as np
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from platipy.imaging.label.utils import get_com

from platipy.imaging.visualisation.visualiser import ImageVisualiser

from platipy.imaging.dose.dvh import (
    calculate_dvh_for_labels,
    calculate_d_x,
    calculate_v_x,
    calculate_d_cc_x,
)


def is_color_dark(color):
    """Returns true if the color is dark, false otherwise.

    Args:
        color (list): A list of values corresponging to r,g,b

    Returns:
        bool: True if dark, False otherwise
    """

    [r, g, b] = color
    hsp = math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    if hsp > 0.5:
        return False
    else:
        return True


def roundup(x, nearest):
    """Round up to the nearest multiple.

    Args:
        x (float|int): Value to round up
        nearest (float|int): Multiple to round up to

    Returns:
        int: Rounded up value
    """
    return int(math.ceil(x / float(nearest))) * nearest


def visualise_dose(
    img,
    dose,
    structures,
    dvh=None,
    d_points=None,
    v_points=None,
    d_cc_points=None,
    structure_for_com=None,
    structure_for_limits=None,
    expansion_for_limits=10,
    title="",
    contour_cmap=matplotlib.colormaps.get_cmap("rainbow"),
    dose_cmap=matplotlib.colormaps.get_cmap("inferno"),
    structure_name_dict=None,
    img_vis_kw=None,
):
    """A function to visualise an image with dose grid overlaid along with the DVH and dose
    metrics.

    Args:
        img (sitk.Image): Image on which to overlay the dose grid
        dose (sitk.Image): Dose grid
        structures (dict): Dictionary of structures (corresponding to those provided in DVH, if
            provided) with structure name as key and sitk.Image of mask as value.
        dvh (pd.DataFrame, optional): Dose Volume Histogram, if not provided it will be computed.
            Defaults to None.
        d_points (list|float|int, optional): The points at which to calculate D metrics. Defaults
            to None.
        v_points (list|float|int, optional): The points at which to calculate V metrics. Defaults
            to None.
        d_cc_points (list|float|int, optional): Points at which to calculate Dcc metrics. Defaults
            to None.
        structure_for_com (sitk.Image, optional): Mask to use to compute the centre of mass to cut
            the image slices. Defaults to None.
        structure_for_limits (sitk.Image, optional): Mask to set the limits of the visualisation.
            Defaults to None.
        expansion_for_limits (int, optional): Expansion around the structure for limits. Defaults
            to 10.
        title (str, optional): Title to display on visualisation. Defaults to "".
        contour_cmap (plt.cm.colormap, optional): Matplotlib color map to use for contour colors.
            Defaults to matplotlib.colormaps.get_cmap("rainbow").
        dose_cmap (plt.cm.colormap, optional): Matplotlib color map to use for dose colors.
            Defaults to matplotlib.colormaps.get_cmap("inferno").
        structure_name_dict (dict, optional): Dictionary to map alternative structure names.
            Defaults to None.
        img_vis_kw (dict, optional): Dictionary of keyword arguments to pass to ImageVisualiser.
            Defaults to None.

    Returns:
        matplotlib.figure: The figure, can be saved as usual (fig.savefig(...)).
        pd.DataFrame: The dose metrics computed for display in the table.
    """

    if img_vis_kw is None:
        img_vis_kw = {}

    if dvh is None:
        dvh = calculate_dvh_for_labels(dose, structures)
    else:
        dvh = dvh.copy()

    df_metrics = dvh[["label", "mean"]]

    if d_points is not None:
        df_metrics_d = calculate_d_x(dvh, d_points)
        df_metrics = df_metrics.merge(df_metrics_d)

    if v_points is not None:
        df_metrics_v = calculate_v_x(dvh, v_points)
        df_metrics = df_metrics.merge(df_metrics_v)

    if d_cc_points is not None:
        df_metrics_d_cc = calculate_d_cc_x(dvh, d_cc_points)
        df_metrics = df_metrics.merge(df_metrics_d_cc)

    df_metrics = df_metrics.set_index("label")

    cut = None

    if structure_for_com is None:
        # Not structure specified, get the area of highest dose and get the COM of that
        cut = get_com(dose > dose * 0.9)
    else:
        # the user has specified a structure
        cut = get_com(structures[structure_for_com])

    img_vis_kw["cut"] = cut

    vis = ImageVisualiser(img, **img_vis_kw)

    # Resample the dose grid into the image space
    dose = sitk.Resample(dose, img)

    # Cut off lowest 10% of dose for visualisation
    arr = sitk.GetArrayFromImage(dose)
    arr[arr < arr.max() * 0.1] = 0
    dose = sitk.GetImageFromArray(arr)
    dose.CopyInformation(img)

    round_to_nearest = 5
    if arr.max() < 20:
        round_to_nearest = 1

    if arr.max() < 2:
        round_to_nearest = 0.1

    max_val = roundup(arr.max(), round_to_nearest)

    vis.add_scalar_overlay(
        dose,
        discrete_levels=int(max_val / round_to_nearest),
        colormap=dose_cmap,
        alpha=0.5,
        max_value=max_val,
        name="Dose (Gy)",
    )
    vis.add_contour(structures, show_legend=False, colormap=contour_cmap)

    if structure_for_limits is not None:
        vis.set_limits_from_label(structure_for_limits, expansion=expansion_for_limits)

    fig = vis.show()

    dvh.index = dvh.label
    dvh.transpose()
    plt_dvh = dvh.iloc[:, 3:].transpose()

    ax = fig.axes[1]

    subax_x = (ax.bbox.x0 / fig.bbox.width) + 0.175
    subax_y = (ax.bbox.y0 / fig.bbox.height) + 0.05

    subax = fig.add_axes([subax_x, subax_y, 1 - subax_x, 1 - subax_y - 0.3])
    plt_dvh.plot(ax=subax, colormap=contour_cmap, legend=False)

    plt.xlabel("Dose (Gy)")
    plt.ylabel("Frequency")
    plt.title("Dose Volume Histogram (DVH)")

    if structure_name_dict is not None:
        rows = [structure_name_dict[i] for i in list(df_metrics.index)]
    else:
        rows = list(df_metrics.index)
    columns = list(df_metrics.columns)
    cell_text = []
    for _, row in df_metrics.iterrows():
        cell_text.append([f"{s:.2f}" for s in row.values])

    colors = list(matplotlib.colormaps.get_cmap(contour_cmap)(np.linspace(0, 1, len(rows))))

    # Create the table
    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        rowColours=colors,
        colLabels=columns,
        fontsize=10,
        bbox=[0.4, 0.5, 0.6, 0.4],
    )

    # Some nice formatting
    for cell in table.get_celld():
        table[cell].set_text_props(va="center")
        table[cell].set_edgecolor("w")

        if cell[0] == 0:  # header
            table[cell].set_text_props(weight="bold", color="w")
            table[cell].set_facecolor("k")

    # Geometry fixes
    for row in range(len(rows) + 1):
        for c in range(len(columns)):
            table[row, c].set_width(0.1)
        if row > 0:
            table[row, -1].set_width(0)

            if is_color_dark(colors[row - 1][:3]):
                table[row, -1].set_text_props(color="w")
            else:
                table[row, -1].set_text_props(color="k")

    for col in range(len(columns)):
        table[0, col].set_facecolor("k")

    table.auto_set_font_size(True)
    font_size = table.get_celld()[1, 0].get_fontsize()

    # insert metadata information
    ax.text(
        x=0.25,
        y=0.96,
        s=title,
        color="black",
        ha="left",
        va="top",
        size=font_size,
        wrap=True,
        weight="bold",
    )

    # Return the figure
    return fig, df_metrics

# Copyright 2020 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from platipy.imaging.label.utils import get_com
from platipy.imaging import ImageVisualiser

import numpy as np

import SimpleITK as sitk

import matplotlib.lines as mlines
import matplotlib as plt


from platipy.imaging.label.comparison import (
    compute_metric_dsc,
    compute_metric_hd,
    compute_metric_masd,
)


def contour_comparison(
    img,
    contour_dict_a,
    contour_dict_b,
    contour_label_a="Set A",
    contour_label_b="Set B",
    s_select=None,
    structure_for_com=None,
    structure_for_limits=None,
    title="TITLE TEXT",
    subtitle="SUBTITLE TEXT",
    subsubtitle="SUBSUBTITLE TEXT",
    contour_cmap=plt.cm.get_cmap("rainbow"),
    structure_name_dict=None,
    img_vis_kw={},
):
    """Generates a custom figure for comparing two sets of contours (delineations) on an image.

    Includes a formatted table of contouring (similarity) metrics.

    Args:
        img (SimpleITK.Image): The image to visualise.
        contour_dict_a (dict): A dict object with contour names (str) as keys and SimpleITK.Image
            as values.
        contour_dict_b (dict): A dict object with contour names (str) as keys and SimpleITK.Image
            as values.
        contour_label_a (str, optional): Name to put in legend. Defaults to "Set A".
        contour_label_b (str, optional): Name to put in legend. Defaults to "Set B".
        s_select (list, optional): A list of contours to display. Defaults to all keys common to
            both contour sets.
        structure_for_com (str, optional): Name of the contour to set slice location.
            Defaults to the largest structure by volume.
        structure_for_limits (str, optional): Name of the contour to set view limits.
         Defaults to None, and entire image is displayed.
        title (str, optional): Title of the plot, set to "" to remove. Defaults to "TITLE TEXT".
        subtitle (str, optional): Subitle of the plot, set to "" to remove.
            Defaults to "SUBTITLE TEXT".
        subsubtitle (str, optional): Subsubtitle of the plot, set to "" to remove.
            Defaults to "SUBSUBTITLE TEXT".
        contour_cmap (plt.cm.colormap, optional): Contour colormap.
            Defaults to plt.cm.get_cmap("rainbow").
        structure_name_dict (dict, optional): A "translation" dictionary used to overwrite the
            names of contours in the metric table. Defaults to using the names in the contour_dict.
        img_vis_kw (dict, optional): Passed to the ImageVisualiser class. Defaults to {}.

    Returns:
        matplotlib.figure: The figure, can be saved as usual (fig.savefig(...)).
    """

    # If no contour names are seleted we just use those which both contour_dicts have
    if s_select is None:
        s_select = [i for i in contour_dict_a.keys() if i in contour_dict_b.keys()]

    # If there is no option for the COM structure we use the largest
    if structure_for_com is None:
        s_vol = [sitk.GetArrayFromImage(contour_dict_a[s]).sum() for s in s_select]
        structure_for_com = s_select[np.argmax(s_vol)]

    # Visualise!
    vis = ImageVisualiser(img, cut=get_com(contour_dict_a[structure_for_com]), **img_vis_kw)

    # Add contour set A
    vis.add_contour(
        {s + "a": contour_dict_a[s] for s in s_select},
        show_legend=False,
        color={
            s + "a": c
            for s, c in zip(
                s_select, plt.cm.get_cmap(contour_cmap)(np.linspace(0, 1, len(s_select)))
            )
        },
    )

    # Add contour set B
    vis.add_contour(
        {s + "b": contour_dict_b[s] for s in s_select},
        show_legend=False,
        color={
            s + "b": c
            for s, c in zip(
                s_select, plt.cm.get_cmap(contour_cmap)(np.linspace(0, 1, len(s_select)))
            )
        },
        linestyle="dashed",
    )

    # Optionally, set limits
    if structure_for_limits is not None:
        vis.set_limits_from_label(contour_dict_a[structure_for_limits], expansion=30)

    # Create the figure
    fig = vis.show()

    # Choose the blank axis (top right)
    ax = fig.axes[1]

    # Set row names is structure_name_dict is given
    if structure_name_dict is not None:
        rows = [structure_name_dict[i] for i in s_select]
    else:
        rows = s_select

    # Compute some metrics
    columns = ("DSC", "MDA [mm]", "HD [mm]")
    cell_text = []
    for s, row in zip(s_select, rows):
        cell_text.append(
            [
                f"{compute_metric_dsc(contour_dict_a[s],contour_dict_b[s]):.2f}",
                f"{compute_metric_masd(contour_dict_a[s],contour_dict_b[s]):.2f}",
                f"{compute_metric_hd(contour_dict_a[s],contour_dict_b[s]):.2f}",
            ]
        )

    # Create the table
    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        rowColours=plt.cm.get_cmap(contour_cmap)(np.linspace(0, 1, len(s_select))),
        colLabels=columns,
        fontsize=10,
        bbox=[0.4, 0.15, 0.55, 0.6],
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

        table[row, 0].set_width(0.2)
        table[row, 1].set_width(0.2)
        table[row, 2].set_width(0.2)
        if row > 0:
            table[row, -1].set_width(0.2)

    table.auto_set_font_size(True)
    fs = table.get_celld()[1, 0].get_fontsize()

    # Insert text
    ax.text(
        0.95,
        0.98,
        title,
        color="navy",
        ha="right",
        va="top",
        size=fs + 4,
    )
    ax.text(0.95, 0.90, subtitle, color="darkgreen", ha="right", va="top", size=fs)
    ax.text(0.95, 0.85, subsubtitle, color="k", ha="right", va="top", size=fs)

    # Insert legend
    _solid = mlines.Line2D([], [], color="k", label=contour_label_a)
    _dashed = mlines.Line2D([], [], color="k", linestyle="dashed", label=contour_label_b)
    ax.legend(
        handles=[_solid, _dashed],
        bbox_to_anchor=(0.4, 0.02, 0.55, 0.1),
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
        fontsize=fs,
    )

    # Return the figure
    return fig

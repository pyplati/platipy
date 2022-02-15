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

import pandas as pd

import matplotlib.lines as mlines
import matplotlib as plt
import matplotlib.colors as mcolors


from platipy.imaging.label.comparison import (
    compute_metric_dsc,
    compute_metric_hd,
    compute_metric_masd,
    compute_volume,
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
    title="",
    subtitle="",
    subsubtitle="",
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

    if "cut" not in img_vis_kw:

        cut = None

        # first check if the user has specified a structure
        if structure_for_com is None:
            # If there is no option for the COM structure we use the largest
            # first calculate the "volume" of each structure (actually the sum of voxels)
            s_vol = [sitk.GetArrayFromImage(contour_dict_a[s]).sum() for s in s_select]

            if sum(s_vol) == 0:
                # if all structures are zero, try the second contour set
                s_vol = [sitk.GetArrayFromImage(contour_dict_b[s]).sum() for s in s_select]

                if sum(s_vol) == 0:
                    # if all of these structures are also zero, we don't have any contours!
                    cut = None

                else:
                    # otherwise, get the COM of the largest structure (in contour_set_b)
                    cut = get_com(contour_dict_b[s_select[np.argmax(s_vol)]])
            else:
                # otherwise, get the COM of the largest structure (in contour_set_a)
                cut = get_com(contour_dict_a[s_select[np.argmax(s_vol)]])

        else:
            # the user has specified a structure
            # first, check the structure isn't empty!
            if sitk.GetArrayFromImage(contour_dict_a[structure_for_com]).sum() != 0:
                cut = get_com(contour_dict_a[structure_for_com])
            # if it is, try the same structure in contour_set_b
            elif sitk.GetArrayFromImage(contour_dict_b[structure_for_com]).sum() != 0:
                cut = get_com(contour_dict_b[structure_for_com])

        img_vis_kw["cut"] = cut

    # Colormap options
    if isinstance(contour_cmap, (mcolors.ListedColormap, mcolors.LinearSegmentedColormap)):
        colors_a = {
            s + "a": c
            for s, c in zip(
                s_select, plt.cm.get_cmap(contour_cmap)(np.linspace(0, 1, len(s_select)))
            )
        }

        colors_b = {
            s + "b": c
            for s, c in zip(
                s_select, plt.cm.get_cmap(contour_cmap)(np.linspace(0, 1, len(s_select)))
            )
        }

    elif isinstance(contour_cmap, dict):
        colors_a = {s + "a": contour_cmap[s] for s in s_select}
        colors_b = {s + "b": contour_cmap[s] for s in s_select}

    # Visualise!
    vis = ImageVisualiser(img, **img_vis_kw)

    # Add contour set A
    vis.add_contour(
        {s + "a": contour_dict_a[s] for s in s_select},
        show_legend=False,
        color=colors_a,
    )

    # Add contour set B
    vis.add_contour(
        {s + "b": contour_dict_b[s] for s in s_select},
        show_legend=False,
        color=colors_b,
        linestyle="dashed",
    )

    # Optionally, set limits
    if structure_for_limits is not None:
        vis.set_limits_from_label(contour_dict_a[structure_for_limits], expansion=20)

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
    df_metrics = pd.DataFrame(
        columns=["STRUCTURE", "DSC", "MDA_mm", "HD_mm", "VOL_A_cm3", "VOL_B_cm3"]
    )
    columns = ("DSC", "MDA\n[mm]", "HD\n[mm]", "Vol.\nRatio")

    cell_text = []

    for s, row in zip(s_select, rows):
        dsc = compute_metric_dsc(contour_dict_a[s], contour_dict_b[s])
        mda = compute_metric_masd(contour_dict_a[s], contour_dict_b[s])
        hd = compute_metric_hd(contour_dict_a[s], contour_dict_b[s])
        vol_a = compute_volume(contour_dict_a[s])
        vol_b = compute_volume(contour_dict_b[s])

        cell_text.append(
            [
                f"{dsc:.2f}",
                f"{mda:.2f}",
                f"{hd:.2f}",
                f"{vol_b/vol_a:.2f}",
            ]
        )

        # compute metrics and add to dataframe
        df_metrics = df_metrics.append(
            {
                "STRUCTURE": s,
                "DSC": dsc,
                "MDA_mm": mda,
                "HD_mm": hd,
                "VOL_A_cm3": vol_a,
                "VOL_B_cm3": vol_b,
            },
            ignore_index=True,
        )

    # If there are no labels we can make the table bigger
    if title == "" and subsubtitle == "" and subsubtitle == "":
        v_extent = 0.88
    else:
        v_extent = 0.7

    v_extent = min([v_extent, 0.1 * len(list(contour_dict_a.keys()))])

    # Create the table
    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        rowColours=list(colors_a.values()),
        colLabels=columns,
        fontsize=10,
        bbox=[0.25, 0.1, 0.73, v_extent],
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

        table[row, 0].set_width(0.1)
        table[row, 1].set_width(0.1)
        table[row, 2].set_width(0.1)
        table[row, 3].set_width(0.1)
        if row > 0:
            table[row, -1].set_width(0)

    for col in range(len(columns)):
        table[0, col].set_height(0.075)

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
    ax.text(0.95, 0.92, subtitle, color="darkgreen", ha="right", va="top", size=fs + 2)
    ax.text(0.95, 0.87, subsubtitle, color="k", ha="right", va="top", size=fs + 2)

    # Insert legend
    _solid = mlines.Line2D([], [], color="k", label=contour_label_a)
    _dashed = mlines.Line2D([], [], color="k", linestyle="dashed", label=contour_label_b)
    ax.legend(
        handles=[_solid, _dashed],
        bbox_to_anchor=(0.25, 0.02, 0.73, 0.1),
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
        fontsize=fs,
        loc="lower left",
    )

    # Return the figure
    return fig, df_metrics

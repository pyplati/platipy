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


import matplotlib.pyplot as plt
import numpy as np

from platipy.imaging.label.fusion import process_probability_image
from platipy.imaging.utils.crop import label_to_roi, crop_to_roi


def gen_primes():
    """Generate an infinite sequence of prime numbers."""
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}

    # The running integer that's checked for primeness
    q = 2

    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            #
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            #
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]

        q += 1


def quick_optimise_probability(
    metric_function,
    manual_contour,
    probability_image,
    p_0=0.5,
    delta=0.5,
    tolerance=0.01,
    mode="min",
    create_figure=False,
    auto_crop=True,
    metric_args={},
):
    """Optimise the probability threshold used to generate a binary segmentation.
    This is a simple parameter sweep, with linearly decreasing resolution. It will usually converge
    between 5 and 10 iterations.

    Args:
        metric_function (function to return float): The metric function, this takes in two binary
            masks (a reference, and test [SimpleITK.Image]) and returns a metric (as a float).
            Typical choices would be the DSC, a surface distance metric, dose difference,
            relative volume. Additional arguments can be passed in through `metric_args` argument.
        manual_contour (SimpleITK.Image): The reference (manual) contour.
        probability_image (SimpleITK.Image): The probability map from which the optimal threshold
            will be derived. This does NOT have to be scaled to [0,1].
        p_0 (float, optional): Initial guess of the optimal threshold. Defaults to 0.5.
        delta (float, optional): The window size of the optimiser. Defaults to 0.5.
        tolerance (float, optional): If the metric changes by an amount less that `tolerance`
            the optimiser will stop. Defaults to 0.01.
        mode (str, optional): Specifies whether the metric should be maximised ("max") or minimised
            ("min"). Defaults to "min".
        create_figure (bool, optional): Create a matplotlib figure showing the optimisation. This
            is not returned, so make sure you can capture this (e.g. using IPython).
            Defaults to False.
        auto_crop (bool, optional): Crop the image volumes to the region of interest. Speeds up the
            process signficiantly so don't turn off unless you have a good reason to!
            Defaults to True.
        metric_args (dict, optional): Additional arguments passes to the metric function. This
            could be useful if you are calculating a dose-based metric and require a dose grid to
            be passed to the metric function. Defaults to {}.

    Returns:
        tuple (float, float): The optimal probability, optimal metric value.
    """
    # Auto crop images
    if auto_crop:
        cb_size, cb_index = label_to_roi(
            (manual_contour > 0) | (probability_image > 0), expansion_mm=10
        )
        manual_contour = crop_to_roi(manual_contour, cb_size, cb_index)
        probability_image = crop_to_roi(probability_image, cb_size, cb_index)

    # Set up
    n_iter = 0
    p_best = p_0

    auto_contour = process_probability_image(probability_image, threshold=p_0)

    m_n = metric_function(manual_contour, auto_contour, **metric_args)
    m_best = m_n

    print("Starting optimisation.")
    print(f"n = 0 | p = {p_best:.3f} | metric = {m_n:.3f}")

    p_list = [p_best]
    m_list = [m_best]

    improv = 0

    while np.abs(improv) > tolerance or n_iter <= 3:

        n_iter += 1
        m_n = m_best

        p_new = [
            p_best - 3 * delta / 4,
            p_best - delta / 2,
            p_best - delta / 4,
            p_best + delta / 4,
            p_best + delta / 2,
            p_best + 3 * delta / 4,
        ]
        m_new = [
            metric_function(
                manual_contour,
                process_probability_image(probability_image, threshold=p),
                **metric_args,
            )
            for p in p_new
        ]

        p_list = p_list + p_new
        m_list = m_list + m_new

        if mode == "min":
            p_best = p_list[np.argmin(m_list)]
            m_best = np.min(m_list)
        elif mode == "max":
            p_best = p_list[np.argmax(m_list)]
            m_best = np.max(m_list)

        improv = m_best - m_n

        delta /= 4

        print(f"n = {n_iter} | p = {p_best:.3f} | metric = {m_best:.3f}")

    if create_figure:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(p_list, m_list, c="k", zorder=1)
        ax.plot(*list(zip(*sorted(zip(p_list, m_list)))), c="k", zorder=1)
        ax.scatter(
            (p_best), (m_best), c="r", label=f"Optimum ({p_best:.2f},{m_best:.2f})", zorder=2
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability Difference (from Optimal)")
        ax.set_ylabel("Metric Value")
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_title(f"Optimiser | {metric_function.__name__}, mode = {mode}")
        ax.legend()
        fig.show()

    return p_best, m_best

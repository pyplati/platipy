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

import numpy as np
import SimpleITK as sitk


def process_probability_image(probability_image, threshold=0.5):
    """
    Generate a mask given a probability image, performing some basic post processing as well.
    """

    # Check type
    if not isinstance(probability_image, sitk.Image):
        probability_image = sitk.GetImageFromArray(probability_image)

    # Normalise probability map
    probability_image = probability_image / sitk.GetArrayFromImage(probability_image).max()

    # Get the starting binary image
    binary_image = sitk.BinaryThreshold(probability_image, lowerThreshold=threshold)

    # Fill holes
    binary_image = sitk.BinaryFillhole(binary_image)

    # Apply the connected component filter
    labelled_image = sitk.ConnectedComponent(binary_image)

    # Measure the size of each connected component
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labelled_image)
    label_indices = label_shape_filter.GetLabels()
    voxel_counts = [label_shape_filter.GetNumberOfPixels(i) for i in label_indices]
    if voxel_counts == []:
        return binary_image

    # Select the largest region
    largest_component_label = label_indices[np.argmax(voxel_counts)]
    largest_component_image = labelled_image == largest_component_label

    return sitk.Cast(largest_component_image, sitk.sitkUInt8)

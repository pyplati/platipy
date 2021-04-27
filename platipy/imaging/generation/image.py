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


def insert_sphere(arr, sp_radius=4, sp_centre=(0, 0, 0)):
    """Insert a sphere into the give array

    Args:
        arr (np.array): Array in which to insert sphere
        sp_radius (int, optional): The radius of the sphere. Defaults to 4.
        sp_centre (tuple, optional): The position at which the sphere should be inserted. Defaults
                                     to (0, 0, 0).

    Returns:
        np.array: An array with the sphere inserted
    """

    arr_copy = arr[:]

    x, y, z = np.indices(arr.shape)

    arr_copy[
        (x - sp_centre[0]) ** 2 + (y - sp_centre[1]) ** 2 + (z - sp_centre[2]) ** 2
        <= sp_radius ** 2
    ] = 1

    return arr_copy

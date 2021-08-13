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

import SimpleITK as sitk
import numpy as np

from scipy.ndimage.measurements import center_of_mass

from platipy.imaging.utils.math import gen_primes


def correct_volume_overlap(binary_label_dict):
    """
    Label structures by primes
    Smallest prime = largest volume
    """

    # Calculate volume
    f_vol = lambda x: sitk.GetArrayViewFromImage(x).sum()
    volume_dict = {i: f_vol(binary_label_dict[i]) for i in binary_label_dict.keys()}

    keys, vals = zip(*volume_dict.items())
    volume_rank = np.argsort(vals)[::-1]

    # print(keys, volume_rank)

    ranked_names = np.array(keys)[volume_rank]

    # Get overlap using prime factors
    prime_labelled_image = sum(binary_label_dict.values()) > 0
    combined_label = sum(binary_label_dict.values()) > 0

    for p, label in zip(gen_primes(), ranked_names):
        prime_labelled_image = prime_labelled_image * (
            (p - 1) * binary_label_dict[label] + combined_label
        )

    output_label_dict = {}
    for p, label in zip(gen_primes(), ranked_names):
        output_label_dict[label] = combined_label * (sitk.Modulus(prime_labelled_image, p) == 0)

        combined_label = sitk.Mask(combined_label, output_label_dict[label] == 0)

    return output_label_dict


def get_com(label, as_int=True, real_coords=False):
    """
        Get centre of mass of a SimpleITK.Image

    Args:
        label (sitk.Image): Label mask image.
        as_int (bool, optional): Returns each components as int if true. Defaults to True.
        real_coords (bool, optional): Return coordinates in physical space if true. Defaults to
            False.

    Returns:
        list: List of coordinates
    """
    arr = sitk.GetArrayFromImage(label)
    com = center_of_mass(arr)

    if real_coords:
        com = label.TransformContinuousIndexToPhysicalPoint(com[::-1])

    else:
        if as_int:
            com = [int(i) for i in com]

    return com


def vectorised_transform_index_to_physical_point(image, point_array, rotate=True):
    """
    Transforms a set of points from array indices to real-space
    """
    if rotate:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return point_array * spacing + origin


def vectorised_transform_physical_point_to_index(image, point_array, rotate=True):
    """
    Transforms a set of points from real-space to array indices
    """
    if rotate:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return (point_array - origin) / spacing


def generate_primes():
    """Generate an infinite sequence of prime numbers."""
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    d = {}

    # The running integer that's checked for primeness
    q = 2

    while True:
        if q not in d:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            #
            yield q
            d[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            #
            for p in d[q]:
                d.setdefault(p + q, []).append(p)
            del d[q]

        q += 1


def prime_encode_structure_list(structure_list):
    """Encode a list of structures using prime labels

    Args:
        structure_list (list [SimpleITK.Image]): A list of binary masks.

    Returns:
        SimpleITK.Image: The prime-encoded structure
    """

    # Create an image with all ones
    img_size = structure_list[0].GetSize()
    prime_encoded_image = sitk.GetImageFromArray(np.ones(img_size[::-1]))
    prime_encoded_image = sitk.Cast(prime_encoded_image, sitk.sitkUInt64)

    prime_generator = generate_primes()

    for s_img, prime in zip(structure_list, prime_generator):
        # Cast to int
        s_img_int = sitk.Cast(s_img > 0, sitk.sitkUInt64)

        print(prime)
        # Multiply with the encoded image
        prime_encoded_image = (
            sitk.MaskNegated(prime_encoded_image, s_img_int)
            + sitk.Mask(prime_encoded_image, s_img_int) * prime * s_img_int
        )

    return prime_encoded_image


def prime_decode_image(prime_encoded_image):
    """Decode a prime-encoded image

    Args:
        prime_encoded_image (SimpleITK.Image): A prime-encoded image.

    Returns:
        list [SimpleITK.Image]: A list of binary masks.
    """

    prime_generator = generate_primes()

    structure_list = []
    num_nonzero_voxels = 1

    for prime in prime_generator:
        # Get the next prime
        print(prime)

        # Calculate the region originally defined with this prime
        s_img = sitk.Equal(sitk.Modulus(prime_encoded_image, prime), 0)

        # Check how many voxels we have
        num_nonzero_voxels = sitk.GetArrayViewFromImage(s_img).sum()

        if num_nonzero_voxels > 0:
            structure_list.append(s_img)
        else:
            break

    return structure_list


def binary_encode_structure_list(structure_list):
    """Encode a list of binary labels using binary encoding

    Args:
        structure_list (list (SimpleITK.Image)): The list of binary label maps.

    Raises:
        ValueError: A maximum of 32 structures can be encoded!

    Returns:
        SimpleITK.Image: The encoded image, can be saved etc. as usual.
    """

    if len(structure_list) > 32:
        raise ValueError("You can only encode a maximum of 32 structures with this method!")

    # Create an image with all zeros
    img_size = structure_list[0].GetSize()
    binary_encoded_arr = np.zeros(img_size[::-1]).astype(int)

    for power, s_img in enumerate(structure_list):
        # Convert image to array
        s_arr = sitk.GetArrayFromImage(s_img).astype(int)
        # Bitwise-or with existing array
        binary_encoded_arr = np.bitwise_or(
            binary_encoded_arr, s_arr.astype(bool) * 2 ** (power + 1)
        )

    binary_encoded_img = sitk.GetImageFromArray(binary_encoded_arr)
    binary_encoded_img.CopyInformation(structure_list[0])

    # Image can be cast to 8,16, or 32 bit integer (depending on number of structures)
    # Safest is just to use 32 bits
    binary_encoded_img = sitk.Cast(binary_encoded_img, sitk.sitkUInt32)

    return binary_encoded_img


def binary_decode_image(binary_encoded_img):
    """Decode a binary label map to a list of structures.

    Args:
        binary_encoded_img (SimpleITK.Image): The encoded image.

    Returns:
        list (SimpleITK.Image): The list of images.
    """

    binary_encoded_arr = sitk.GetArrayFromImage(binary_encoded_img).astype(int)

    structure_list = []
    num_nonzero_voxels = 1

    for power in range(32):

        # Calculate the region originally defined with this prime
        s_arr = np.bitwise_and(binary_encoded_arr, 2 ** (power + 1))

        # Check how many voxels we have
        num_nonzero_voxels = s_arr.sum()

        if num_nonzero_voxels > 0:
            s_img = sitk.GetImageFromArray(s_arr) > 0
            s_img.CopyInformation(binary_encoded_img)
            structure_list.append(s_img)
        else:
            continue

    return structure_list

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

import logging
from pathlib import Path

import pydicom
import numpy as np
import SimpleITK as sitk

from skimage.draw import polygon


logger = logging.getLogger(__name__)


def read_dicom_image(dicom_path, return_slice_z_positions=False):
    """Read a DICOM image series

    Args:
        dicom_path (str|pathlib.Path): Path to the DICOM series to read
        return_slice_z_positions (bool, optional): When True, also return a
            ``numpy.ndarray`` of per-slice ``ImagePositionPatient[2]``
            values in the same order ITK used to stack the series. This
            array is consumed by
            :func:`transform_point_set_from_dicom_struct` to resolve each
            contour plane's Z to the nearest actual slice -- correct on
            non-uniform-Z series, where SimpleITK's ``ImageSeriesReader``
            otherwise collapses ``spacing[2]`` to an averaged value and
            breaks ``TransformPhysicalPointToIndex``. Defaults to False
            for backward compatibility.

    Returns:
        sitk.Image: The image as a SimpleITK Image. When
        ``return_slice_z_positions=True``, returns
        ``(image, slice_z_positions)``; ``slice_z_positions`` is
        ``None`` if any DICOM file is missing the IPP tag (anonymized
        series, corrupt input) so the rasterizer cleanly falls back to
        the legacy path.
    """
    dicom_images = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(dicom_path))
    image = sitk.ReadImage(dicom_images)
    if not return_slice_z_positions:
        return image
    return image, _read_slice_z_positions(dicom_images)


def _read_slice_z_positions(dicom_file_names):
    """Read per-slice ``ImagePositionPatient[2]`` in the order ITK stacked
    them.

    ``GetGDCMSeriesFileNames`` returns files in the order
    ``ImageSeriesReader`` uses to build the volume. Reading each file's
    IPP[2] in that order gives us a per-z-index Z position that the
    rasterizer can use as ground truth when mapping contour Z's to
    slice indices -- bypassing SimpleITK's averaged ``spacing[2]`` for
    non-uniform-Z series.

    Returns ``None`` if any file is missing the IPP tag.
    """
    try:
        zs = []
        for path in dicom_file_names:
            ds = pydicom.dcmread(
                path, force=True, stop_before_pixels=True,
                specific_tags=["ImagePositionPatient"],
            )
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is None or len(ipp) < 3:
                return None
            zs.append(float(ipp[2]))
    except Exception:  # pragma: no cover - defensive; falls back to legacy
        return None
    if not zs:
        return None
    return np.asarray(zs, dtype=np.float64)


def _resolve_slice_index(contour_z_mm, fallback_index, slice_z_positions):
    """Map a contour's physical Z to a slice index.

    When ``slice_z_positions`` (per-DICOM IPP[2] in the order ITK
    stacked the series) is provided, return the index of the nearest
    cached slice. This is robust against non-uniform-Z series where
    SimpleITK's ``ImageSeriesReader`` compresses per-slice positions
    into a single averaged ``spacing[2]`` and
    ``TransformPhysicalPointToIndex`` then rounds against that average
    -- causing contours on the irregular side of the gap to land on
    the wrong slice.

    When ``slice_z_positions`` is ``None`` (e.g. IPP tag missing from
    an anonymized series) the existing ``fallback_index`` is returned
    so behaviour matches the legacy path.
    """
    if slice_z_positions is None:
        return int(fallback_index)
    return int(np.argmin(np.abs(np.asarray(slice_z_positions) - float(contour_z_mm))))


def read_dicom_struct_file(filename):
    """Read a DICOM RTSTRUCT file

    Args:
        filename (str|pathlib.Path): Path to the RTSTRUCT to read

    Returns:
        pydicom.Dataset: The RTSTRUCT as a DICOM Dataset
    """
    # ``pydicom.read_file`` was deprecated in pydicom 2.x and removed in
    # pydicom 3.x; ``pydicom.dcmread`` is the cross-version name and
    # accepts the same arguments. This keeps the loader working under
    # both supported pydicom majors (``pyproject.toml`` declares
    # ``pydicom ^2.1.2`` which resolves to 2.x or 3.x depending on
    # the resolver).
    dicom_struct_file = pydicom.dcmread(filename, force=True)
    return dicom_struct_file


def fix_missing_data(contour_data):
    """Fixed a set of contour data if there are values missing

    Args:
        contour_data (pydicom.Sequence): The contour sequence from the DICOM object

    Returns:
        np.array: The array of contour data with missing values fixed
    """
    contour_data = np.array(contour_data)
    if contour_data.any() == "":
        logger.debug("Missing values detected.")
        missing_values = np.where(contour_data == "")[0]
        if missing_values.shape[0] > 1:
            logger.debug("More than one value missing, fixing this isn't implemented yet...")
        else:
            logger.debug("Only one value missing.")
            missing_index = missing_values[0]
            missing_axis = missing_index % 3
            if missing_axis == 0:
                logger.debug("Missing value in x axis: interpolating.")
                if missing_index > len(contour_data) - 3:
                    lower_value = contour_data[missing_index - 3]
                    upper_value = contour_data[0]
                elif missing_index == 0:
                    lower_value = contour_data[-3]
                    upper_value = contour_data[3]
                else:
                    lower_value = contour_data[missing_index - 3]
                    upper_value = contour_data[missing_index + 3]
                contour_data[missing_index] = 0.5 * (lower_value + upper_value)
            elif missing_axis == 1:
                logger.debug("Missing value in y axis: interpolating.")
                if missing_index > len(contour_data) - 2:
                    lower_value = contour_data[missing_index - 3]
                    upper_value = contour_data[1]
                elif missing_index == 0:
                    lower_value = contour_data[-2]
                    upper_value = contour_data[4]
                else:
                    lower_value = contour_data[missing_index - 3]
                    upper_value = contour_data[missing_index + 3]
                contour_data[missing_index] = 0.5 * (lower_value + upper_value)
            else:
                logger.debug("Missing value in z axis: taking slice value")
                temp = contour_data[2::3].tolist()
                temp.remove("")
                contour_data[missing_index] = np.min(np.array(temp, dtype=np.double))
    return contour_data


def transform_point_set_from_dicom_struct(
    dicom_image, dicom_struct, spacing_override=None, slice_z_positions=None,
):
    """Converts a set of points from a DICOM RTSTRUCT into a mask array

    Args:
        dicom_image (sitk.Image): The reference image
        dicom_struct (pydicom.Dataset): The DICOM RTSTRUCT
        spacing_override (list): The spacing to override. Defaults to None
        slice_z_positions (numpy.ndarray, optional): Per-DICOM
            ``ImagePositionPatient[2]`` array (one entry per slice, in
            the order ITK stacked the series). When supplied, each
            contour plane's Z is resolved to the nearest cached slice
            instead of going through SimpleITK's
            ``TransformPhysicalPointToIndex``. This is the only path
            that handles non-uniform-Z series (mixed 3/6 mm gaps etc.)
            correctly; on uniform-Z series both paths agree. When
            ``None`` (the default), the legacy
            ``TransformPhysicalPointToIndex`` rounding is used.

    Returns:
        tuple: Returns a list of masks and a list of structure names
    """

    if spacing_override:
        current_spacing = list(dicom_image.GetSpacing())
        new_spacing = tuple(
            [
                current_spacing[k] if spacing_override[k] == 0 else spacing_override[k]
                for k in range(3)
            ]
        )
        dicom_image.SetSpacing(new_spacing)

    struct_point_sequence = {cs.ReferencedROINumber: cs for cs in dicom_struct.ROIContourSequence}

    struct_list = []
    final_struct_name_sequence = []

    for struct_ds in dicom_struct.StructureSetROISequence:
        image_blank = np.zeros(dicom_image.GetSize()[::-1], dtype=np.uint8)

        struct_name = "_".join(struct_ds.ROIName.split())
        struct_index = struct_ds.ROINumber
        logger.debug("Converting structure %s with name: %s", struct_index, struct_name)

        if not struct_index in struct_point_sequence:
            logger.debug("No ROIContourSequence found for this structure, skipping.")
            continue

        if not hasattr(struct_point_sequence[struct_index], "ContourSequence"):
            logger.debug("No ContourSequence found for this structure, skipping.")
            continue

        if len(struct_point_sequence[struct_index].ContourSequence) == 0:
            logger.debug("Contour sequence empty for this structure, skipping.")
            continue

        if (
            not struct_point_sequence[struct_index].ContourSequence[0].ContourGeometricType
            == "CLOSED_PLANAR"
        ):
            logger.debug("This is not a closed planar structure, skipping.")
            continue

        # Track in case something goes wrong in here we will skip the contour
        skip_contour = False
        last_z_loc = None
        for sl in range(len(struct_point_sequence[struct_index].ContourSequence)):

            contour_data = fix_missing_data(
                struct_point_sequence[struct_index].ContourSequence[sl].ContourData
            )

            struct_slice_contour_data = np.array(contour_data, dtype=np.double)
            vertex_arr_physical = struct_slice_contour_data.reshape(
                struct_slice_contour_data.shape[0] // 3, 3
            )

            point_arr = np.array(
                [dicom_image.TransformPhysicalPointToIndex(i) for i in vertex_arr_physical]
            ).T

            [x_vertex_arr_image, y_vertex_arr_image] = point_arr[[0, 1]]

            # CLOSED_PLANAR contours have every vertex on a single
            # physical Z plane by construction, so we resolve the slice
            # index from that physical Z. When ``slice_z_positions`` is
            # available we use nearest-IPP lookup (correct on
            # non-uniform-Z series); otherwise we fall back to the
            # legacy rounded continuous index produced by
            # ``TransformPhysicalPointToIndex``.
            contour_z_mm = float(vertex_arr_physical[0, 2])
            z_index = _resolve_slice_index(
                contour_z_mm=contour_z_mm,
                fallback_index=point_arr[2][0],
                slice_z_positions=slice_z_positions,
            )

            # The all-vertices-share-Z sanity check operates on the
            # physical contour data so it isn't fooled by rounding
            # behaviour of the index transform on non-uniform-Z series
            # (a contour with all points at the same physical Z could
            # otherwise round to different indices and be incorrectly
            # rejected).
            if np.any(vertex_arr_physical[:, 2] != contour_z_mm):
                logger.debug("Error: axial slice index varies in contour. Skipping Contour.")
                logger.debug("Structure:   %s", struct_name)
                logger.debug("Slice index: %d", z_index)
                skip_contour = True
                break

            # Spacing-mismatch warning: only meaningful on the legacy
            # fallback path. With ``slice_z_positions`` cached, the
            # mismatch is by construction (and handled correctly by the
            # nearest-IPP lookup above), so the warning would be noise.
            if (
                slice_z_positions is None
                and last_z_loc is not None
                and np.abs(
                    np.abs(vertex_arr_physical[2][0] - last_z_loc)
                    - dicom_image.GetSpacing()[2]
                )
                > 0.01
            ):
                logger.warning(
                    "RTSTRUCT slice increment doesn't match image spacing. "
                    "Check data and override if necessary."
                )

            last_z_loc = vertex_arr_physical[2][0]

            if z_index >= dicom_image.GetSize()[2]:
                logger.debug("Warning: Slice index greater than image size. Skipping slice.")
                logger.debug("Structure:   %s", struct_name)
                logger.debug("Slice index: %d", z_index)
                continue

            if z_index < 0:
                logger.debug("Warning: Slice index less than zero. Skipping slice.")
                logger.debug("Structure:   %s", struct_name)
                logger.debug("Slice index: %d", z_index)
                continue

            slice_arr = np.zeros(image_blank.shape[-2:], dtype=np.uint8)

            filled_indices_x, filled_indices_y = polygon(
                x_vertex_arr_image, y_vertex_arr_image, shape=slice_arr.shape
            )
            slice_arr[filled_indices_y, filled_indices_x] = 1

            image_blank[z_index] ^= slice_arr

        if not skip_contour:
            struct_image = sitk.GetImageFromArray(1 * (image_blank > 0))
            struct_image.CopyInformation(dicom_image)
            struct_list.append(sitk.Cast(struct_image, sitk.sitkUInt8))
            final_struct_name_sequence.append(struct_name)

    return struct_list, final_struct_name_sequence


def convert_rtstruct(
    dcm_img,
    dcm_rt_file,
    prefix="Struct_",
    output_dir=".",
    output_img=None,
    spacing=None,
    replace_slashes_with="",
):
    """Convert a DICOM RTSTRUCT to NIFTI masks.

    The masks are stored as NIFTI files in the output directory

    Args:
        dcm_img (str|pathlib.Path): Path to the reference DICOM image series
        dcm_rt_file (str|pathlib.Path): Path to the DICOM RTSTRUCT file
        prefix (str, optional): The prefix to give the output files. Defaults to "Struct_".
        output_dir (str|pathlib.Path, optional): Path to the output directory. Defaults to ".".
        output_img (str|pathlib.Path, optional): If set, write the reference image to this file as
                                                 in NIFTI format. Defaults to None.
        spacing (list, optional): Values of image spacing to override. Defaults to None.
        replace_slashes_with (str, optional): String to replace "/" and "\" with. Set to None
            disable replacement of slashes. Defaults to "".
    """

    logger.debug("Converting RTStruct: %s", dcm_rt_file)
    logger.debug("Using image series: %s", dcm_img)
    logger.debug("Output file prefix: %s", prefix)
    logger.debug("Output directory: %s", output_dir)

    # ``slice_z_positions`` is the per-DICOM ImagePositionPatient[2]
    # array in the order ITK stacked the series. Threaded through to
    # the rasterizer so contour Z's get resolved to slice indices via
    # nearest-IPP rather than SimpleITK's averaged ``spacing[2]``, which
    # on non-uniform-Z series mis-routes contours on the irregular
    # side of the gap.
    dicom_image, slice_z_positions = read_dicom_image(
        dcm_img, return_slice_z_positions=True,
    )
    dicom_struct = read_dicom_struct_file(dcm_rt_file)

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    image_output_path = None
    if output_img is not None:

        if not isinstance(output_img, Path):
            if not output_img.endswith(".nii.gz"):
                output_img = f"{output_img}.nii.gz"
            output_img = output_dir.joinpath(output_img)

        image_output_path = output_img

        logger.debug("Image series to be converted to: %s", image_output_path)

    if spacing:

        if isinstance(spacing, str):
            spacing = [float(i) for i in spacing.split(",")]

        logger.debug("Overriding image spacing with: %s", spacing)

    struct_list, struct_name_sequence = transform_point_set_from_dicom_struct(
        dicom_image, dicom_struct, spacing, slice_z_positions=slice_z_positions,
    )
    logger.debug("Converted all structures. Writing output.")
    for struct_index, struct_image in enumerate(struct_list):
        struct_name = struct_name_sequence[struct_index]

        if replace_slashes_with is not None:
            struct_name = struct_name.replace("/", replace_slashes_with)
            struct_name = struct_name.replace("\\", replace_slashes_with)

        out_name = f"{prefix}{struct_name}.nii.gz"
        out_name = output_dir.joinpath(out_name)

        logger.debug("Writing file to: %s", output_dir)
        sitk.WriteImage(struct_image, str(out_name))

    if image_output_path is not None:
        sitk.WriteImage(dicom_image, str(image_output_path))

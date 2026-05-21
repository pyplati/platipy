# Copyright 2026 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression test for the non-uniform-Z slice-matching fix.

Background
==========

Clinical CT acquisitions sometimes have **non-uniform Z spacing** --
e.g. mixed 3 mm and 6 mm slice gaps. This is common on the
NSCLC-Radiomics public cohort (LUNG1-014/-021/-085/-095/-194/-246 all
have it).

The legacy ``transform_point_set_from_dicom_struct`` mapped every
contour point's physical coordinate to a voxel index via SimpleITK's
``TransformPhysicalPointToIndex``. On non-uniform-Z series, ITK's
``ImageSeriesReader`` compresses the per-slice positions into a
single *averaged* ``spacing[2]``; the per-point index then rounds
against that averaged spacing and misses contour planes that sit on
the irregular side of the gap.

The fix caches the per-DICOM ``ImagePositionPatient[2]`` array via
``read_dicom_image(return_slice_z_positions=True)`` and resolves each
contour plane's Z to the nearest actual slice instead, falling back
to the legacy rounded continuous index when the cached array is
unavailable.

This test exercises four layers:

1. ``read_dicom_image(return_slice_z_positions=True)`` returns the
   expected per-slice Z array on a uniform-Z synthetic dataset
   (no regression on the most-common case).
2. The same call on a non-uniform-Z synthetic dataset returns the
   per-DICOM IPP[2] values exactly -- including the 6 mm jumps.
3. End-to-end ``convert_rtstruct`` on a non-uniform-Z synthetic
   series produces a mask non-empty on every contour-bearing slice.
4. ``_resolve_slice_index`` falls back to ``int(fallback_index)``
   when the cached array is ``None`` (anonymized series, etc.).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from platipy.dicom.io.rtstruct_to_nifti import (
    _resolve_slice_index,
    convert_rtstruct,
    read_dicom_image,
)
from platipy.dicom.tests import synthetic_rt


# ---------------------------------------------------------------------------
# Layer 1 -- uniform-Z synthetic dataset (regression: fix is a no-op here)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def uniform_dataset(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("uniform_z")
    z_positions = [float(k) for k in range(12)]
    uids, sop_uids = synthetic_rt.build_ct_series(out / "CT", z_positions)
    synthetic_rt.build_rtstruct(
        out / "RT.dcm",
        uids=uids,
        sop_uids=sop_uids,
        z_positions=z_positions,
    )
    return {
        "ct_dir": out / "CT",
        "rt_path": out / "RT.dcm",
        "expected_zs": np.asarray(z_positions, dtype=np.float64),
    }


def test_per_slice_z_populated_on_uniform_dataset(uniform_dataset):
    """``read_dicom_image(return_slice_z_positions=True)`` returns the
    per-DICOM IPP[2] array matching the input Zs on a uniform-Z
    dataset. A regression in the ingest path would leave the array
    as None (the fix would silently degrade to the legacy round path)
    or populate it with the wrong values; both are caught here.
    """
    img, zs = read_dicom_image(
        str(uniform_dataset["ct_dir"]), return_slice_z_positions=True,
    )
    assert zs is not None
    np.testing.assert_allclose(zs, uniform_dataset["expected_zs"], atol=1e-6)

    # Legacy single-return signature must still work.
    img2 = read_dicom_image(str(uniform_dataset["ct_dir"]))
    assert isinstance(img2, sitk.Image)


# ---------------------------------------------------------------------------
# Layer 2 -- non-uniform-Z CT: per-slice Z array tracks the mixed gaps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nonuniform_dataset(tmp_path_factory) -> dict:
    """Synthetic CT with mixed 3/6 mm Z gaps + RTSTRUCT whose
    per-contour Z values are aligned to the new IPPs."""
    out = tmp_path_factory.mktemp("nonuniform_z")
    gaps = [3.0, 3.0, 3.0, 6.0, 6.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    z_positions = [0.0]
    for g in gaps:
        z_positions.append(z_positions[-1] + g)
    assert len(z_positions) == 12

    uids, sop_uids = synthetic_rt.build_ct_series(out / "CT", z_positions)
    synthetic_rt.build_rtstruct(
        out / "RT.dcm",
        uids=uids,
        sop_uids=sop_uids,
        z_positions=z_positions,
    )
    return {
        "ct_dir": out / "CT",
        "rt_path": out / "RT.dcm",
        "expected_zs": np.asarray(z_positions, dtype=np.float64),
    }


def test_per_slice_z_tracks_nonuniform_gaps(nonuniform_dataset):
    """A non-uniform-Z CT must round-trip the per-DICOM IPP[2] values
    exactly through ``read_dicom_image``. Failure here indicates the
    ingest path silently smooths the per-slice info, which is the
    underlying problem the fix exists to solve.
    """
    img, zs = read_dicom_image(
        str(nonuniform_dataset["ct_dir"]), return_slice_z_positions=True,
    )
    assert zs is not None
    np.testing.assert_allclose(
        zs, nonuniform_dataset["expected_zs"], atol=1e-6,
        err_msg="cached per-slice Z array does not match per-DICOM IPP[2]",
    )

    # SimpleITK's averaged spacing[2] must NOT match any individual gap --
    # if it did, the fixture isn't exercising the non-uniform case.
    avg_spacing_z = img.GetSpacing()[2]
    assert avg_spacing_z != pytest.approx(3.0)
    assert avg_spacing_z != pytest.approx(6.0)


def test_convert_rtstruct_end_to_end_on_nonuniform_z(nonuniform_dataset, tmp_path):
    """End-to-end: ``convert_rtstruct`` on a non-uniform-Z synthetic
    series should produce a mask with a non-zero square on every CT
    slice (since the synthetic RTSTRUCT has one contour per slice with
    Z matching the per-slice IPP[2]).

    Pre-fix, contours on the 6 mm-gap slices land on the wrong
    z-index and the resulting mask has both gaps (slices with no
    contour despite the RTSTRUCT defining one) and duplicates (the
    misrouted contour landing on a neighbour's slice).
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    convert_rtstruct(
        dcm_img=str(nonuniform_dataset["ct_dir"]),
        dcm_rt_file=str(nonuniform_dataset["rt_path"]),
        prefix="Test_",
        output_dir=out_dir,
        output_img=None,
    )

    mask_paths = sorted(out_dir.glob("Test_*.nii.gz"))
    assert len(mask_paths) == 1, f"expected one mask file, got {mask_paths}"

    mask_img = sitk.ReadImage(str(mask_paths[0]))
    mask_arr = sitk.GetArrayFromImage(mask_img)
    nonzero_slices = sorted(
        k for k in range(mask_arr.shape[0]) if mask_arr[k].any()
    )
    assert nonzero_slices == list(range(mask_arr.shape[0])), (
        "every slice should contain the synthetic square; missing slices "
        "indicate the non-uniform-Z fix did not route contours correctly. "
        f"nonzero_slices={nonzero_slices}, total={mask_arr.shape[0]}"
    )


# ---------------------------------------------------------------------------
# Layer 3 -- _resolve_slice_index uses the array when available
# ---------------------------------------------------------------------------


def test_resolve_slice_index_uses_nearest_ipp_when_available():
    """When ``slice_z_positions`` is provided, ``_resolve_slice_index``
    must pick the index of the nearest cached Z -- regardless of the
    fallback continuous index, which would normally come from
    SimpleITK's averaged-spacing transform.
    """
    slice_zs = np.asarray(
        [0.0, 3.0, 6.0, 9.0, 15.0, 21.0, 24.0, 27.0, 30.0],
        dtype=np.float64,
    )
    # Contour at z=14.7 mm -- nearest cached slice is index 4 (z=15.0).
    target_z = 14.7
    expected_idx = int(np.argmin(np.abs(slice_zs - target_z)))
    assert expected_idx == 4

    # ``fallback_index`` is intentionally a value that disagrees; the
    # cached array must win.
    got = _resolve_slice_index(
        contour_z_mm=target_z,
        fallback_index=3,
        slice_z_positions=slice_zs,
    )
    assert got == expected_idx, (
        f"expected nearest-IPP index {expected_idx}; got {got}. "
        "cached array was not consulted -- regression in the fix."
    )


# ---------------------------------------------------------------------------
# Layer 4 -- _resolve_slice_index falls back when array missing
# ---------------------------------------------------------------------------


def test_resolve_slice_index_falls_back_when_array_missing():
    """When ``slice_z_positions`` is ``None`` (e.g. anonymized series
    with stripped IPP tags), ``_resolve_slice_index`` must return the
    ``fallback_index`` argument unchanged. Without this fallback the
    fix would crash on legitimate but unusual inputs.
    """
    for fallback in [0, 1, 5, 7, 42]:
        got = _resolve_slice_index(
            contour_z_mm=12345.0,  # irrelevant when array is None
            fallback_index=fallback,
            slice_z_positions=None,
        )
        assert got == fallback, (
            f"fallback path broken: fallback={fallback} -> {got}"
        )

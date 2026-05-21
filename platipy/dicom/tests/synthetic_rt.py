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

"""Synthetic CT + RTSTRUCT generator for self-contained pytest fixtures.

The existing :mod:`platipy.dicom.tests.test_convert` downloads real
LCTSC series via :func:`platipy.imaging.tests.data.get_lung_dicom`.
That path is network-dependent and isn't suitable for testing
non-uniform-Z slice matching, which needs precise control over each
slice's ``ImagePositionPatient[2]``. This helper fills the gap with a
~zero-dependency builder that emits a CT series and matching RTSTRUCT
directly through pydicom.

Geometry is intentionally minimal -- 16x16 axial slices, single square
contour per slice -- so the resulting masks are easy to reason about
while still exercising the full conversion path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTStructureSetStorage,
    generate_uid,
)


def _save(ds: FileDataset, path: str) -> None:
    """Save a dataset cleanly on both pydicom 2.x and 3.x.

    pydicom 3.x added ``little_endian`` / ``implicit_vr`` kwargs to
    ``save_as`` and deprecated the ``is_little_endian`` /
    ``is_implicit_VR`` attributes; pydicom 2.x requires the attributes
    and rejects the kwargs. Try the 3.x kwargs first and fall back to
    attribute setting on TypeError so the helper works across the
    version range declared by ``pyproject.toml`` (``pydicom ^2.1.2``).
    """
    try:
        ds.save_as(path, enforce_file_format=True, little_endian=True, implicit_vr=False)
    except TypeError:
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(path, write_like_original=False)


@dataclass(frozen=True)
class CTSeriesUIDs:
    study: str
    series: str
    frame_of_reference: str


def make_uids() -> CTSeriesUIDs:
    return CTSeriesUIDs(
        study=generate_uid(),
        series=generate_uid(),
        frame_of_reference=generate_uid(),
    )


def build_ct_series(
    out_dir: Path,
    z_positions: Sequence[float],
    *,
    uids: CTSeriesUIDs | None = None,
    rows: int = 16,
    cols: int = 16,
    pixel_spacing_mm: float = 1.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> tuple[CTSeriesUIDs, list[str]]:
    """Write a synthetic CT series with the given per-slice Z positions.

    Returns the UIDs and the per-slice SOPInstanceUIDs in the order
    written (matches Z-ascending order). Pixel data is a constant zero
    plane -- only the geometry tags matter.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if uids is None:
        uids = make_uids()

    sop_uids: list[str] = []
    pixel_array = np.zeros((rows, cols), dtype=np.uint16)

    for k, z in enumerate(z_positions):
        sop_uid = generate_uid()
        sop_uids.append(sop_uid)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(
            filename_or_obj=str(out_dir / f"slice_{k:04d}.dcm"),
            dataset={},
            file_meta=file_meta,
            preamble=b"\x00" * 128,
        )

        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = uids.study
        ds.SeriesInstanceUID = uids.series
        ds.FrameOfReferenceUID = uids.frame_of_reference

        ds.PatientID = "SYN-001"
        ds.PatientName = "Synthetic^Phantom"
        ds.PatientBirthDate = "19700101"
        ds.PatientSex = "O"
        ds.Modality = "CT"
        ds.StudyID = "1"
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1

        ds.ImagePositionPatient = [float(origin_x), float(origin_y), float(z)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.SliceLocation = float(z)
        ds.PixelSpacing = [float(pixel_spacing_mm), float(pixel_spacing_mm)]
        ds.SliceThickness = 1.0
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0
        ds.PixelData = pixel_array.tobytes()

        _save(ds, str(out_dir / f"slice_{k:04d}.dcm"))

    return uids, sop_uids


def build_rtstruct(
    out_path: Path,
    *,
    uids: CTSeriesUIDs,
    sop_uids: Sequence[str],
    z_positions: Sequence[float],
    square_center_xy: tuple[float, float] = (8.0, 8.0),
    square_half_size_mm: float = 4.0,
    roi_name: str = "TestSquare",
    roi_number: int = 1,
) -> None:
    """Write an RTSTRUCT with a single ROI: one CLOSED_PLANAR square
    contour per CT slice. The square is constant in (x,y); only the Z
    varies. Each contour's ``ReferencedSOPInstanceUID`` points at the
    matching CT slice.
    """
    if len(sop_uids) != len(z_positions):
        raise ValueError("sop_uids and z_positions must be the same length")

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    rt_sop_uid = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = rt_sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj=str(out_path),
        dataset={},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )

    ds.SOPClassUID = RTStructureSetStorage
    ds.SOPInstanceUID = rt_sop_uid
    ds.StudyInstanceUID = uids.study
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = uids.frame_of_reference

    ds.PatientID = "SYN-001"
    ds.PatientName = "Synthetic^Phantom"
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "O"
    ds.Modality = "RTSTRUCT"
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.StructureSetLabel = "Synthetic"
    ds.StructureSetName = "Synthetic"
    ds.StructureSetDate = "19700101"
    ds.StructureSetTime = "000000"

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = uids.series
    ref_series.ContourImageSequence = DicomSequence()
    for sop in sop_uids:
        ci = Dataset()
        ci.ReferencedSOPClassUID = CTImageStorage
        ci.ReferencedSOPInstanceUID = sop
        ref_series.ContourImageSequence.append(ci)

    rt_ref_study = Dataset()
    rt_ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    rt_ref_study.ReferencedSOPInstanceUID = uids.study
    rt_ref_study.RTReferencedSeriesSequence = DicomSequence([ref_series])

    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = uids.frame_of_reference
    ref_for.RTReferencedStudySequence = DicomSequence([rt_ref_study])
    ds.ReferencedFrameOfReferenceSequence = DicomSequence([ref_for])

    ssroi = Dataset()
    ssroi.ROINumber = roi_number
    ssroi.ReferencedFrameOfReferenceUID = uids.frame_of_reference
    ssroi.ROIName = roi_name
    ssroi.ROIGenerationAlgorithm = "MANUAL"
    ds.StructureSetROISequence = DicomSequence([ssroi])

    cx, cy = square_center_xy
    h = square_half_size_mm
    contour_seq = DicomSequence()
    for sop, z in zip(sop_uids, z_positions):
        verts = [
            (cx - h, cy - h, float(z)),
            (cx + h, cy - h, float(z)),
            (cx + h, cy + h, float(z)),
            (cx - h, cy + h, float(z)),
        ]
        flat = [v for tri in verts for v in tri]
        contour = Dataset()
        contour.ContourGeometricType = "CLOSED_PLANAR"
        contour.NumberOfContourPoints = len(verts)
        contour.ContourData = flat
        ci = Dataset()
        ci.ReferencedSOPClassUID = CTImageStorage
        ci.ReferencedSOPInstanceUID = sop
        contour.ContourImageSequence = DicomSequence([ci])
        contour_seq.append(contour)

    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = roi_number
    roi_contour.ROIDisplayColor = [255, 0, 0]
    roi_contour.ContourSequence = contour_seq
    ds.ROIContourSequence = DicomSequence([roi_contour])

    rtroi_obs = Dataset()
    rtroi_obs.ObservationNumber = 1
    rtroi_obs.ReferencedROINumber = roi_number
    rtroi_obs.RTROIInterpretedType = "ORGAN"
    rtroi_obs.ROIInterpreter = ""
    ds.RTROIObservationsSequence = DicomSequence([rtroi_obs])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save(ds, str(out_path))

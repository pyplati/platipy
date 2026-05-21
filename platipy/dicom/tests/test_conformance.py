"""Conformance test: platipy ``convert_rtstruct`` vs RTMaskConformanceTest analytical
ground truth.

Generates the RTMaskConformanceTest fixture (synthetic CT + RTSTRUCT + analytic
per-ROI NIfTI ground truth), runs platipy's ``convert_rtstruct`` to convert the
RTSTRUCT into per-ROI binary NIfTIs, and asserts each ROI passes the published
conformance thresholds (Dice, Surface DSC @ 1 mm, HD95, MSD, relative volume error).

Unlike ``test_convert.py`` (real LCTSC data) and
``test_nonuniform_z_slice_matching.py`` (focused slice-matching test), the ground
truth here is computed *analytically* (sub-voxel quadrature against the closed-form
shape definitions) -- independent of any rasterizer -- so a Dice failure here is a
real accuracy regression, not a discretization artifact.

This module is opt-in: it imports the third-party ``rtmask_conformance`` package,
which is installed via the conformance requirements file::

    poetry install --with dev
    pip install -r requirements-conformance.txt

If the package is not installed the entire module is skipped via
``pytest.importorskip``, so the default ``poetry run pytest`` run is unaffected.

Threshold overrides go in ``platipy/dicom/tests/conformance.yaml`` (set
``RTMASK_CONFORMANCE_CONFIG`` to use a different file). See
https://github.com/brianmanderson/RTMaskConformanceTest for the schema.
"""
from __future__ import annotations

import os
from pathlib import Path

import pydicom
import pytest

rtmask_conformance = pytest.importorskip(
    "rtmask_conformance",
    reason=(
        "install rtmask-conformance: poetry install --with dev && "
        "pip install -r requirements-conformance.txt"
    ),
)

from rtmask_conformance import CONFORMANCE_ROIS, generate_fixture, load_config  # noqa: E402
from rtmask_conformance.generate import GenerateOptions  # noqa: E402
from rtmask_conformance.verify import Status, evaluate_one  # noqa: E402

from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct  # noqa: E402

# n_quadrature=2 (8 sub-voxel samples) is enough to make the ground-truth masks
# stable to ~1 voxel of partial-volume disagreement on the boundary -- well below
# pass thresholds and an order of magnitude faster than the n=8 the published
# fixtures use.
_FIXTURE_QUADRATURE = 2


def _find_ct_dir_and_rtstruct(fixture_root: Path) -> tuple[Path, Path]:
    """Locate the CT series directory and RTSTRUCT file inside the generated fixture.

    The fixture layout is owned by ``rtmask_conformance.generate_fixture``; rather
    than couple to a specific subdirectory shape, we walk the tree and identify
    files by DICOM ``Modality``. CT slices share a directory by construction; the
    RTSTRUCT is a single file.
    """
    rtstruct_path: Path | None = None
    ct_dir: Path | None = None
    for dcm_path in fixture_root.rglob("*.dcm"):
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        modality = getattr(ds, "Modality", "")
        if modality == "RTSTRUCT":
            rtstruct_path = dcm_path
        elif modality == "CT" and ct_dir is None:
            ct_dir = dcm_path.parent
    if rtstruct_path is None or ct_dir is None:
        pytest.fail(
            f"fixture incomplete under {fixture_root}: "
            f"ct_dir={ct_dir}, rtstruct={rtstruct_path}"
        )
    return ct_dir, rtstruct_path


@pytest.fixture(scope="session")
def conformance_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic CT + RTSTRUCT + analytic GT NIfTIs."""
    out = tmp_path_factory.mktemp("rtmask_conformance_fixture")
    generate_fixture(out, options=GenerateOptions(n_quadrature=_FIXTURE_QUADRATURE))
    return out


@pytest.fixture(scope="session")
def platipy_predictions(
    conformance_fixture: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run platipy's ``convert_rtstruct`` against the fixture.

    ``convert_rtstruct`` writes one binary NIfTI per ROI named
    ``<prefix><roi>.nii.gz``. With ``prefix=""`` the filenames are exactly what
    the rtmask-conformance verifier expects (``<roi>.nii.gz``).
    """
    pred_dir = tmp_path_factory.mktemp("platipy_preds")
    ct_dir, rtstruct_path = _find_ct_dir_and_rtstruct(conformance_fixture)
    convert_rtstruct(
        dcm_img=ct_dir,
        dcm_rt_file=rtstruct_path,
        prefix="",
        output_dir=pred_dir,
    )
    return pred_dir


_DEFAULT_CONFIG_YAML = Path(__file__).with_name("conformance.yaml")


@pytest.fixture(scope="session")
def conformance_config():
    """Load thresholds.

    Resolution order:
      1. Explicit ``RTMASK_CONFORMANCE_CONFIG`` env var (any path).
      2. ``platipy/dicom/tests/conformance.yaml`` if it exists (platipy's calibrated
         relaxations vs the package defaults -- see that file's header).
      3. The package-shipped defaults.
    """
    config_path = os.environ.get("RTMASK_CONFORMANCE_CONFIG")
    if config_path is None and _DEFAULT_CONFIG_YAML.is_file():
        config_path = str(_DEFAULT_CONFIG_YAML)
    return load_config(config_path)


@pytest.mark.parametrize("roi", CONFORMANCE_ROIS)
def test_platipy_conformance(
    roi: str,
    conformance_fixture: Path,
    platipy_predictions: Path,
    conformance_config,
):
    """Each ROI: platipy's mask must match analytic ground truth within the
    published thresholds (Dice, Surface DSC, HD95, MSD, volume error).
    """
    pred_path = platipy_predictions / f"{roi}.nii.gz"
    gt_path = conformance_fixture / "groundtruth" / f"{roi}.nii.gz"
    assert gt_path.is_file(), f"fixture incomplete: {gt_path}"
    assert pred_path.is_file(), f"convert_rtstruct produced no mask for {roi!r}"

    result = evaluate_one(roi, pred_path, gt_path, conformance_config)

    if result.status == Status.GEOMETRY_MISMATCH:
        pytest.fail(
            f"{roi}: geometry mismatch between platipy output and ground truth: "
            f"{result.geometry_diagnostic}"
        )
    if result.status != Status.PASS:
        pytest.fail(
            f"{roi}: {result.status.value}\n"
            f"  violations: {result.violations}\n"
            f"  metrics:    {result.metrics}\n"
            f"  thresholds: {result.thresholds}"
        )

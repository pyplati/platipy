from datetime import datetime
from pathlib import Path
import pydicom

import SimpleITK as sitk


def get_suv_bw_scale_factor(dicom_file):
    # Modified from
    # https://qibawiki.rsna.org/images/6/62/SUV_vendorneutral_pseudocode_happypathonly_20180626_DAC.pdf

    ds = pydicom.read_file(dicom_file)

    if ds.Units == "CNTS":
        # Try to find the Philips private scale factor")
        return float(ds[0x7053, 0x1000].value)

    assert ds.Modality == "PT"
    assert "DECY" in ds.CorrectedImage
    assert "ATTN" in ds.CorrectedImage
    assert "START" in ds.DecayCorrection
    assert ds.Units == "BQML"

    half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)

    if "SeriesTime" in ds:
        series_date_time = ds.SeriesDate + "_" + ds.SeriesTime
    if "." in series_date_time:
        series_date_time = series_date_time[
            : -(len(series_date_time) - series_date_time.index("."))
        ]
    series_date_time = datetime.strptime(series_date_time, "%Y%m%d_%H%M%S")

    if "SeriesTime" in ds:
        start_time = (
            ds.SeriesDate
            + "_"
            + ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
        )
    if "." in start_time:
        start_time = start_time[: -(len(start_time) - start_time.index("."))]
    start_time = datetime.strptime(start_time, "%Y%m%d_%H%M%S")

    decay_time = (series_date_time - start_time).seconds
    injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    decayed_dose = injected_dose * pow(2, -decay_time / half_life)
    patient_weight = float(ds.PatientWeight)
    suv_bw_scale_factor = patient_weight * 1000 / decayed_dose

    return suv_bw_scale_factor


def convert_dicom(dcm_path, output_path):
    """Convert a DICOM image to NIFTI

    Args:
        dcm_path (str|pathlib.Path): Directory containing the DICOM series
        output_path (str|pathlib.Path): The directory to save the NIFTI to

    Returns:
        pathlib.Path: Path to the saved NIFTI file
    """

    first_dcm_file = next(Path(dcm_path).glob("*.dcm"))
    ds = pydicom.read_file(first_dcm_file)

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    output_file = output_path.joinpath(f"{ds.Modality}.{ds.SeriesInstanceUID}.nii.gz")

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    if ds.Modality == "PT":

        if ds.Units == "BQML" or ds.Units == "CNTS":
            image *= get_suv_bw_scale_factor(dicom_names[0])
        else:
            print("Unknown PET Units! Values may be incorrect")

    sitk.WriteImage(image, str(output_file))

    return output_file

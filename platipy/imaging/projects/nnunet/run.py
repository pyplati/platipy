
import os
import tempfile
from pathlib import Path
from loguru import logger

import SimpleITK as sitk

from nnunet.inference.pretrained_models.download_pretrained_model import get_available_models, download_and_install_from_url

def available_nnunet_models():

    available_models = get_available_models()
    available_models["Task400_OPEN_HEART_3d_lowres"] = {
            'description': "Whole heart model (all folds, 3d_lowres only) trained on data from"
                "TCIA (NSCLC-Radiomics & LCTSC)",
            'url': "https://zenodo.org/record/6585664/files/Task400_OPEN_HEART_3d_lowres.zip?download=1"
        }
    return available_models

NNUNET_SETTINGS_DEFAULTS = {
    "task": "Task400_OPEN_HEART_3d_lowres",
    "trainer_class_name": "nnUNetTrainerV2",
    "cascade_trainer_class_name": "nnUNetTrainerV2CascadeFullRes",
    "folds": None,
    "model": "3d_lowres",
    "lowres_segmentations": None,
    "num_threads_preprocessing": 6,
    "num_threads_nifti_save": 2,
    "disable_tta": False,
    "all_in_gpu": None,
    "disable_mixed_precision": False,
    "chk": "model_final_checkpoint"
}

def run_segmentation(img, settings=NNUNET_SETTINGS_DEFAULTS):

    if not "RESULTS_FOLDER" in os.environ:
        home = Path.home()
        platipy_dir = home.joinpath(".platipy")
        home.mkdir(exist_ok=True)
        os.environ["RESULTS_FOLDER"] = str(platipy_dir.joinpath("nnUNet_models"))

    nnunet_model_path = Path(os.environ["RESULTS_FOLDER"])

    task = settings["task"]
    model = settings["model"]

    # Check if task model is already installed
    task_path = nnunet_model_path.joinpath("nnUNet", model, task)
    print(task_path)

    if not task_path.exists():
        # Check if the task is available and install it if so
        available_models = available_nnunet_models()

        if not task in available_models:
            raise ValueError(f"{task} not available")

        task_url = available_models[task]["url"]
        logger.info("Installing Task from URL")
        download_and_install_from_url(task_url)

    # Prepare the image in a temporary directory for nnunet to run on
    input_path = Path(tempfile.mkdtemp())
    io_path = input_path.joinpath(f"{settings['task']}_0000.nii.gz")
    sitk.WriteImage(img, str(io_path))

    output_path = Path(tempfile.mkdtemp())

    model = settings["model"]
    folds = settings["folds"]
    num_threads_preprocessing = settings["num_threads_preprocessing"]
    num_threads_nifti_save = settings["num_threads_nifti_save"]
    lowres_segmentations = settings["lowres_segmentations"]
    all_in_gpu = settings["all_in_gpu"]
    disable_mixed_precision = settings["disable_mixed_precision"]
    disable_tta = settings["disable_tta"]
    trainer_class_name = settings["trainer_class_name"]
    cascade_trainer_class_name = settings["cascade_trainer_class_name"]
    mode = "normal"
    default_plans_identifier = "nnUNetPlansv2.1"
    chk = settings["chk"]

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name

    model_folder_name = task_path.joinpath(trainer + f"__{default_plans_identifier}")

    # Import in here to make sure environment is already set
    from nnunet.inference.predict import predict_from_folder

    predict_from_folder(str(model_folder_name), str(input_path), str(output_path), folds, False,
        num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations, 0, 1,
        not disable_tta, overwrite_existing=True, mode=mode, overwrite_all_in_gpu=all_in_gpu,
        mixed_precision=not disable_mixed_precision, step_size=0.5, checkpoint_name=chk)

    results = {}
    for op in output_path.glob("*.nii.gz"):

        label_map = sitk.ReadImage(str(op))
        num_labels = sitk.GetArrayViewFromImage(label_map).max()

        for l in range(num_labels):
            results[f"Struct_{l}"] = label_map == (l+1)

    os.remove(io_path)

    return results

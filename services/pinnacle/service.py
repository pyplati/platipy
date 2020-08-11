"""
Service to run Pinnacle Export.
"""
import os
import tarfile
import tempfile
import shutil
import json
import pydicom

from loguru import logger

from pymedphys.experimental.pinnacle import PinnacleExport

from platipy.framework import app, DataObject, celery

PINNACLE_EXPORT_SETTINGS_DEFAULTS = {
    "exportModalities": ["CT", "RTSTRUCT", "RTPLAN", "RTDOSE"],
    "exportSeriesUIDs": []
}


@app.register("Pinnacle Export", default_settings=PINNACLE_EXPORT_SETTINGS_DEFAULTS)
def pinnacle_export_service(data_objects, working_dir, settings):
    """
    Implements the platipy framework to provide a pinnacle tar export service
    """

    logger.info("Running Pinnacle Export")
    logger.info("Using settings: " + str(settings))

    return_objects = []
    for data_object in data_objects:
        logger.info("Running on data object: " + data_object.path)

        if not data_object.type == "FILE" or not tarfile.is_tarfile(data_object.path):
            logger.error(f"Can only process TAR file. Skipping file: {data_object.path}")
            continue

        archive_path = tempfile.mkdtemp()

        # Extract the tar archive
        tar = tarfile.open(data_object.path)
        for member in tar.getmembers():
            if not ":" in member.name:
                tar.extract(member, path=archive_path)

        # Read the path to the patient directory from the data object meta data
        pat_path = data_object.meta_data["patient_path"]
        pinn_extracted = os.path.join(archive_path, pat_path)

        pinn = PinnacleExport(pinn_extracted, None)

        # Find the plan we want to export in the list of plans
        if len(pinn.plans) == 0:
            logger.error("No Plans found for patient")
            continue

        export_plan = None
        for plan in pinn.plans:
            if (
                "plan_name" in data_object.meta_data.keys()
                and plan.plan_info["PlanName"] == data_object.meta_data["plan_name"]
            ):
                export_plan = plan
                break

            if export_plan is None:
                export_plan = plan

        # If a trial was given, try to find it and set it
        for trial in export_plan.trials:
            trial_name = trial["Name"]
            if (
                "trial" in data_object.meta_data.keys()
                and trial_name == data_object.meta_data["trial"]
            ):
                export_plan.active_trial = trial_name

        output_dir = os.path.join(working_dir, str(data_object.id))
        if os.path.exists(output_dir):
            # Just in case it was already run for this data object, lets remove all old output
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        if "CT" in settings["exportModalities"]:
            logger.info("Exporting Primary CT")
            pinn.export_image(export_plan.primary_image, export_path=output_dir)

        if "RTSTRUCT" in settings["exportModalities"]:
            logger.info("Exporting RTSTRUCT")
            pinn.export_struct(export_plan, output_dir)

        if "RTPLAN" in settings["exportModalities"]:
            logger.info("Exporting RTPLAN")
            pinn.export_plan(export_plan, output_dir)

        if "RTDOSE" in settings["exportModalities"]:
            logger.info("Exporting RTDOSE")
            pinn.export_dose(export_plan, output_dir)

        for image in pinn.images:
            if image.image_info[0]["SeriesUID"] in settings["exportSeriesUIDs"]:
                pinn.export_image(image, export_path=output_dir)

        # Find the output files
        output_files = os.listdir(output_dir)
        output_files.sort()
        output_objects = [os.path.join(output_dir, f) for f in output_files]

        # Create the output data objects
        for obj in output_objects:

            # Write some meta data to patient comments field
            file_name = os.path.basename(obj)
            if file_name.startswith("R"):  # Don't add to the image series

                dicom_dataset = pydicom.read_file(obj)

                meta_data = {}
                meta_data["service"] = {
                    "tool": "Pinnacel Export Tool",
                    "trial": export_plan.active_trial["Name"],
                    "plan_date": export_plan.active_trial["ObjectVersion"]["WriteTimeStamp"],
                    "plan_locked": export_plan.plan_info["PlanIsLocked"],
                }

                if dicom_dataset.Modality == "RTPLAN":
                    meta_data["warning"] = ("WARNING: OUTPUT GENERATED FOR RTPLAN FILE IS "
                                            "UNVERIFIED AND MOST LIKELY INCORRECT!")

                if "meta" in data_object.meta_data.keys():
                    meta_data["meta"] = data_object.meta_data["meta"]

                if dicom_dataset.Modality == "RTPLAN":
                    dicom_dataset.RTPlanDescription = ("Pinnacle Export Meta Data written to "
                                                       "SOPAuthorizationComment")
                dicom_dataset.SOPAuthorizationComment = json.dumps(meta_data)

                dicom_dataset.save_as(obj)

            output_data_object = DataObject(type="DICOM", path=obj, parent=data_object)
            return_objects.append(output_data_object)

        # Delete files extracted from TAR
        shutil.rmtree(archive_path)

    logger.info("Finished Pinnacle Export")

    return return_objects


if __name__ == "__main__":

    # Run app by calling "python service.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "PINNACLE_EXPORT_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8001,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )

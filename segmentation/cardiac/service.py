"""
Service to run multi-atlas based cardiac segmentation.

Rob Finnegan
"""
import os

import SimpleITK as sitk
from loguru import logger
# import pydicom

# Need include celery here to be able to from Docker container
#pylint: disable=unused-import
from impit.framework import (
    app,
    DataObject,
    celery
 )

# from impit.dicom.nifti_to_rtstruct.convert import convert_nifti
from impit.segmentation.atlas.registration import (
    initial_registration,
    transform_propagation,
    fast_symmetric_forces_demons_registration,
    apply_field,
)

from impit.segmentation.atlas.label import (
    compute_weight_map,
    combine_labels,
    process_probability_image,
)

from impit.segmentation.atlas.iterative_atlas_removal import run_iar

from impit.segmentation.cardiac.cardiac import (
    AutoLungSegment,
    CropImage,
    vesselSplineGeneration,
)


CARDIAC_SETTINGS_DEFAULTS = {
    "outputFormat": "Auto_{0}.nii.gz",
    "atlasSettings": {
        # "atlasIdList": ["11", "12", "13", "14"],
        "atlasIdList": ['08','13','17','33','12','16','22','27','35'],
        "atlasStructures": ["WHOLEHEART", "LANTDESCARTERY"],
        # For development, run: 'export ATLAS_PATH=/atlas/path'
        "atlasPath": os.environ["ATLAS_PATH"],
    },
    "lungMaskSettings": {
        "coronalExpansion": 15,
        "axialExpansion": 5,
        "sagittalExpansion": 0,
        "lowerNormalisedThreshold": -0.1,
        "upperNormalisedThreshold": 0.4,
        "voxelCountThreshold": 5e4,
    },
    "rigidSettings": {
        "initialReg": "Affine",
        "options": {
            "shrinkFactors": [8, 2, 1],
            "smoothSigmas": [8, 2, 1],
            "samplingRate": 0.25,
            "finalInterp": sitk.sitkBSpline,
        },
        "trace": True,
        "guideStructure": False,
    },
    "deformableSettings": {
        "resolutionStaging": [4, 2, 1],
        "iterationStaging": [50, 25, 25],
        "ncores": 8,
        "trace": True,
    },
    "IARSettings": {
        "referenceStructure": "WHOLEHEART",
        "smoothDistanceMaps": True,
        "smoothSigma": 1,
        "zScoreStatistic": "MAD",
        "outlierMethod": "IQR",
        "outlierFactor": 1.5,
        "minBestAtlases": 3,
        "project_on_sphere":False,
    },
    "labelFusionSettings": {"voteType": "local", "optimalThreshold": {"WHOLEHEART": 0.44}},
    "vesselSpliningSettings": {
        "vesselNameList": ["LANTDESCARTERY"],
        "vesselRadius_mm": {"LANTDESCARTERY": 2.2},
        "spliningDirection": {"LANTDESCARTERY": "z"},
        "stopCondition": {"LANTDESCARTERY": "count"},
        "stopConditionValue": {"LANTDESCARTERY": 1},
    },
}


@app.register("Cardiac Segmentation", default_settings=CARDIAC_SETTINGS_DEFAULTS)
def cardiac_service(data_objects, working_dir, settings):
    """
    Implements the impit framework to provide cardiac atlas based segmentation.
    """

    logger.info("Running Cardiac Segmentation")
    logger.info("Using settings: " + str(settings))

    output_objects = []
    for data_object in data_objects:
        logger.info("Running on data object: " + data_object.path)

        # Read the image series
        load_path = data_object.path
        if data_object.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(data_object.path)

        img = sitk.ReadImage(load_path)

        """
        Initialisation - Read in atlases
        - image files
        - structure files

            Atlas structure:
            'ID': 'Original': 'CT Image'    : sitk.Image
                              'Struct A'    : sitk.Image
                              'Struct B'    : sitk.Image
                  'RIR'     : 'CT Image'    : sitk.Image
                              'Transform'   : transform parameter map
                              'Struct A'    : sitk.Image
                              'Struct B'    : sitk.Image
                  'DIR'     : 'CT Image'    : sitk.Image
                              'Transform'   : displacement field transform
                              'Weight Map'  : sitk.Image
                              'Struct A'    : sitk.Image
                              'Struct B'    : sitk.Image


        """

        logger.info("")
        # Settings
        atlas_path = settings["atlasSettings"]["atlasPath"]
        atlas_id_list = settings["atlasSettings"]["atlasIdList"]
        atlas_structures = settings["atlasSettings"]["atlasStructures"]

        atlas_set = {}
        for atlas_id in atlas_id_list:
            atlas_set[atlas_id] = {}
            atlas_set[atlas_id]["Original"] = {}

            atlas_set[atlas_id]["Original"]["CT Image"] = sitk.ReadImage(
                "{0}/Case_{1}/Images/Case_{1}_CROP.nii.gz".format(atlas_path, atlas_id)
            )

            for struct in atlas_structures:
                atlas_set[atlas_id]["Original"][struct] = sitk.ReadImage(
                    "{0}/Case_{1}/Structures/Case_{1}_{2}_CROP.nii.gz".format(
                        atlas_path, atlas_id, struct
                    )
                )

    """
    Step 1 - Automatic cropping using a translation transform
    - Registration of atlas images (maximum 5)
    - Potential expansion of the bounding box to ensure entire volume of interest is enclosed
    - Target image is cropped
    """
    # Settings
    quick_reg_settings = {"shrinkFactors": [8, 2],
                          "smoothSigmas": [8, 2],
                          "samplingRate": 0.2
                         }

    registered_crop_images = []

    for atlas_id in atlas_id_list[:max([5, len(atlas_id_list)])]:
        # Register the atlases
        atlas_set[atlas_id]["RIR"] = {}
        atlas_image = atlas_set[atlas_id]["Original"]["CT Image"]

        reg_image, crop_tfm = initial_registration(
            img,
            atlas_image,
            moving_structure=False,
            fixed_structure=False,
            options=quick_reg_settings,
            trace=False,
            reg_method="Rigid",
        )

        registered_crop_images.append(reg_image)

    combined_image_extent = (sum(registered_crop_images) > 0)

    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(combined_image_extent)
    bounding_box = np.array(shape_filter.GetBoundingBox(1))

    expansion = settings["autoCropSettings"]["expansion"]
    expansion_array = expansion*np.array(img.GetSpacing())

    # Avoid starting outside the image
    crop_box_index = np.max([bounding_box[:3]-expansion_array, np.array([0,0,0])], axis=0)

    # Avoid ending outside the image
    crop_box_size = np.min([np.array(image.GetSize())-crop_box_index,  bounding_box[3:]+2*expansion_array], axis=0)

    crop_box_size = [int(i) for i in crop_box_size]
    crop_box_index = [int(i) for i in crop_box_index]

    img_crop = sitk.RegionOfInterest(img, size=crop_box_size, index=crop_box_index)

        """
        Step 2 - Rigid registration of target images
        - Individual atlas images are registered to the target
        - The transformation is used to propagate the labels onto the target
        """
        initial_reg = settings["rigidSettings"]["initialReg"]
        rigid_options = settings["rigidSettings"]["options"]
        trace = settings["rigidSettings"]["trace"]
        guide_structure = settings["rigidSettings"]["guideStructure"]

        for atlas_id in atlas_id_list:
            # Register the atlases
            atlas_set[atlas_id]["RIR"] = {}
            atlas_image = atlas_set[atlas_id]["Original"]["CT Image"]

            if guide_structure:
                atlas_struct = atlas_set[atlas_id]["Original"][guide_structure]
            else:
                atlas_struct = False

            rigid_image, initial_tfm = initial_registration(
                img_crop,
                atlas_image,
                moving_structure=atlas_struct,
                options=rigid_options,
                trace=trace,
                reg_method=initial_reg,
            )

            # Save in the atlas dict
            atlas_set[atlas_id]["RIR"]["CT Image"] = rigid_image
            atlas_set[atlas_id]["RIR"]["Transform"] = initial_tfm

            # sitk.WriteImage(rigidImage, f'./RR_{atlas_id}.nii.gz')

            for struct in atlas_structures:
                input_struct = atlas_set[atlas_id]["Original"][struct]
                atlas_set[atlas_id]["RIR"][struct] = transform_propagation(
                    img_crop, input_struct, initial_tfm, structure=True, interp=sitk.sitkLinear
                )

        """
        Step 3 - Deformable image registration
        - Using Fast Symmetric Diffeomorphic Demons
        """
        # Settings
        resolution_staging = settings["deformableSettings"]["resolutionStaging"]
        iteration_staging = settings["deformableSettings"]["iterationStaging"]
        ncores = settings["deformableSettings"]["ncores"]
        trace = settings["deformableSettings"]["trace"]

        for atlas_id in atlas_id_list:
            # Register the atlases
            atlas_set[atlas_id]["DIR"] = {}
            atlas_image = atlas_set[atlas_id]["RIR"]["CT Image"]
            deform_image, deform_field = fast_symmetric_forces_demons_registration(
                img_crop,
                atlas_image,
                resolution_staging=resolution_staging,
                iteration_staging=iteration_staging,
                ncores=ncores,
                trace=trace,
            )

            # Save in the atlas dict
            atlas_set[atlas_id]["DIR"]["CT Image"] = deform_image
            atlas_set[atlas_id]["DIR"]["Transform"] = deform_field

            # sitk.WriteImage(deformImage, f'./DIR_{atlas_id}.nii.gz')

            for struct in atlas_structures:
                input_struct = atlas_set[atlas_id]["RIR"][struct]
                atlas_set[atlas_id]["DIR"][struct] = apply_field(
                    input_struct, deform_field, structure=True, interp=sitk.sitkLinear
                )

        """
        Step 4 - Iterative atlas removal
        - This is an automatic process that will attempt to remove inconsistent atlases from the entire set

        """
        # Compute weight maps
        # Here we use simple GWV as this minises the potentially negative influence of mis-registered atlases
        for atlas_id in atlas_id_list:
            atlas_image = atlas_set[atlas_id]["DIR"]["CT Image"]
            weight_map = compute_weight_map(img_crop, atlas_image, vote_type='global')
            atlas_set[atlas_id]["DIR"]["Weight Map"] = weight_map

        reference_structure = settings["IARSettings"]["referenceStructure"]
        smooth_distance_maps = settings["IARSettings"]["smoothDistanceMaps"]
        smooth_sigma = settings["IARSettings"]["smoothSigma"]
        z_score_statistic = settings["IARSettings"]["zScoreStatistic"]
        outlier_method = settings["IARSettings"]["outlierMethod"]
        outlier_factor = settings["IARSettings"]["outlierFactor"]
        min_best_atlases = settings["IARSettings"]["minBestAtlases"]
        project_on_sphere = settings["IARSettings"]["project_on_sphere"]

        atlas_set = run_iar(
            atlas_set=atlas_set,
            structure_name=reference_structure,
            smooth_maps=smooth_distance_maps,
            smooth_sigma=smooth_sigma,
            z_score=z_score_statistic,
            outlier_method=outlier_method,
            min_best_atlases=min_best_atlases,
            n_factor=outlier_factor,
            iteration=0,
            single_step=False,
            project_on_sphere=project_on_sphere
        )

        """
        Step 4 - Vessel Splining

        """

        vessel_name_list = settings["vesselSpliningSettings"]["vesselNameList"]
        vessel_radius_mm = settings["vesselSpliningSettings"]["vesselRadius_mm"]
        splining_direction = settings["vesselSpliningSettings"]["spliningDirection"]
        stop_condition = settings["vesselSpliningSettings"]["stopCondition"]
        stop_condition_value = settings["vesselSpliningSettings"]["stopConditionValue"]

        segmented_vessel_dict = vesselSplineGeneration(
            atlas_set,
            vessel_name_list,
            vesselRadiusDict=vessel_radius_mm,
            stopConditionTypeDict=stop_condition,
            stopConditionValueDict=stop_condition_value,
            scanDirectionDict=splining_direction,
        )

        """
        Step 5 - Label Fusion
        """
        # Compute weight maps
        # Here we use local weighted fusion
        for atlas_id in list(atlas_set.keys()):
            atlas_image = atlas_set[atlas_id]["DIR"]["CT Image"]
            weight_map = compute_weight_map(img_crop, atlas_image)
            atlas_set[atlas_id]["DIR"]["Weight Map"] = weight_map

        combined_label_dict = combine_labels(atlas_set, atlas_structures)

        """
        Step 6 - Paste the cropped structure into the original image space
        """

        output_format = settings["outputFormat"]

        template_im = sitk.Cast((img * 0), sitk.sitkUInt8)

        vote_structures = settings["labelFusionSettings"]["optimalThreshold"].keys()

        for structure_name in vote_structures:
            optimal_threshold = settings["labelFusionSettings"]["optimalThreshold"][structure_name]
            binary_struct = process_probability_image(
                combined_label_dict[structure_name], optimal_threshold
            )
            paste_img = sitk.Paste(
                template_im, binary_struct, binary_struct.GetSize(), (0, 0, 0), (sag0, cor0, ax0)
            )

            # Write the mask to a file in the working_dir
            mask_file = os.path.join(working_dir, output_format.format(structure_name))
            sitk.WriteImage(paste_img, mask_file)

            # Create the output Data Object and add it to the list of output_objects
            output_data_object = DataObject(type="FILE", path=mask_file, parent=data_object)
            output_objects.append(output_data_object)

        for structure_name in vessel_name_list:
            binary_struct = segmented_vessel_dict[structure_name]
            paste_img = sitk.Paste(
                template_im, binary_struct, binary_struct.GetSize(), (0, 0, 0), (sag0, cor0, ax0)
            )

            # Write the mask to a file in the working_dir
            mask_file = os.path.join(working_dir, output_format.format(structure_name))
            sitk.WriteImage(paste_img, mask_file)

            # Create the output Data Object and add it to the list of output_objects
            output_data_object = DataObject(type="FILE", path=mask_file, parent=data_object)
            output_objects.append(output_data_object)

        # If the input was a DICOM, then we can use it to generate an output RTStruct
        # if data_object.type == 'DICOM':

        #     dicom_file = load_path[0]
        #     logger.info('Will write Dicom using file: {0}'.format(dicom_file))
        #     masks = {settings['outputContourName']: mask_file}

        #     # Use the image series UID for the file of the RTStruct
        #     suid = pydicom.dcmread(dicom_file).SeriesInstanceUID
        #     output_file = os.path.join(working_dir, 'RS.{0}.dcm'.format(suid))

        #     # Use the convert nifti function to generate RTStruct from nifti masks
        #     convert_nifti(dicom_file, masks, output_file)

        #     # Create the Data Object for the RTStruct and add it to the list
        #     do = DataObject(type='DICOM', path=output_file, parent=d)
        #     output_objects.append(do)

        #     logger.info('RTStruct generated')

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python sample.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "SAMPLE_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )

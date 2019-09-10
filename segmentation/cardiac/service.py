import sys
sys.path.append('../../..')

from impit.framework import app, DataObject, celery
from impit.dicom.nifti_to_rtstruct.convert import convert_nifti

import SimpleITK as sitk
import pydicom
from loguru import logger
import os
os.system('export ATLAS_PATH=/home/robbie/Documents/University/PhD/Research/Software/impit/TempCardiacData')

from cardiac import *

# TO DO
# 1. add options for rigid registration


cardiac_settings_defaults = {
    'outputFormat'          : 'Auto_{0}.nii.gz',
    'atlasSettings'         : {
                                'atlasIdList':               ['04D1','1FA5','7AAC','8505','5FBF'],
                                'atlasStructures':           ['COR','LAD'],
                                'atlasPath':                 '/home/robbie/Documents/University/PhD/Research/Software/impit/TempCardiacData'#os.environ['ATLAS_PATH']
                              },
    'lungMaskSettings'      : {
                                'coronalExpansion':          15,
                                'axialExpansion':            5,
                                'sagittalExpansion':         0,
                                'lowerNormalisedThreshold':  -0.1,
                                'upperNormalisedThreshold':  0.4,
                                'voxelCountThreshold':       5e4
                              },
    'rigidSettings'         : {


                              },
    'deformableSettings'    : {
                                'resolutionStaging':        [8,4,2,1],
                                'iterationStaging':         [25,10,10,5],
                                'ncores':                   4
                              },
    'IARSettings'    : {
                                'referenceStructure':       'COR',
                                'smoothDistanceMaps':       True,
                                'smoothSigma':              1,
                                'zScoreStatistic':          'MAD',
                                'outlierMethod':            'IQR',
                                'outlierFactor':            1.5,
                                'minBestAtlases':           4
                              },
    'labelFusionSettings'   : {
                                'voteType':                 'local',
                                'optimalThreshold':         {'COR':0.34}
                              },
    'vesselSpliningSettings': {
                                'vesselNameList':           ['LAD'],
                                'vesselRadius_mm':          {'LAD':2.2},
                                'spliningDirection':        {'LAD':'z'},
                                'stopCondition':            {'LAD':'count'},
                                'stopConditionValue':       {'LAD':0}
                              }
}

@app.register('Cardiac Segmentation', default_settings=cardiac_settings_defaults)
def cardiac_service(data_objects, working_dir, settings):

    logger.info('Running Cardiac Segmentation')
    logger.info('Using settings: ' + str(settings))

    output_objects = []
    for d in data_objects:
        logger.info('Running on data object: ' + d.path)

        # Read the image series
        load_path = d.path
        if d.type == 'DICOM':
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(d.path)

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
        logger.info('')
        # Settings
        atlasPath = settings['atlasSettings']['atlasPath']
        atlasIdList = settings['atlasSettings']['atlasIdList']
        atlasStructures = settings['atlasSettings']['atlasStructures']

        atlasSet = {}
        for atlasId in atlasIdList:
            atlasSet[atlasId] = {}
            atlasSet[atlasId]['Original'] = {}

            atlasSet[atlasId]['Original']['CT Image'] = sitk.ReadImage('{0}/Case_{1}/Images/Case_{1}.nii.gz'.format(atlasPath, atlasId))

            for struct in atlasStructures:
                atlasSet[atlasId]['Original'][struct] = sitk.ReadImage('{0}/Case_{1}/Structures/Case_{1}_{2}.nii.gz'.format(atlasPath, atlasId, struct))

        """
        Step 1 - Automatic cropping using the lung volume
        - Airways are segmented
        - A bounding box is defined
        - Potential expansion of the bounding box to ensure entire heart volume is enclosed
        - Target image is cropped
        """
        # Settings
        sagittalExpansion = settings['lungMaskSettings']['sagittalExpansion']
        coronalExpansion  = settings['lungMaskSettings']['coronalExpansion']
        axialExpansion    = settings['lungMaskSettings']['axialExpansion']

        lowerNormalisedThreshold = settings['lungMaskSettings']['lowerNormalisedThreshold']
        upperNormalisedThreshold = settings['lungMaskSettings']['upperNormalisedThreshold']
        voxelCountThreshold      = settings['lungMaskSettings']['voxelCountThreshold']

        # Get the bounding box containing the lungs
        lungBoundingBox, _ = AutoLungSegment(img, l=lowerNormalisedThreshold, u=upperNormalisedThreshold, NPthresh=voxelCountThreshold)

        # Add an optional expansion
        sag0 = max([lungBoundingBox[0] - sagittalExpansion, 0])
        cor0 = max([lungBoundingBox[1] - coronalExpansion,  0])
        ax0  = max([lungBoundingBox[2] - axialExpansion,    0])

        sagD = min([lungBoundingBox[3] + sagittalExpansion, img.GetSize()[0] - sag0])
        corD = min([lungBoundingBox[4] + coronalExpansion,  img.GetSize()[1] - cor0])
        axD  = min([lungBoundingBox[5] + axialExpansion,    img.GetSize()[2] - ax0])

        cropBox = (sag0, cor0, ax0, sagD, corD, axD)

        # Crop the image down
        imgCrop = CropImage(img, cropBox)

        # We should check here that the lung segmentation has worked, otherwise we need another option!
        # For example, translation registration with a pre-cropped image

        """
        Step 2 - Rigid registration of target images
        - Individual atlas images are registered to the target
        - The transformation is used to propagate the labels onto the target
        """
        for atlasId in atlasIdList:
            # Register the atlases
            atlasSet[atlasId]['RIR'] = {}
            atlasImage = atlasSet[atlasId]['Original']['CT Image']
            rigidImage, rigidTfm = RigidRegistration(imgCrop, atlasImage)

            # Save in the atlas dict
            atlasSet[atlasId]['RIR']['CT Image']  = rigidImage
            atlasSet[atlasId]['RIR']['Transform'] = rigidTfm


            for struct in atlasStructures:
                inputStruct = atlasSet[atlasId]['Original'][struct]
                atlasSet[atlasId]['RIR'][struct] = RigidPropagation(imgCrop, inputStruct, rigidTfm, structure=True, interpOrder=0)


        """
        Step 3 - Deformable image registration
        - Using Fast Symmetric Diffeomorphic Demons
        """
        # Settings
        resolutionStaging = settings['deformableSettings']['resolutionStaging']
        iterationStaging  = settings['deformableSettings']['iterationStaging']
        ncores            = settings['deformableSettings']['ncores']

        for atlasId in atlasIdList:
            # Register the atlases
            atlasSet[atlasId]['DIR'] = {}
            atlasImage = atlasSet[atlasId]['RIR']['CT Image']
            deformImage, deformField = FastSymmetricForcesDemonsRegistration(imgCrop, atlasImage, resolutionStaging=resolutionStaging, iterationStaging=iterationStaging, ncores=ncores)

            # Save in the atlas dict
            atlasSet[atlasId]['DIR']['CT Image']  = deformImage
            atlasSet[atlasId]['DIR']['Transform'] = deformField

            for struct in atlasStructures:
                inputStruct = atlasSet[atlasId]['RIR'][struct]
                atlasSet[atlasId]['DIR'][struct] = ApplyField(inputStruct, deformField, 1, 0)

                sitk.WriteImage(atlasSet[atlasId]['DIR'][struct], f'{atlasId}.{struct}.nii.gz')


        """
        Step 4 - Iterative atlas removal
        - This is an automatic process that will attempt to remove inconsistent atlases from the entire set

        """

        # Compute weight maps
        for atlasId in atlasIdList:
            atlasImage = atlasSet[atlasId]['DIR']['CT Image']
            weightMap = computeWeightMap(imgCrop, atlasImage)
            atlasSet[atlasId]['DIR']['Weight Map'] = weightMap

        referenceStructure  = settings['IARSettings']['referenceStructure']
        smoothDistanceMaps  = settings['IARSettings']['smoothDistanceMaps']
        smoothSigma         = settings['IARSettings']['smoothSigma']
        zScoreStatistic     = settings['IARSettings']['zScoreStatistic']
        outlierMethod       = settings['IARSettings']['outlierMethod']
        outlierFactor       = settings['IARSettings']['outlierFactor']
        minBestAtlases      = settings['IARSettings']['minBestAtlases']

        atlasSet = IAR(atlasSet         = atlasSet,
                       structureName    = referenceStructure,
                       smoothMaps       = smoothDistanceMaps,
                       smoothSigma      = smoothSigma,
                       zScore           = zScoreStatistic,
                       outlierMethod    = outlierMethod,
                       minBestAtlases   = minBestAtlases,
                       N_factor         = outlierFactor,
                       logFile          = 'IAR_{0}.log'.format(datetime.datetime.now()),
                       debug            = False,
                       iteration        = 0,
                       singleStep       = False)

        """
        Step 4 - Vessel Splining

        """

        vesselNameList      = settings['vesselSpliningSettings']['vesselNameList']
        vesselRadius_mm     = settings['vesselSpliningSettings']['vesselRadius_mm']
        spliningDirection   = settings['vesselSpliningSettings']['spliningDirection']
        stopCondition       = settings['vesselSpliningSettings']['stopCondition']
        stopConditionValue  = settings['vesselSpliningSettings']['stopConditionValue']

        segmentedVesselDict = vesselSplineGeneration(atlasSet, vesselNameList, vesselRadiusDict=vesselRadius_mm, stopConditionTypeDict=stopCondition, stopConditionValueDict=stopConditionValue, scanDirectionDict=spliningDirection)


        """
        Step 5 - Label Fusion
        """
        combinedLabelDict = combineLabels(atlasSet, atlasStructures)


        """
        Step 6 - Paste the cropped structure into the original image space
        """

        outputFormat = settings['outputFormat']

        templateIm = sitk.Cast((img * 0),sitk.sitkUInt8)

        voteStructures = settings['labelFusionSettings']['optimalThreshold'].keys()

        for structureName in voteStructures:
            optimalThreshold = settings['labelFusionSettings']['optimalThreshold'][structureName]
            binaryStruct = processProbabilityImage(combinedLabelDict[structureName], 0.5)
            pasteImg = sitk.Paste(templateIm, binaryStruct, binaryStruct.GetSize(), (0,0,0), (sag0, cor0, ax0))

            # Write the mask to a file in the working_dir
            mask_file = os.path.join(
                working_dir, outputFormat.format(structureName))
            sitk.WriteImage(pasteImg, mask_file)

            # Create the output Data Object and add it to the list of output_objects
            do = DataObject(type='FILE', path=mask_file, parent=d)
            output_objects.append(do)

        for structureName in vesselNameList:
            binaryStruct = segmentedVesselDict[structureName]
            pasteImg = sitk.Paste(templateIm, binaryStruct, binaryStruct.GetSize(), (0,0,0), (sag0, cor0, ax0))

            # Write the mask to a file in the working_dir
            mask_file = os.path.join(
                working_dir, outputFormat.format(structureName))
            sitk.WriteImage(pasteImg, mask_file)

            # Create the output Data Object and add it to the list of output_objects
            do = DataObject(type='FILE', path=mask_file, parent=d)
            output_objects.append(do)

        # If the input was a DICOM, then we can use it to generate an output RTStruct
        # if d.type == 'DICOM':

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

    dicom_listener_port=7777
    dicom_listener_aetitle="SAMPLE_SERVICE"

    app.run(debug=True, host="0.0.0.0", port=8000,
        dicom_listener_port=dicom_listener_port,
        dicom_listener_aetitle=dicom_listener_aetitle)

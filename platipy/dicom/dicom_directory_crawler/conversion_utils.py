
import re

import pydicom
import numpy as np
import SimpleITK as sitk

from skimage.draw import polygon
from loguru import logger

def get_dicom_info_from_description(dicom_object, return_extra=False):
    """
    Attempts to return some information from a DICOM
    This is typically used for naming converted NIFTI files

    Args:
        dicom_object (pydicom.dataset.FileDataset): The DICOM object
        return_extra (bool, optional): return information that is usually not required

    Returns:
        info (str): Some extracted information
    """
    dicom_sop_class_uid = dicom_object.SOPClassUID
    dicom_sop_class_name = pydicom._uid_dict.UID_dictionary[dicom_sop_class_uid][0]

    if 'Image' in dicom_sop_class_name:
        # Get the modality
        image_modality = dicom_object.Modality
    
        logger.info(f'    Image modality: {image_modality}')

        if image_modality=='CT':
            # There is typically not much extra information
            # At the moment, we do not return anything for CT imaging
            if return_extra:
                protocol_name = dicom_object.ProtocolName
            
                if protocol_name != '':
                    return re.sub(r'[^\w]', '_', protocol_name).upper() 

            return ''

        elif image_modality=='MR':
            # Not much consistency, but we can get the protocol name
            protocol_name = dicom_object.ProtocolName
            
            if protocol_name != '':
                return re.sub(r'[^\w]', '_', protocol_name).upper()

            else:
                # If there is no information in "ProtocolName" we can try another field
                sequence_name = dicom_object.SequenceName

                return re.sub(r'[^\w]', '_', sequence_name).upper()

        elif image_modality=='PT':
            # Not much experience with this
            # We can search through the corrections applied
            # Return whether or not attentuation is applied

            corrections = dicom_object.CorrectedImage

            if 'ATTN' in corrections:
                return 'AC'
            else:
                return 'NAC'





def sort_dicom_image_list(dicom_image_list, sort_by='SliceLocation'):
    """
    Sorts a list of DICOM image files based on a DICOM tag value

    Args:
        dicom_image_list (list): [description]
        sort_by (str, optional): [description]. Defaults to 'SliceLocation'.
    """
    sorter_float = lambda dcm_file: float(pydicom.read_file(dcm_file, force=True)[sort_by].value)

    return sorted(dicom_image_list, key=sorter_float)

def fix_missing_data(contour_data_list):
    """
    Fixes missing points in contouring using simple linear interpolation


    Args:
        contour_data_list (list): The contour data for each slice

    Returns:
        contour_data (numpy array): Interpolated contour data
    """
    contour_data = np.array(contour_data_list)
    if contour_data.any()=='':
        logger.warning("    Missing values detected.")
        missing_values = np.where(contour_data=='')[0]
        if missing_values.shape[0]>1:
            logger.warning("    More than one value missing, fixing this isn't implemented yet...")
        else:
            logger.warning("    Only one value missing.")
            missing_index = missing_values[0]
            missing_axis = missing_index%3
            if missing_axis==0:
                logger.warning("    Missing value in x axis: interpolating.")
                if missing_index>len(contour_data)-3:
                    lower_val = contour_data[missing_index-3]
                    upper_val = contour_data[0]
                elif missing_index==0:
                    lower_val = contour_data[-3]
                    upper_val = contour_data[3]
                else:
                    lower_val = contour_data[missing_index-3]
                    upper_val = contour_data[missing_index+3]
                contour_data[missing_index] = 0.5*(lower_val+upper_val)
            elif missing_axis==1:
                logger.warning("    Missing value in y axis: interpolating.")
                if missing_index>len(contour_data)-2:
                    lower_val = contour_data[missing_index-3]
                    upper_val = contour_data[1]
                elif missing_index==0:
                    lower_val = contour_data[-2]
                    upper_val = contour_data[4]
                else:
                    lower_val = contour_data[missing_index-3]
                    upper_val = contour_data[missing_index+3]
                contour_data[missing_index] = 0.5*(lower_val+upper_val)
            else:
                logger.warning("    Missing value in z axis: taking slice value")
                temp = contour_data[2::3].tolist()
                temp.remove('')
                contour_data[missing_index] = np.min(np.array(temp, dtype=np.double))
    return contour_data

def transform_point_set_from_dicom_struct(dicom_image, dicom_struct, spacing_override=False):

    if spacing_override:
        current_spacing = list(dicom_image.GetSpacing())
        new_spacing = tuple([current_spacing[k] if spacing_override[k]==0 else spacing_override[k] for k in range(3)])
        dicom_image.SetSpacing(new_spacing)

    struct_point_sequence = dicom_struct.ROIContourSequence
    struct_name_sequence = ['_'.join(i.ROIName.split()) for i in dicom_struct.StructureSetROISequence]

    structure_list = []
    final_struct_name_sequence = []

    for structIndex, structure_name in enumerate(struct_name_sequence):
        image_blank = np.zeros(dicom_image.GetSize()[::-1], dtype=np.uint8)
        logger.info("    Converting structure {0} with name: {1}".format(structIndex, structure_name))

        if not hasattr(struct_point_sequence[structIndex], "ContourSequence"):
            logger.warning("    No contour sequence found for this structure, skipping.")
            continue

        if not struct_point_sequence[structIndex].ContourSequence[0].ContourGeometricType=="CLOSED_PLANAR":
            logger.warning("    This is not a closed planar structure, skipping.")
            continue

        for sl in range(len(struct_point_sequence[structIndex].ContourSequence)):

            contour_data = fix_missing_data(struct_point_sequence[structIndex].ContourSequence[sl].ContourData)

            struct_slice_contour_data = np.array(contour_data, dtype=np.double)
            vertexArr_physical = struct_slice_contour_data.reshape(struct_slice_contour_data.shape[0]//3,3)

            point_arr = np.array([dicom_image.TransformPhysicalPointToIndex(i) for i in vertexArr_physical]).T

            [xVertexArr_image, yVertexArr_image] = point_arr[[0,1]]
            zIndex = point_arr[2][0]

            if np.any(point_arr[2]!=zIndex):
                logger.error("    Axial slice index varies in contour. Quitting now.")
                logger.error("    Structure:   {0}".format(structure_name))
                logger.error("    Slice index: {0}".format(zIndex))
                quit()

            if zIndex>=dicom_image.GetSize()[2]:
                logger.warning("    Slice index greater than image size. Skipping slice.")
                logger.warning("    Structure:   {0}".format(structure_name))
                logger.warning("    Slice index: {0}".format(zIndex))
                continue

            sliceArr = np.zeros(dicom_image.GetSize()[:2], dtype=np.uint8)
            filledIndicesX, filledIndicesY = polygon(xVertexArr_image, yVertexArr_image, shape=sliceArr.shape)
            sliceArr[filledIndicesX, filledIndicesY] = 1
            image_blank[zIndex] += sliceArr.T

        struct_image = sitk.GetImageFromArray(1*(image_blank>0))
        struct_image.CopyInformation(dicom_image)
        structure_list.append(sitk.Cast(struct_image, sitk.sitkUInt8))
        structure_name_clean = re.sub(r'[^\w]', '_', structure_name).upper()
        while '__' in structure_name_clean:
            structure_name_clean = structure_name_clean.replace('__','_')
        final_struct_name_sequence.append(structure_name_clean)

    return final_struct_name_sequence, structure_list
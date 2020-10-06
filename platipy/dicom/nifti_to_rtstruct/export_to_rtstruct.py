# Copyright 2020 CSIRO

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pydicom
import datetime
import itertools
import yaml
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian


def copy_attrs_s2t(attrs, source, target, debug):

    for attr in attrs:
        try:
            target.__setattr__(attr, source.__getattr__(attr))
        except AttributeError:
            if debug:
                print("%s not found in dataset" % attr)


def load_param_file(filename):

    aa = yaml.load(open(filename))
    path = os.path.dirname(os.path.abspath(filename))

    for roi in aa["ROIS"]:

        ff = aa["ROIS"][roi]["ContourFile"]
        bb = yaml.load(open(path + "/" + ff))
        aa["ROIS"][roi]["Contours"] = bb["Contours"]

    return aa


def export_to_rtstruct(params, dat_ct, debug=False):

    StudyComponentManagementSOPClass = "1.2.840.10008.3.1.2.3.2"
    CTImageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    dt_now = datetime.datetime.now()

    dat = Dataset()
    dat.ImageType = ["DERIVED", "PRIMARY"]

    # Lets do the patient module stuff here
    pm_attrs = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
    ]
    copy_attrs_s2t(pm_attrs, dat_ct, dat, debug)

    # Let's write the study module
    sm_attrs = [
        "StudyInstanceUID",
        "StudyDate",
        "StudyTime",
        "ReferringPhysicianName",
        "StudyID",
        "AccessionNumber",
        "StudyDescription",
        "PhysiciansOfRecord",
        "ReferencedStudySequence",
    ]
    copy_attrs_s2t(sm_attrs, dat_ct, dat, debug)

    # RT Series Module
    dat.Modality = "RTSTRUCT"
    dat.ConversionType = "SYN"
    dat.SeriesInstanceUID = pydicom.uid.generate_uid()  # None, create one!
    dat.SeriesNumber = 6  # Get this from the
    dat.SeriesDescription = "ARIA RadOnc Structure Sets"

    dat.StationName = "CSIRO"

    dat_sis = Dataset()
    dat_sis.ReferencedSOPClassUID = StudyComponentManagementSOPClass
    dat_sis.ReferencedSOPInstanceUID = dat_ct.StudyInstanceUID
    dat.SourceImageSequence = Sequence([dat_sis])

    # General Equipment Module
    dat.Manufacturer = "CSIRO"

    # Common Module
    # ??
    cm_attrs = ["AESourceApplicationEntityTitle", "TimezoneOffsetFromUTC"]
    copy_attrs_s2t(cm_attrs, dat_ct, dat, debug)
    # Structure Set Module
    dat.InstanceCreationDate = dt_now.strftime("%Y%m%d")
    dat.InstanceCreationTime = dt_now.strftime("%H%M%S.00000")

    dat.StructureSetLabel = "WHOLE PELVIS"
    dat.StructureSetName = "CSIRO"
    #   dat.StructureSetDescription = ??
    #    dat.InstanceNumber = ??
    dat.StructureSetDate = dt_now.strftime("%Y%m%d")
    dat.StructureSetTime = dt_now.strftime("%H%M%S.00000")

    # rt reference frame of reference sequence
    rt_fors = Dataset()
    rt_fors.FrameOfReferenceUID = dat_ct.FrameOfReferenceUID
    # rt reference study
    rt_ref_study = Dataset()
    rt_ref_study.ReferencedSOPClassUID = dat_ct.SOPClassUID
    rt_ref_study.ReferencedSOPInstanceUID = dat_ct.SOPInstanceUID

    # rt reference series sequence
    rt_ref_ser = Dataset()
    rt_ref_ser.SeriesInstanceUID = dat_ct.SeriesInstanceUID
    # contour Image sequence
    ll = []
    startNum = params["StartSliceNum"]
    for ii in range(0, params["NumSlices"]):
        temp_dat = Dataset()
        tempSOPInstUID = params["UIDPrefix"] + str(
            startNum
        )  # this can just be grabbed from the
        temp_dat.ReferencedSOPClassUID = CTImageSOPClassUID
        temp_dat.ReferencedSOPInstanceUID = tempSOPInstUID
        ll.append(temp_dat)
        startNum = startNum + 1

    rt_ref_ser.ContourImageSequence = Sequence(ll)
    rt_ref_study.RTReferencedSeriesSequence = Sequence([rt_ref_ser])
    rt_fors.RTReferencedStudySequence = Sequence([rt_ref_study])

    dat.ReferencedFrameOfReferenceSequence = Sequence([rt_fors])

    # Structure Set ROI sequence
    ll = []
    for ii, roi in enumerate(params["ROIS"]):
        tdat = Dataset()
        tdat.ROINumber = ii + 1
        tdat.ReferencedFrameOfReferenceUID = dat_ct.FrameOfReferenceUID
        tdat.ROIName = roi
        tdat.ROIGenerationAlgorithm = "AUTOMATIC"
        ll.append(tdat)

    dat.StructureSetROISequence = Sequence(ll)

    # ROI Contour Sequence
    ll = []

    for ii, roi in enumerate(params["ROIS"]):
        roi_con = Dataset()
        # Contour Sequence
        roi_con.ReferencedROINumber = ii + 1
        roi_con.ROIDisplayColor = params["ROIS"][roi]["Color"]
        cont_ll = []
        for jj, contour in enumerate(params["ROIS"][roi]["Contours"]):
            cont = Dataset()
            sliceNum = contour["SliceNumber"]
            cont_img = Dataset()
            cont_img.ReferencedSOPClassUID = CTImageSOPClassUID
            tempSOPInstUID = params["UIDPrefix"] + str(
                params["StartSliceNum"] + sliceNum
            )
            cont_img.ReferencedSOPInstanceUID = tempSOPInstUID
            cont.ContourImageSequence = Sequence([cont_img])
            cont.ContourGeometricType = contour["GeometricType"]
            cont.NumberOfContourPoints = len(contour["Data"])
            cont.ContourData = list(itertools.chain.from_iterable(contour["Data"]))
            cont_ll.append(cont)
        roi_con.ContourSequence = Sequence(cont_ll)

        ll.append(roi_con)

    dat.ROIContourSequence = Sequence(ll)

    ll = []
    # ROI observations module
    for ii, roi in enumerate(params["ROIS"]):
        roi_obs = Dataset()
        roi_obs.ObservationNumber = ii + 1
        roi_obs.ReferencedROINumber = ii + 1
        roi_obs.RTROIIntepretedType = params["ROIS"][roi]["InterpretedType"]
        roi_obs.Intepreter = ""
        ll.append(roi_obs)

    dat.RTROIObservationsSequence = Sequence(ll)

    meta = Dataset()
    # meta.is_implicit_VR = False
    # meta.is_little_endian = True
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"
    meta.FileMetaInformationGroupLength = 0
    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.TransferSyntaxUID = ImplicitVRLittleEndian
    dat.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    dat.SOPInstanceUID = pydicom.uid.generate_uid()
    #    dat.ImplementationClassUID = dicom.UID.generate_uid()
    fd = FileDataset(
        "temp.dcm",
        dat,
        file_meta=meta,
        preamble=b"\0" * 128,
        is_implicit_VR=True,
        is_little_endian=True,
    )
    return fd

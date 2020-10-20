# Copyright 2020 University of New South Wales, University of Sydney, Ingham Institute

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
import tempfile
import pydicom

from pydicom.dataset import Dataset, FileDataset

from pydicom.uid import ImplicitVRLittleEndian
from pynetdicom import (
    AE,
    evt,
    VerificationPresentationContexts,
    StoragePresentationContexts,
    QueryRetrievePresentationContexts,
    PYNETDICOM_IMPLEMENTATION_UID,
    PYNETDICOM_IMPLEMENTATION_VERSION,
)
from pynetdicom.sop_class import (
    VerificationSOPClass,
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelGet,
    PatientRootQueryRetrieveInformationModelMove
)
from pynetdicom.pdu_primitives import SCP_SCU_RoleSelectionNegotiation

from loguru import logger


class DicomConnector:
    def __init__(self, host="127.0.0.1", port=0, ae_title="", output_directory=tempfile.mkdtemp()):
        self.host = host
        self.port = port
        self.ae_title = ae_title if ae_title else ""

        logger.debug(
            "DicomConnector with host: "
            + self.host
            + " port: "
            + str(self.port)
            + " AETitle: "
            + self.ae_title
        )

        self.output_directory = output_directory
        self.current_dir = None
        self.recieved_callback = None

    def verify(self):
        # Verify Connection

        ae = AE()
        ae.requested_contexts = VerificationPresentationContexts

        # Associate with a peer DICOM AE
        if len(self.ae_title) > 0:
            assoc = ae.associate(self.host, self.port, ae_title=self.ae_title)
        else:
            assoc = ae.associate(self.host, self.port)

        result = None

        if assoc.is_established:
            status = assoc.send_c_echo()

            if status:
                result = status.Status

            # Release the association
            assoc.release()

        return not result is None

    def do_find(self, dataset, query_model=PatientRootQueryRetrieveInformationModelFind):

        ae = AE()

        ae.requested_contexts = QueryRetrievePresentationContexts

        if len(self.ae_title) > 0:
            assoc = ae.associate(self.host, self.port, ae_title=self.ae_title)
        else:
            assoc = ae.associate(self.host, self.port)

        results = []
        if assoc.is_established:
            logger.info("Association accepted by the peer")

            responses = assoc.send_c_find(dataset, query_model=query_model)

            for _, ds in responses:
                results.append(ds)

            # Release the association
            assoc.release()

        logger.info("Got " + str(len(results)) + " results")

        return results

    def get_studies_for_patient(self, patient_id):

        dataset = Dataset()
        dataset.StudyInstanceUID = ""
        dataset.StudyDescription = ""
        dataset.PatientID = patient_id
        dataset.PatientName = ""
        dataset.QueryRetrieveLevel = "STUDY"

        return self.do_find(dataset)

    def get_series_for_study(self, study_instance_uid, modality):

        dataset = Dataset()
        dataset.StudyInstanceUID = study_instance_uid
        dataset.Modality = modality
        dataset.SeriesInstanceUID = ""
        dataset.SeriesDescription = ""
        dataset.QueryRetrieveLevel = "SERIES"

        return self.do_find(dataset)

    def move_series(
        self,
        seriesInstanceUID,
        move_aet="PYNETDICOM",
        query_model=PatientRootQueryRetrieveInformationModelMove,
    ):

        ae = AE()
        ae.requested_contexts = QueryRetrievePresentationContexts

        if len(self.ae_title) > 0:
            assoc = ae.associate(self.host, self.port, ae_title=self.ae_title)
        else:
            assoc = ae.associate(self.host, self.port)

        if assoc.is_established:
            logger.info("Association accepted by the peer")

            dataset = Dataset()
            dataset.SeriesInstanceUID = seriesInstanceUID
            dataset.QueryRetrieveLevel = "SERIES"

            responses = assoc.send_c_move(dataset, move_aet, query_model=query_model)

            for (_, _) in responses:
                pass

            # Release the association
            assoc.release()

        logger.info("Finished")

    def download_series(
        self,
        series_instance_uid,
        recieved_callback=None,
        query_model=PatientRootQueryRetrieveInformationModelGet,
    ):

        self.recieved_callback = recieved_callback

        ae = AE()

        # Specify which SOP Classes are supported as an SCU
        for context in QueryRetrievePresentationContexts:
            ae.add_requested_context(context.abstract_syntax, ImplicitVRLittleEndian)
        for context in StoragePresentationContexts[:115]:
            ae.add_requested_context(context.abstract_syntax, ImplicitVRLittleEndian)

        # Add SCP/SCU Role Selection Negotiation to the extended negotiation
        # We want to act as a Storage SCP
        ext_neg = []
        for context in StoragePresentationContexts:
            role = SCP_SCU_RoleSelectionNegotiation()
            role.sop_class_uid = context.abstract_syntax
            role.scp_role = True
            role.scu_role = False
            ext_neg.append(role)

        handlers = [(evt.EVT_C_STORE, self.on_c_store)]

        if len(self.ae_title) > 0:
            assoc = ae.associate(
                self.host,
                self.port,
                ae_title=self.ae_title,
                ext_neg=ext_neg,
                evt_handlers=handlers,
            )
        else:
            assoc = ae.associate(self.host, self.port, ext_neg=ext_neg, evt_handlers=handlers)

        if assoc.is_established:
            logger.info("Association accepted by the peer")

            dataset = Dataset()
            dataset.SeriesInstanceUID = series_instance_uid
            dataset.QueryRetrieveLevel = "SERIES"

            responses = assoc.send_c_get(dataset, query_model=query_model)

            for (_, _) in responses:
                pass

            # Release the association
            assoc.release()

        logger.info("Finished")

        return self.current_dir

    def on_c_store(self, event):

        dataset = event.dataset

        mode_prefix = "UN"
        mode_prefixes = {
            "CT Image Storage": "CT",
            "Enhanced CT Image Storage": "CTE",
            "MR Image Storage": "MR",
            "Enhanced MR Image Storage": "MRE",
            "Positron Emission Tomography Image Storage": "PT",
            "Enhanced PET Image Storage": "PTE",
            "RT Image Storage": "RI",
            "RT Dose Storage": "RD",
            "RT Plan Storage": "RP",
            "RT Structure Set Storage": "RS",
            "Computed Radiography Image Storage": "CR",
            "Ultrasound Image Storage": "US",
            "Enhanced Ultrasound Image Storage": "USE",
            "X-Ray Angiographic Image Storage": "XA",
            "Enhanced XA Image Storage": "XAE",
            "Nuclear Medicine Image Storage": "NM",
            "Secondary Capture Image Storage": "SC",
        }

        try:
            mode_prefix = mode_prefixes[dataset.SOPClassUID.name]
        except KeyError:
            mode_prefix = "UN"

        suid = dataset.SeriesInstanceUID
        series_dir = suid
        if self.output_directory is not None:
            series_dir = os.path.join(self.output_directory, suid)

        filename = "{0!s}.{1!s}".format(mode_prefix, dataset.SOPInstanceUID)

        if not os.path.exists(series_dir):
            os.mkdir(series_dir)

        filename = os.path.join(series_dir, filename)

        if os.path.exists(filename):
            logger.debug("DICOM file already exists, overwriting")

        context = event.context
        meta = Dataset()
        meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        meta.ImplementationClassUID = PYNETDICOM_IMPLEMENTATION_UID
        meta.TransferSyntaxUID = context.transfer_syntax

        # The following is not mandatory, set for convenience
        meta.ImplementationVersionName = PYNETDICOM_IMPLEMENTATION_VERSION
        file_ds = FileDataset(filename, {}, file_meta=meta, preamble=b"\0" * 128)
        file_ds.update(dataset)
        file_ds.is_little_endian = context.transfer_syntax.is_little_endian
        file_ds.is_implicit_VR = context.transfer_syntax.is_implicit_VR

        status_ds = Dataset()
        status_ds.Status = 0x0000

        try:
            # We use `write_like_original=False` to ensure that a compliant
            #   File Meta Information Header is written
            file_ds.save_as(filename, write_like_original=False)
            status_ds.Status = 0x0000  # Success
        except IOError:
            logger.warning("Could not write file to specified directory:")
            logger.warning("    {0!s}".format(os.path.dirname(filename)))
            logger.warning("Directory may not exist or you may not have write " "permission")
            # Failed - Out of Resources - IOError
            status_ds.Status = 0xA700
        except Exception as exception:
            logger.warning("Could not write file to specified directory:")
            logger.warning("    {0!s}".format(os.path.dirname(filename)))
            logger.warning(exception)
            # Failed - Out of Resources - Miscellaneous error
            status_ds.Status = 0xA701

        self.current_dir = series_dir

        return status_ds

    def send_dcm(self, dcm_file):
        """
        send_dcm accepts the location of a single dcm file to send over the connector or a list
        of dcm files to send in one association.
        """

        dcm_files = dcm_file
        if isinstance(dcm_file, str):
            dcm_files = [dcm_file]

        transfer_syntax = [ImplicitVRLittleEndian]

        ae = AE()

        for context in StoragePresentationContexts:
            ae.add_requested_context(context.abstract_syntax, transfer_syntax)

        if len(self.ae_title) > 0:
            assoc = ae.associate(self.host, self.port, ae_title=self.ae_title)
        else:
            assoc = ae.associate(self.host, self.port)

        status = ""
        if assoc.is_established:
            logger.debug("Sending file: {0!s}".format(dcm_file))

            for dcm_file in dcm_files:
                dataset = pydicom.read_file(dcm_file)
                status = assoc.send_c_store(dataset)

            # Release the association
            assoc.release()

        return status

    def on_c_echo(self, event):
        """Respond to a C-ECHO service request.

        Parameters
        ----------
        context : namedtuple
            The presentation context that the verification request was sent under.
        info : dict
            Information about the association and verification request.

        Returns
        -------
        status : int or pydicom.dataset.Dataset
            The status returned to the peer AE in the C-ECHO response. Must be
            a valid C-ECHO status value for the applicable Service Class as
            either an ``int`` or a ``Dataset`` object containing (at a
            minimum) a (0000,0900) *Status* element.
        """
        logger.debug("C-ECHO!")
        return 0x0000

    def on_association_accepted(self, event):
        self.current_dir = None

    def on_association_released(self, event):

        if self.recieved_callback and self.current_dir:
            self.recieved_callback(self.current_dir)

    def listen(self, recieved_callback, ae_title="PYNETDICOM"):

        self.recieved_callback = recieved_callback

        # Initialise the Application Entity and specify the listen port
        ae = AE(ae_title=ae_title)

        # Add the supported presentation context
        ae.add_supported_context(VerificationSOPClass)
        for context in StoragePresentationContexts:
            ae.add_supported_context(context.abstract_syntax)

        handlers = [
            (evt.EVT_C_MOVE, self.on_c_store),
            (evt.EVT_C_STORE, self.on_c_store),
            (evt.EVT_C_ECHO, self.on_c_echo),
            (evt.EVT_ACCEPTED, self.on_association_accepted),
            (evt.EVT_RELEASED, self.on_association_released),
        ]

        # Start listening for incoming association requests
        ae.start_server(("", self.port), evt_handlers=handlers)

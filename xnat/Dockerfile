FROM python:3.11-slim

WORKDIR /opt

# Install system dependencies 
RUN apt-get -y update && apt-get install -y \
    git \
    && apt-get clean

# Install platipy
RUN pip install --upgrade pip && \
    pip install git+https://github.com/tomaroberts/platipy.git@xnat-docker-image

# Copy Python script into container
COPY run_convert_rtstruct.py /usr/local/bin
RUN chmod 775 /usr/local/bin/run_convert_rtstruct.py

# Entrypoint arguments:
# 1) INPUT_DCM_DIRNAME
# 2) INPUT_RT_FILENAME
# 3) OUTPUT_NII_DIRNAME
# 4) OUTPUT_NII_FILENAME
# 5) OUTPUT_FILE_PREFIX
ENTRYPOINT ["python", "/usr/local/bin/run_convert_rtstruct.py"]

# command.json as LABEL
LABEL org.nrg.commands="[{\"name\": \"platipy RTSTRUCT to NIfTI\", \"label\": \"platipy RTSTRUCT to NIfTI\", \"description\": \"Convert RTSTRUCT files to NIfTI using platipy\", \"version\": \"1.0\", \"schema-version\": \"1.0\", \"info-url\": \"None\", \"image\": \"tomaroberts/platipy-xnat-rtstruct-to-nifti:latest\", \"type\": \"docker\", \"command-line\": \"python /usr/local/bin/run_convert_rtstruct.py /input-dcm-mount input-rtstuct-filename /output-nii-mount output-nii-filename\", \"override-entrypoint\": true, \"mounts\": [{\"name\": \"input-dcm-mount\", \"writable\": false, \"path\": \"/input-dcm-mount\"}, {\"name\": \"output-nii-mount\", \"writable\": true, \"path\": \"/output-nii-mount\"}], \"environment-variables\": {}, \"ports\": {}, \"inputs\": [{\"name\": \"nifti-img-filename\", \"description\": \"Filename given to output NIFTI image data, e.g. image-data.nii.gz\", \"type\": \"string\", \"default-value\": \"image-data.nii.gz\", \"required\": true, \"replacement-key\": \"output-nii-filename\", \"select-values\": []}], \"outputs\": [{\"name\": \"output-nifti\", \"description\": \"Folder containing output NIfTI files\", \"required\": true, \"mount\": \"output-nii-mount\", \"path\": null, \"glob\": null}], \"xnat\": [{\"name\": \"platipy rtstruct_to_nifti.convert_rtstruct\", \"label\": \"rtstruct_to_nifti\", \"description\": \"Run RTSRUCT to NIfTI conversion\", \"contexts\": [\"xnat:imageScanData\"], \"external-inputs\": [{\"name\": \"scan\", \"label\": \"Input scan\", \"type\": \"Scan\", \"matcher\": \"'DICOM' in @.resources[*].label\", \"required\": true, \"load-children\": true}], \"derived-inputs\": [{\"name\": \"scan-dicoms\", \"description\": \"The dicom resource on the scan\", \"type\": \"Resource\", \"matcher\": \"@.label == 'DICOM'\", \"required\": true, \"provides-files-for-command-mount\": \"input-dcm-mount\", \"load-children\": true, \"derived-from-wrapper-input\": \"scan\", \"multiple\": false}], \"output-handlers\": [{\"name\": \"nifti-resource\", \"accepts-command-output\": \"output-nifti-dir\", \"as-a-child-of\": \"scan\", \"type\": \"Resource\", \"label\": \"NIFTI\", \"tags\": []}]}], \"container-labels\": {}, \"generic-resources\": {}, \"ulimits\": {}}]"

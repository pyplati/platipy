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
LABEL org.nrg.commands="[{\"name\": \"platipy-rtstruct-to-nifti\", \"label\": \"platipy RTSTRUCT to NIfTI\", \"description\": \"Convert RTSTRUCT files to NIfTI using platipy\", \"version\": \"1.0\", \"schema-version\": \"1.0\", \"info-url\": \"None\", \"image\": \"tomaroberts/platipy-xnat-rtstruct-to-nifti:latest\", \"type\": \"docker\", \"workdir\": \"/opt\", \"command-line\": \"python /usr/local/bin/run_convert_rtstruct.py /input-dcm-mount /input-rt-mount/#RT_FILE_NAME# /output-nii-mount nifti-img-filename\", \"override-entrypoint\": true, \"mounts\": [{\"name\": \"input-dcm-mount\", \"writable\": false, \"path\": \"/input-dcm-mount\"}, {\"name\": \"input-rt-mount\", \"writable\": false, \"path\": \"/input-rt-mount\"}, {\"name\": \"output-nii-mount\", \"writable\": true, \"path\": \"/output-nii-mount\"}], \"environment-variables\": {}, \"ports\": {}, \"inputs\": [{\"name\": \"nifti-img-filename\", \"description\": \"Filename given to output NIFTI image data\", \"type\": \"string\", \"default-value\": \"image-data.nii.gz\", \"required\": true}], \"outputs\": [{\"name\": \"output-nifti\", \"description\": \"Folder containing output NIfTI files\", \"required\": true, \"mount\": \"output-nii-mount\"}], \"xnat\": [{\"name\": \"platipy-rtstruct_to_nifti.convert_rtstruct\", \"label\": \"rtstruct_to_nifti\", \"description\": \"Run RTSRUCT to NIfTI conversion on a Scan\", \"contexts\": [\"xnat:imageScanData\"], \"external-inputs\": [{\"name\": \"scan\", \"label\": \"Input scan\", \"type\": \"Scan\", \"required\": true}], \"derived-inputs\": [{\"name\": \"scan-dicoms\", \"description\": \"The DICOM resource on the Scan\", \"type\": \"Resource\", \"required\": true, \"derived-from-wrapper-input\": \"scan\", \"provides-files-for-command-mount\": \"input-dcm-mount\", \"matcher\": \"@.label == 'DICOM'\"}, {\"name\": \"scan-rtstruct\", \"description\": \"The RTSTRUCT resource on the Scan\", \"type\": \"Resource\", \"derived-from-wrapper-input\": \"scan\", \"provides-files-for-command-mount\": \"input-rt-mount\", \"matcher\": \"@.label == 'secondary'\"}, {\"name\": \"scan-rt-struct-file\", \"type\": \"File\", \"derived-from-wrapper-input\": \"scan-rtstruct\"}, {\"name\": \"scan-rt-struct-file-name\", \"type\": \"string\", \"derived-from-wrapper-input\": \"scan-rt-struct-file\", \"derived-from-xnat-object-property\": \"name\", \"provides-value-for-command-input\": \"RT_FILE_NAME\"}], \"output-handlers\": [{\"name\": \"nifti-resource\", \"type\": \"Resource\", \"accepts-command-output\": \"output-nifti\", \"as-a-child-of\": \"scan\", \"label\": \"NIFTI\"}]}]}]"

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

# TODO: fill in org.nrg.commands appropriately once command.json completed
# LABEL org.nrg.commands="[]"

# Entrypoint arguments:
# 1) INPUT_DCM_DIRNAME
# 2) INPUT_RT_FILENAME
# 3) OUTPUT_NII_DIRNAME
# 4) OUTPUT_NII_FILENAME
# 5) OUTPUT_FILE_PREFIX
ENTRYPOINT ["python", "/usr/local/bin/run_convert_rtstruct.py"]

FROM python:3.11-slim

WORKDIR /opt

# Install system dependencies 
RUN apt-get -y update && apt-get install -y \
    && apt-get clean

# Install platipy
RUN pip install --upgrade pip && \
    pip install platipy @ git+https://github.com/tomaroberts/platipy@xnat-docker-image

# Copy Python script into container
ADD xnat/run_convert_rtstruct.py /usr/local/bin
RUN chmod 775 /usr/local/bin/run_convert_rtstruct.py

# TODO: fill in org.nrg.commands appropriately once command.json completed
# LABEL org.nrg.commands="[]"

# TODO: ENTRYPOINT not required for XNAT Ready Docker Image
# ENTRYPOINT ["python", "run_convert_rtstruct.py", "", "", "", "", ""]

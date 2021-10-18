# Proximal Bronchial Tree Auto-Segmentation Service

A Docker image is provided to quickly spin up the PBT auto-segmentation service. The following command can be used to start the Docker container:

```
docker run -v ~/bronchus_data:/data -p 8010:8000 -p 7786:7777 -d --restart=always --name=bronchus_service platipy/platipy:bronchus
```

A few notes on this command:
- The ~/bronchus_data directory will be used to store data by the service
- Port 8010 is mapped to communicate with the service via HTTP
- Port 7786 is mapped to communicate with the service via DICOM
- ```--restart=always``` ensures that the container is always started even after a reboot of the 
system

If you find the PBT auto-segmentation work useful in your research, please cite:

> Ghandourh, W., Dowling, J., Chlap, P., Oar A., Jacob S., Batumalai, V., Holloway L. (2021). Assessing tumor centrality in lung stereotactic ablative body radiotherapy (SABR): the effects of variations in bronchial tree delineation and potential for automated methods. Medical Dosimetry 46(1). https://doi.org/10.1016/j.meddos.2020.09.004.


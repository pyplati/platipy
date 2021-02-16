# Cardiac Atlas Auto-Segmentation Service

A Docker image is provided to quickly spin up the Cardiac Auto-segementation service. The following
command can be used to start the Docker container:

```
docker run -v ~/cardiac_data:/data -v ~/cardiac_atlas:/atlas -p 8008:8000 -p 7785:7777 -d --restart=always --name=cardiac_service platipy/platipy:cardiac
```

A few notes on this command:
- The ~/cardiac_data directory will be used to store data by the service
- The ~/cardiac_atlas directory should contain the atlas images
- Port 8008 is mapped to communicate with the service via HTTP
- Port 7785 is mapped to communicate with the service via DICOM
- ```--restart=always``` ensures that the container is always started even after a reboot of the system

If you find the Cardiac Atlas auto-segmentation work useful in your research, please cite:

> Finnegan, R., Dowling, J., Koh, E.-S., Tang, S., Otton, J., Delaney, G., Batumalai, V., Luo, C., Atluri, P., Satchithanandha, A., Thwaites, D., Holloway, L. (2019). Feasibility of multi-atlas cardiac segmentation from thoracic planning CT in a probabilistic framework. Phys. Med. Biol. 64(8) 085006. https://doi.org/10.1088/1361-6560/ab0ea6

If you also use the code for iterative atlas selection, please cite:

> Finnegan, R., Lorenzen, E., Dowling, J., Holloway, L., Thwaites, D., Brink, C. (2020). Localised delineation uncertainty for iterative atlas selection in automatic cardiac segmentation. Phys. Med. Biol. 65(3) 035011. https://doi.org/10.1088/1361-6560/ab652a



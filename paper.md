---
title: 'PlatiPy: Processing Library and Analysis Toolkit for Medical Imaging in Python'
tags:
  - Python
  - medical image analysis
  - radiotherapy
  - visualisation
authors:
  - name: Phillip Chlap^[Co-first author]
    orcid: 0000-0002-6517-8745
    affiliation: "1, 2"
  - name: Robert N. Finnegan^[Co-first author]
    orcid: 0000-0000-0000-0000
    affiliation: "1, 3"
affiliations:
 - name: Ingham Institute for Applied Medical Research, Australia
   index: 1
 - name: South Western Sydney Clinical School, University of New South Wales, Australia
   index: 2
 - name: University of Sydney, Australia
   index: 3
date: 03 Jan 2023
bibliography: paper.bib
---

# Summary

PlatiPy provides a collection of tools and utilities to enable medical image analysis research using Python. This include functions to convert data to/from the commonly used DICOM format into more research friendly NIfTI format, functionality to assist image registration and atlas-based segmentation as well as tools to visualise images quickly and efficiently to enable rapid development of research. Auto-segmentation models developed within various research projects are also deployed as part of the library.

# Statement of need

Python has risen in popularity in medical image analysis research over recent years thanks to its simplicity and open-source nature, in particular the large community supported third party libraries. In particular libraries like SimpleITK [@Lowekamp2013; @Yaniv2018], scikit-learn [@JMLR:v12:pedregosa11a] and PyTorch [@NEURIPS2019_9015] provide a wide range of functionality to enable development of deep learning and atlas-based medical image analysis tools. While these are incredibly useful, researchers are often required to write code to prepare data for these tasks, write wrapper functions around library functions for common procedures and write code to visualise images throughout their pipeline. PlatiPy provides many of these functions removing the need for researchers to reinvent the wheel.

Tools resulting from such research projects often aren't easily usable outside of the context of that project. At times a GitHub repository is provided but can lack documentation and require installation of various dependencies which is not streamlined. By also incorporating these tools into PlatiPy and ensuring good software design principles are met, they can more easily be used in future research projects.

## Image Visualiser

Throughout any medical image analysis project it's useful to produce visualisations of medical images at all stages of the analysis pipeline from initial inspection of the dataset through to camera-ready figures presenting the results. We have observed several researchers avoiding visually inspecting their output often due to the significant boilerplate code needed to produce these using the matplotlib library directly. The Image Visualiser found in PlatiPy wraps this within an easy to use class which can display the cross sections of the medical images as well as overlay structures, scalar volumes and deformation vector fields. An example of such a visualisation is provided in \autoref{fig:vis_example}.

![Example of visualisations which can be produced using PlatiPy.\label{fig:vis_example}](figure_1.png)

This tool has proven useful across many projects and is capable of producing visualisations of results fit for publication, such as those in [@Finnegan2021; @Finnegan2022] *Ad Vicky C's paper once accepted, find out if anyone else has published using PlatiPy vis tools*.

## DICOM Conversion

The DICOM standard has is ubiquitous when working with medical images. A research dataset obtained through a clinic will almost certainly be provided in DICOM format. While a useful format in the clinical scenario, many researchers prefer other image formats better suited to develop source code around. One such format commonly used is NiFTI (Neuroimaging Informatics Technology Initiative). The SimpleITK library is capable of converting image volumes (such as CT and MRI) to this format fairly easily. However other modalities specific to radiotherapy required additional steps to convert. PlatiPy provides functions to convert structure sets (RTSTRUCT) to separate NiFTI masks (one per structure) as well as radiotherapy dose grids (RTDOSE) to a volume store in NiFTI format.

Converting these formats to NiFTI is desired for use in the research tool developed, but once output is generated is may be desirable to convert some out back to the DICOM format for analysis in a clinical system. PlatiPy also provides this functionality to generate DICOM Images, RTSTRUCT and RTDOSE files from NiFTI.

## Auto-segmentation

Several research projects that have utilised functionality in PlatiPy aim to develop an auto-segmentation model. This includes models based on thresholding, atlases and deep learning. Various tools to enable auto-segmentation are included in PlatiPy to perform pre- and post-processing, image registration and deep learning inference.

So far two auto-segmentation models developed using PlatiPy have then been deployed from use directly through the library. The first is a cardiac sub-structure auto-segmentation model which uses a deep learning components to segment the whole heart followed by an atlas-based segmentation along with geometric definitions to segment 17 cardiac sub-structures on radiotherapy CT images [refs Rob]. The other is a bronchial tree segmentation algorithm that uses threshold techniques to segment the lungs followed by the airways in radiotherapy lung CT images [@Ghandourh2021].

## Metric computation

Computing similarity metrics between an auto-segmentation and a reference segmentation is a common task when working in the medical image analysis space. Some libraries, such as SimpleITK, provide implementations for some common metrics. However having these implementations for all commonly used metrics in one place can be useful. PlatiPy supplies these functions as well as functionality to compute several metrics at one and produce a visualisation of the results [\autoref{fig:contour_comp_example}].

![Example of visualisation produced by the contour comparison tool.\label{fig:contour_comp_example}](figure_3.png)

PlatiPy also provides functionality compute Dose Volume Histograms (DVH) *ref* and extract certain metrics from these. Extracting metrics from these DVHs, another common task within Radiation Oncology and Medical Physics research. ***Provide examples of papers which have computed DVH metrics using PlatiPy***

## Synthetic Deformation Vector Field Generation

Some novel tools for generation of synthetic Deformation Vector Fields (DVFs) are provided within the PlatiPy library. These tools make use of structure masks to allow a user to manipulate an image by generating a deformation which expands, contract or shifts around the structure. These tools are applicable for the quality assurance of deformable image registration techniques as a ground truth deformation can be produced. Other applications include data augmentation for training deep learning auto-segmentation models or contour quality assurance models. Some work has also been done towards training a patient specific Generative Adversarial Network (GAN) using synthetically deformed images to track motion during head and neck cancer radiotherapy treatments [**ImageX paper ready to cite here?**].

***Figure showing synthetic deformation example***

# Acknowledgements

The authors would like to thank the Medical Physics groups at the Ingham Institute and University of Sydney who have put the PlatiPy library to use and who's research has driven forward the development of the library. We would also like to acknowledge the funding provided by the Australian Research Data Commons (ARDC) as part of the Australian Cancer Data Network (ACDN) grant. Finally we would like to thank Simon Biggs and the team at PyMedPhys [@Biggs2022] for inspiring the development of the PlatiPy library.

Data used in PlatiPy for examples, automated tests and figures in this paper was obtained from The Cancer Imaging Archive [@Clark2013].

# References

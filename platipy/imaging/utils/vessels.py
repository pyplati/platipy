import warnings

import numpy as np
import SimpleITK as sitk
import vtk

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def COMFromImageList(
    sitkImageList,
    conditionType="count",
    conditionValue=0,
    scanDirection="z",
    debug=False,
):
    """
    Input: list of SimpleITK images
           minimum total slice area required for the tube to be inserted at that slice
           scan direction: x = sagittal, y=coronal, z=axial
    Output: mean centre of mass positions, with shape (NumSlices, 2)
    Note: positions are converted into image space by default
    """
    if scanDirection.lower() == "x":
        if debug:
            print("Scanning in sagittal direction")
        COMZ = []
        COMY = []
        W = []
        C = []

        referenceImage = sitkImageList[0]
        referenceArray = sitk.GetArrayFromImage(referenceImage)
        z, y = np.mgrid[0 : referenceArray.shape[0] : 1, 0 : referenceArray.shape[1] : 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            for sitkImage in sitkImageList:
                volumeArray = sitk.GetArrayFromImage(sitkImage)
                comZ = 1.0 * (z[:, :, np.newaxis] * volumeArray).sum(axis=(1, 0))
                comY = 1.0 * (y[:, :, np.newaxis] * volumeArray).sum(axis=(1, 0))
                weights = np.sum(volumeArray, axis=(1, 0))
                W.append(weights)
                C.append(np.any(volumeArray, axis=(1, 0)))
                comZ /= 1.0 * weights
                comY /= 1.0 * weights
                COMZ.append(comZ)
                COMY.append(comY)

        with warnings.catch_warnings():
            """
            It's fairly likely some slices have just np.NaN values - it raises a warning but we can suppress it here
            """
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanCOMZ = np.nanmean(COMZ, axis=0)
            meanCOMY = np.nanmean(COMY, axis=0)
            if conditionType.lower() == "area":
                meanCOM = (
                    np.dstack((meanCOMZ, meanCOMY))[0]
                    * np.array((np.sum(W, axis=0) > (conditionValue),) * 2).T
                )
            elif conditionType.lower() == "count":
                meanCOM = (
                    np.dstack((meanCOMZ, meanCOMY))[0]
                    * np.array((np.sum(C, axis=0) > (conditionValue),) * 2).T
                )
            else:
                raise ValueError("Invalid condition type, please select from 'area' or 'count'.")

        pointArray = []
        for index, COM in enumerate(meanCOM):
            if np.all(np.isfinite(COM)):
                if np.all(COM > 0):
                    pointArray.append(
                        referenceImage.TransformIndexToPhysicalPoint(
                            (index, int(COM[1]), int(COM[0]))
                        )
                    )

        return pointArray

    elif scanDirection.lower() == "z":
        if debug:
            print("Scanning in axial direction")
        COMX = []
        COMY = []
        W = []
        C = []

        referenceImage = sitkImageList[0]
        referenceArray = sitk.GetArrayFromImage(referenceImage)
        x, y = np.mgrid[0 : referenceArray.shape[1] : 1, 0 : referenceArray.shape[2] : 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            for sitkImage in sitkImageList:
                volumeArray = sitk.GetArrayFromImage(sitkImage)
                comX = 1.0 * (x * volumeArray).sum(axis=(1, 2))
                comY = 1.0 * (y * volumeArray).sum(axis=(1, 2))
                weights = np.sum(volumeArray, axis=(1, 2))
                W.append(weights)
                C.append(np.any(volumeArray, axis=(1, 2)))
                comX /= 1.0 * weights
                comY /= 1.0 * weights
                COMX.append(comX)
                COMY.append(comY)

        with warnings.catch_warnings():
            """
            It's fairly likely some slices have just np.NaN values - it raises a warning but we
            can suppress it here
            """
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanCOMX = np.nanmean(COMX, axis=0)
            meanCOMY = np.nanmean(COMY, axis=0)
            if conditionType.lower() == "area":
                meanCOM = (
                    np.dstack((meanCOMX, meanCOMY))[0]
                    * np.array((np.sum(W, axis=0) > (conditionValue),) * 2).T
                )
            elif conditionType.lower() == "count":
                meanCOM = (
                    np.dstack((meanCOMX, meanCOMY))[0]
                    * np.array((np.sum(C, axis=0) > (conditionValue),) * 2).T
                )
            else:
                print("Invalid condition type, please select from 'area' or 'count'.")
                quit()
        pointArray = []
        for index, COM in enumerate(meanCOM):
            if np.all(np.isfinite(COM)):
                if np.all(COM > 0):
                    pointArray.append(
                        referenceImage.TransformIndexToPhysicalPoint(
                            (int(COM[1]), int(COM[0]), index)
                        )
                    )

        return pointArray


def tubeFromCOMList(COMList, radius, debug=False):
    """
    Input: image-space positions along the tube centreline.
    Output: VTK tube
    Note: positions do not have to be continuous - the tube is interpolated in real space
    """
    points = vtk.vtkPoints()
    for i, pt in enumerate(COMList):
        points.InsertPoint(i, pt[0], pt[1], pt[2])

    # Fit a spline to the points
    if debug:
        print("Fitting spline")
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(10 * points.GetNumberOfPoints())
    functionSource.Update()

    # Generate the radius scalars
    tubeRadius = vtk.vtkDoubleArray()
    n = functionSource.GetOutput().GetNumberOfPoints()
    tubeRadius.SetNumberOfTuples(n)
    tubeRadius.SetName("TubeRadius")
    for i in range(n):
        # We can set the radius based on the given propagated segmentations in that slice?
        # Typically segmentations are elliptical, this could be an issue so for now a constant
        # radius is used
        tubeRadius.SetTuple1(i, radius)

    # Add the scalars to the polydata
    tubePolyData = vtk.vtkPolyData()
    tubePolyData = functionSource.GetOutput()
    tubePolyData.GetPointData().AddArray(tubeRadius)
    tubePolyData.GetPointData().SetActiveScalars("TubeRadius")

    # Create the tubes
    tuber = vtk.vtkTubeFilter()
    tuber.SetInputData(tubePolyData)
    tuber.SetNumberOfSides(50)
    tuber.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tuber.Update()

    return tuber


def writeVTKTubeToFile(tube, filename):
    """
    Input: VTK tube
    Output: exit success
    Note: format is XML VTP
    """
    print("Writing tube to polydata file (VTP)")
    polyDataWriter = vtk.vtkXMLPolyDataWriter()
    polyDataWriter.SetInputData(tube.GetOutput())

    polyDataWriter.SetFileName(filename)
    polyDataWriter.SetCompressorTypeToNone()
    polyDataWriter.SetDataModeToAscii()
    s = polyDataWriter.Write()

    return s


def SimpleITKImageFromVTKTube(tube, SITKReferenceImage, debug=False):
    """
    Input: VTK tube, referenceImage (used for spacing, etc.)
    Output: SimpleITK image
    Note: Uses binary output (background 0, foreground 1)
    """
    size = list(SITKReferenceImage.GetSize())
    origin = list(SITKReferenceImage.GetOrigin())
    spacing = list(SITKReferenceImage.GetSpacing())
    ncomp = SITKReferenceImage.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    arr = sitk.GetArrayFromImage(SITKReferenceImage).transpose(2, 1, 0).flatten()

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()

    dataImporter.CopyImportVoidPointer(arr, len(arr))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(ncomp)

    # Set the new VTK image's parameters
    dataImporter.SetDataExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)

    dataImporter.Update()

    VTKReferenceImage = dataImporter.GetOutput()

    # fill the image with foreground voxels:
    inval = 1
    outval = 0
    VTKReferenceImage.GetPointData().GetScalars().Fill(inval)

    if debug:
        print("Using polydaya to generate stencil.")
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetTolerance(0.5)  # points within 0.5 voxels are included
    pol2stenc.SetInputConnection(tube.GetOutputPort())
    pol2stenc.SetOutputOrigin(VTKReferenceImage.GetOrigin())
    pol2stenc.SetOutputSpacing(VTKReferenceImage.GetSpacing())
    pol2stenc.SetOutputWholeExtent(VTKReferenceImage.GetExtent())
    pol2stenc.Update()

    if debug:
        print("using stencil to generate image.")
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(VTKReferenceImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    if debug:
        print("Generating SimpleITK image.")
    finalImage = imgstenc.GetOutput()
    finalArray = finalImage.GetPointData().GetScalars()
    finalArray = vtk_to_numpy(finalArray).reshape(SITKReferenceImage.GetSize()[::-1])
    if debug:
        print(f"Volume = {finalArray.sum()*sum(spacing):.3f} mm^3")
    finalImageSITK = sitk.GetImageFromArray(finalArray)
    finalImageSITK.CopyInformation(SITKReferenceImage)

    return finalImageSITK


def ConvertSimpleITKtoVTK(img):
    """"""
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0).flatten()
    arr_string = arr.tostring()

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()

    dataImporter.CopyImportVoidPointer(arr_string, len(arr_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(ncomp)

    # Set the new VTK image's parameters
    dataImporter.SetDataExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)

    dataImporter.Update()

    vtk_image = dataImporter.GetOutput()
    return vtk_image


def vesselSplineGeneration(
    referenceImage,
    atlasSet,
    vesselNameList,
    vesselRadiusDict,
    stopConditionTypeDict,
    stopConditionValueDict,
    scanDirectionDict,
    debug=False,
    atlas_label="DIR",
):
    """"""
    splinedVessels = {}

    if isinstance(vesselNameList, str):
        vesselNameList = [vesselNameList]

    for vesselName in vesselNameList:

        # We must set the image direction to identity
        # This is because it is not possible to modify VTK Image directions
        # This may get fixed in a future VTK version

        initial_image_direction = referenceImage.GetDirection()

        imageList = [atlasSet[i][atlas_label][vesselName] for i in atlasSet.keys()]
        for im in imageList:
            im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        vesselRadius = vesselRadiusDict[vesselName]
        stopConditionType = stopConditionTypeDict[vesselName]
        stopConditionValue = stopConditionValueDict[vesselName]
        scanDirection = scanDirectionDict[vesselName]

        pointArray = COMFromImageList(
            imageList,
            conditionType=stopConditionType,
            conditionValue=stopConditionValue,
            scanDirection=scanDirection,
            debug=debug,
        )
        tube = tubeFromCOMList(pointArray, radius=vesselRadius, debug=debug)

        SITKReferenceImage = imageList[0]

        vessel_delineation = SimpleITKImageFromVTKTube(tube, SITKReferenceImage, debug=debug)

        vessel_delineation.SetDirection(initial_image_direction)

        splinedVessels[vesselName] = vessel_delineation

        # We also have to reset the direction to whatever it was
        # This is because SimpleITK doesn't use deep copying
        # And it isn't necessary here as we can save some sweet, sweet memory
        for im in imageList:
            im.SetDirection(initial_image_direction)

    return splinedVessels
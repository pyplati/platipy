#!/usr/bin/env python
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon
import numpy as np
import os
import sys
import shutil

def determineClass(dicomFileName):
    f = pydicom.read_file(dicomFileName, force='True')
    SOPClass = f.SOPClassUID
    SOPName  = pydicom._uid_dict.UID_dictionary[SOPClass][0]

    if SOPName == 'RT Dose Storage':
        return 'dose',''
    elif SOPName == 'RT Structure Set Storage':
        return 'rtstruct',''
    elif SOPName == 'CT Image Storage':
        return 'image', '_ct'
    elif SOPName == 'MR Image Storage':
        return 'image', '_mr'
    elif SOPName == 'Positron Emission Tomography Image Storage':
        return 'image', '_pet'
    else:
        return SOPName,''

def determineSliceLocation(dicomFileName, dicomDir=None):
    try:
        f = pydicom.read_file(dicomFileName, force='True')
        return float(f.SliceLocation)
    except:
        # No slice location - sort based on dicom directory
        dicom_files = sorted(os.listdir(dicomDir))
        local_filename = dicomFileName[::-1][:dicomFileName[::-1].find('/')][::-1]
        return np.where(np.array(dicom_files)==local_filename)[0][0]



def readDICOMImageFromList(s_img_list):
    return sitk.ReadImage(s_img_list)

def readDICOMStructFile(filename):
    dicomStructFile = pydicom.read_file(filename, force=True)
    return dicomStructFile

def fixMissingData(SS):
    contourData = np.array(SS)
    if contourData.any()=='':
        print("Missing values detected.")
        missingVals = np.where(contourData=='')[0]
        if missingVals.shape[0]>1:
            print("More than one value missing, fixing this isn't implemented yet...")
        else:
            print("Only one value missing.")
            missingIndex = missingVals[0]
            missingAxis = missingIndex%3
            if missingAxis==0:
                print("Missing value in x axis: interpolating.")
                if missingIndex>len(contourData)-3:
                    lowerVal = contourData[missingIndex-3]
                    upperVal = contourData[0]
                elif missingIndex==0:
                    lowerVal = contourData[-3]
                    upperVal = contourData[3]
                else:
                    lowerVal = contourData[missingIndex-3]
                    upperVal = contourData[missingIndex+3]
                contourData[missingIndex] = 0.5*(lowerVal+upperVal)
            elif missingAxis==1:
                print("Missing value in y axis: interpolating.")
                if missingIndex>len(contourData)-2:
                    lowerVal = contourData[missingIndex-3]
                    upperVal = contourData[1]
                elif missingIndex==0:
                    lowerVal = contourData[-2]
                    upperVal = contourData[4]
                else:
                    lowerVal = contourData[missingIndex-3]
                    upperVal = contourData[missingIndex+3]
                contourData[missingIndex] = 0.5*(lowerVal+upperVal)
            else:
                print("Missing value in z axis: taking slice value")
                temp = contourData[2::3].tolist()
                temp.remove('')
                contourData[missingIndex] = np.min(np.array(temp, dtype=np.double))
    return contourData

def transformPointSetFromDICOMStruct(DICOMImage, DICOMStruct, writeImage, imageOutputName, spacingOverride):

    if spacingOverride:
        currentSpacing = list(DICOMImage.GetSpacing())
        newSpacing = tuple([currentSpacing[k] if spacingOverride[k]==0 else spacingOverride[k] for k in range(3)])
        DICOMImage.SetSpacing(newSpacing)

    transformPhysicalPointToIndex = DICOMImage.TransformPhysicalPointToIndex

    structPointSequence = DICOMStruct.ROIContourSequence
    structNameSequence = ['_'.join(i.ROIName.split()) for i in DICOMStruct.StructureSetROISequence]

    structList = []
    finalStructNameSequence = []

    for structIndex, structName in enumerate(structNameSequence):
        imageBlank = np.zeros(DICOMImage.GetSize()[::-1], dtype=np.uint8)
        print("Converting structure {0} with name: {1}".format(structIndex, structName))

        if not hasattr(structPointSequence[structIndex], "ContourSequence"):
            print("No contour sequence found for this structure, skipping.")
            continue

        if not structPointSequence[structIndex].ContourSequence[0].ContourGeometricType=="CLOSED_PLANAR":
            print("This is not a closed planar structure, skipping.")
            continue

        for sl in range(len(structPointSequence[structIndex].ContourSequence)):

            contourData = fixMissingData(structPointSequence[structIndex].ContourSequence[sl].ContourData)

            structSliceContourData = np.array(contourData, dtype=np.double)
            vertexArr_physical = structSliceContourData.reshape(structSliceContourData.shape[0]//3,3)

            pointArr = np.array([transformPhysicalPointToIndex(i) for i in vertexArr_physical]).T

            [xVertexArr_image, yVertexArr_image] = pointArr[[0,1]]
            zIndex = pointArr[2][0]

            if np.any(pointArr[2]!=zIndex):
                print("Error: axial slice index varies in contour. Quitting now.")
                print("Structure:   {0}".format(structName))
                print("Slice index: {0}".format(zIndex))
                quit()

            if zIndex>=DICOMImage.GetSize()[2]:
                print("Warning: Slice index greater than image size. Skipping slice.")
                print("Structure:   {0}".format(structName))
                print("Slice index: {0}".format(zIndex))
                continue

            sliceArr = np.zeros(DICOMImage.GetSize()[:2], dtype=np.uint8)
            filledIndicesX, filledIndicesY = polygon(xVertexArr_image, yVertexArr_image, shape=sliceArr.shape)
            sliceArr[filledIndicesX, filledIndicesY] = 1
            # imageBlank[zIndex] += sliceArr
            imageBlank[zIndex] += sliceArr.T

        structImage = sitk.GetImageFromArray(1*(imageBlank>0))
        structImage.CopyInformation(DICOMImage)
        structList.append(sitk.Cast(structImage, sitk.sitkUInt8))
        finalStructNameSequence.append(structName)

    if writeImage:
        print("Saving DICOM image.")
        sitk.WriteImage(DICOMImage, imageOutputName)

    return structList, finalStructNameSequence


if __name__ == '__main__':
    if len(sys.argv)<4:
        print("Read DICOM RT directory, saving any images, structures and dose files as Nifti images.")
        print("Arguments:")
        print("   1: DICOM image directory")
        print("   2: Output location")
        print("      e.g. ../Case_1234/")
        print("   3: Output basename")
        print("      e.g. Case_1234")
        print("   4: (Optional) reverse scan direction.")
        print("      1 = True, 0 [Default] = False")
        print("   5: (Optional) DICOM image directory.")
        print("      Required if trying to convert just an RT Struct")
        sys.exit()
    else:
        DICOMDir = sys.argv[1]
        outputDir = sys.argv[2]
        outputBasename = sys.argv[3]

        try:
            reversedFlag = {'1':True,'0':False}[sys.argv[4]]
        except:
            reversedFlag = False

        if len(sys.argv)>5:
            if sys.argv[5]:
                dicom_image_dir = sys.argv[5]
            else:
                dicom_image_dir = False

        imageOutputName = "{0}/Images/{1}{2}.nii.gz".format(outputDir, outputBasename, "{0}")

        writeImage = True

        try:
            os.mkdir(outputDir)
        except:
            0
        try:
            os.mkdir(outputDir+"/Images")
            os.mkdir(outputDir+"/Structures")
            os.mkdir(outputDir+"/Doses")
        except:
            0

        fileList = sorted([i for i in os.listdir(DICOMDir) if i[-4:]=='.dcm'])
        image = []
        imageLocs = []
        struct = []
        dose = []
        other = 0
        for f in fileList:
            c, subclass = determineClass("{0}/{1}".format(DICOMDir, f))
            if c=='image':
                image.append("{0}/{1}".format(DICOMDir, f))
                sLoc = determineSliceLocation("{0}/{1}".format(DICOMDir, f), DICOMDir)
                imageLocs.append(sLoc)

            elif c=='rtstruct':
                struct.append("{0}/{1}".format(DICOMDir, f))
            elif c=='dose':
                dose.append("{0}/{1}".format(DICOMDir, f))
            else:
                other+=1
                print("SOP class : {0}, saving in new directory.".format(c))
                cName = ''.join(c.split())
                try:
                    os.mkdir(outputDir+"/"+cName)
                except:
                    0
                shutil.copy2("{0}/{1}".format(DICOMDir, f), "{0}/{1}/{2}".format(outputDir, cName, f))

        print('--------------------------------------')
        print('Summary of files found')
        print('--------------------------------------')
        print('Image files:        {0}'.format(len(image)))
        print('RT Structure files: {0}'.format(len(struct)))
        print('Dose files:         {0}'.format(len(dose)))
        print('Other files:        {0}'.format(other))
        print('--------------------------------------')

        occupiedSuffices = {}
        for index, d in enumerate(dose):
            dicomDoseImage = pydicom.read_file(d, force=True)
            itkDoseImage = readDICOMImageFromList(d)

            itkDoseImage = sitk.Cast(itkDoseImage, sitk.sitkFloat32)*float(dicomDoseImage.DoseGridScaling)

            doseType = dicomDoseImage.DoseSummationType
            print("Dose type: {0}, File: {1}".format(doseType, d))

            if len(dose)==1:
                suffix = '_{0}'.format(doseType)
            elif len(dose)>1:
                if doseType=='BEAM':
                    beamNumber = dicomDoseImage.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber
                    suffix = '_BEAM_{0}'.format(beamNumber)
                elif doseType=='FRACTION':
                    fraction = dicomDoseImage.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedFractionGroupNumber
                    suffix = '_FRACTION_{0}'.format(fraction)
                else:
                    if doseType not in occupiedSuffices.keys():
                        suffix = '_{0}'.format(doseType)
                        occupiedSuffices[doseType] = 2
                    elif doseType in occupiedSuffices.keys():
                        suffix = '_{0}_{1}'.format(doseType, occupiedSuffices[doseType])
                        occupiedSuffices[doseType] += 1

            sitk.WriteImage(itkDoseImage, "{0}/Doses/{1}_DOSE{2}.nii.gz".format(outputDir, outputBasename, suffix))

        image = [i for _,i in sorted(zip(imageLocs, image), reverse=reversedFlag)]

        # Sometimes there won't be any images, in this case we need to specify a dicom image directory
        if len(image)==0 and not dicom_image_dir:
            print("No DICOM image found.")
            print("Please enter DICOM image directory in command.")
            print("Exiting.")
            sys.exit()
        elif len(image)==0 and dicom_image_dir:
            print("No DICOM image found.")
            print("You have entered a DICOM image directory.")
            image = [f'{dicom_image_dir}/{i}' for i in os.listdir(dicom_image_dir) if i[-4:]=='.dcm']
            imageLocs = [determineSliceLocation(f'{dicom_image_dir}/{i}', dicom_image_dir) for i in os.listdir(dicom_image_dir) if i[-4:]=='.dcm']
            image = [i for _,i in sorted(zip(imageLocs, image), reverse=reversedFlag)]

        DICOMImageFile = readDICOMImageFromList(image)
        sitk.WriteImage(DICOMImageFile, imageOutputName.format(subclass.upper()))
        for index, s in enumerate(struct):
            if len(struct)==1:
                suffix = ''
            else:
                suffix = '_{0}'.format(index)

            DICOMStructFile = readDICOMStructFile(s)
            structList, structNameSequence = transformPointSetFromDICOMStruct(DICOMImageFile, DICOMStructFile, writeImage, imageOutputName, False)
            print("DICOM image has been written to disk, it will not be re-written.")
            writeImage = False
            print("Converted all structures. Writing output.")
            for structIndex, structImage in enumerate(structList):
                cleanStructName = structNameSequence[structIndex].upper().replace(':','_')
                cleanStructName = cleanStructName.replace('/','_')
                cleanStructName = cleanStructName.replace(';','_')
                outName = "{0}/Structures/{1}_{2}{3}.nii.gz".format(outputDir,outputBasename,cleanStructName,suffix)
                print("Writing file to: {0}".format(outName))
                try:
                    sitk.WriteImage(structImage, outName)
                except:
                    print("Cannot write file: {0}".format(outName))

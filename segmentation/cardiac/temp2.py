ids = ['FF22']
imList = [sitk.ReadImage(f'../../TempCardiacData/Case_{i}/Structures/Case_{i}_LAD.nii.gz') for i in ids]
pointArray = COMFromImageList(imList)
tube       = tubeFromCOMList(pointArray, radius=3)

atlasSet={'0':{'DIR':{'LAD':imList[0]}}}

vesselRadius        = 3
stopConditionType   = 'counts'
stopConditionValue  = 0
scanDirection       = 'z'

pointArray = COMFromImageList(imageList, conditionType=stopConditionType, conditionValue=stopConditionValue, scanDirection=scanDirection)
tube       = tubeFromCOMList(pointArray, radius=vesselRadius)

SITKReferenceImage  = imageList[0]
VTKReferenceImage   = ConvertSimpleITKtoVTK(SITKReferenceImage)

#splinedVessels[vesselName] = SimpleITKImageFromVTKTube(tube, VTKReferenceImage, SITKReferenceImage, verbose = False)

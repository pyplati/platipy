"""
Provides custom radiomics for radiomics service
"""
import numpy

from radiomics import base


class RadiomicsCustom(base.RadiomicsFeaturesBase):
    """
    Custom Radiomics Class
    """

    # def __init__(self, inputImage, inputMask, **kwargs):
    #     super(RadiomicsCustom, self).__init__(inputImage, inputMask, **kwargs)

    # self.pixelSpacing = inputImage.GetSpacing()
    # self.voxelArrayShift = kwargs.get('voxelArrayShift', 0)

    # def _initCalculation(self, voxelCoordinates=None):
    #     super(RadiomicsCustom, self)._initCalculation(voxelCoordinates)

    #     # self.target_voxel_array = self.imageArray[voxelCoordinates].astype("float")
    #     self.logger.debug("Custom feature class initialized")

    def get25PercentileFeatureValue(self):

        return numpy.percentile(self.imageArray[self.maskArray], 25)

    def get75PercentileFeatureValue(self):

        return numpy.percentile(self.imageArray[self.maskArray], 75)

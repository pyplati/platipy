# Copyright 2022 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import SimpleITK as sitk

from scipy.stats import norm

from platipy.imaging.generation.dvf import (
    generate_field_asymmetric_contract,
    generate_field_asymmetric_extend,
    generate_field_expand,
    generate_field_radial_bend,
    generate_field_shift,
)


class ShiftModel:
    """Class to represent a model of structure-based shift"""

    def __init__(self, mask, model, shift_loc, shift_scale):

        self.deformation_type = "SHIFT"

        self.__mask = mask
        self.__model = model
        self.__shift_loc = shift_loc
        self.__shift_scale = shift_scale

        if isinstance(self.__model, list):
            self.model_type = "LIST"
        elif isinstance(model, str):
            self.model_type = model.upper()

    def generate_sample(self):
        rng = np.random.default_rng()

        if self.model_type == "LIST":
            return rng.choice(self.__model)

        elif self.model_type == "NORMAL":
            return rng.normal(loc=self.__shift_loc, scale=self.__shift_scale)

        elif self.model_type == "FIXED":
            return self.__shift_loc

    def generate_unit_field(self):
        sitk_unit_field = generate_field_shift(self.__mask, vector_shift=(10, 10, 10))
        self.unit_field = sitk.GetArrayFromImage(sitk_unit_field)


class ExpansionModel:
    """
    Class to represent a model of structure-based expansion
    Note: also includes contraction, and expansion + contraction
    """

    def __init__(
        self,
        mask,
        model,
        expansion_loc,
        expansion_scale,
        bone_mask=None,
        internal_deformation=False,
    ):

        self.deformation_type = "EXPANSION"

        self.__mask = mask
        self.__model = model
        self.__expansion_loc = expansion_loc
        self.__expansion_scale = expansion_scale
        self.__bone_mask = bone_mask
        self.__use_internal_deformation = use_internal_deformation

        if isinstance(self.__model, list):
            self.model_type = "LIST"
        elif isinstance(self.__model, str):
            self.model_type = self.__model.upper()

    def generate_sample(self):
        rng = np.random.default_rng()

        if self.model_type == "LIST":
            return rng.choice(self.__model)

        elif self.model_type == "NORMAL":
            return rng.normal(loc=self.__expansion_loc, scale=self.__expansion_scale)

        elif self.model_type == "FIXED":
            return self.__expansion_loc

    def generate_unit_field(self):
        sitk_unit_field = generate_field_shift(self.__mask, vector_shift=(10, 10, 10))
        self.unit_field = sitk.GetArrayFromImage(sitk_unit_field)


def find_dir_parameters(img, scale_mm, num_stages):
    regularisation_kernel_scale = scale_mm

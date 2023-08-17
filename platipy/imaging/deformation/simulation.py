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

from platipy.imaging.generation.dvf import (
    generate_field_asymmetric_contract,
    generate_field_asymmetric_extend,
    generate_field_expand,
    generate_field_radial_bend,
    generate_field_shift,
)

from platipy.imaging.projects.deformation.utils import ShiftModel


class DeformationSimulation:
    def __init__(
        self,
        image,
    ):

        self.__image = image
        self.__deformation_models = []

    def add_shift_model(
        self,
        structure,
        model,
        shift_loc=None,
        shift_scale=None,
    ):

        shift_model = ShiftModel(
            mask=structure, model=model, shift_loc=shift_loc, shift_scale=shift_scale
        )

        # generate "unit vector" of shift
        # then multiply by the sampled number
        # this would be much quicker
        shift_model.generate_unit_field()

        self.__deformation_models.append(shift_model)

    def add_bending_deformation():
        None

    def add_random_field_deformation(self, ext_mask=None):

        None

    def add_PCA_field_deformation(self, PCAFieldModel):
        None

    def generate_samples(self, num_samples="all"):

        samples_combined = []

        print(num_samples)

        if isinstance(num_samples, str):
            if num_samples.upper() == "ALL":

                list_models = [
                    def_model
                    for def_model in self.__deformation_models
                    if def_model.model_type == "LIST"
                ]

                # need to get all samples / do a combination of list_models
                # TODO

                for def_model in list_models:
                    params = def_model.generate_sample()
                    print(def_model.model_type, param)

                    param_mm = [p / 10 for p in param]

                    dvf = np.multiply(def_model.unit_field, param_mm)

                    # smooth
                    # would need to add up all the vector fields and then smooth

                    samples_individual.append(dvf)

                samples_combined.append(sum(samples_individual))

        else:
            for n in range(num_samples):

                samples_individual = []

                for def_model in self.__deformation_models:
                    param = def_model.generate_sample()
                    print(def_model.model_type, param)

                    param_mm = [p / 10 for p in param]

                    print("param_mm", param_mm)

                    print(def_model.unit_field)

                    return def_model

                    dvf = np.multiply(def_model.unit_field, param_mm)

                    samples_individual.append(dvf)

                samples_combined.append(sum(samples_individual))

        return samples_combined

    def generator(self):
        None

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

import zipfile
import tempfile
import urllib.request
from pathlib import Path

def download_and_extract_zip_file(zip_url, output_directory):
    """Downloads a zip file from a URL and extracts its contents to the supplied output directory.

    Args:
        zip_url (str): The URL of the zip file to download.
        output_directory (str|pathlib.Path): The directory in whcih to extract the zip file.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir).joinpath("tmp.zip")

        with urllib.request.urlopen(zip_url) as dl_file:
            with open(temp_file, "wb") as out_file:
                out_file.write(dl_file.read())

        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(output_directory)

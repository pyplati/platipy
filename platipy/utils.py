import zipfile
import tempfile
import urllib.request
from pathlib import Path

def download_and_extract_zip_file(zip_url, output_directory):

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir).joinpath("tmp.zip")

        with urllib.request.urlopen(zip_url) as dl_file:
            with open(temp_file, "wb") as out_file:
                out_file.write(dl_file.read())

        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(output_directory)

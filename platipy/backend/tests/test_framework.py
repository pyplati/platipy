# Copyright 2020 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import os
import json
import tempfile
from platipy.backend import app


@pytest.fixture
def client():
    return app.test_client()


@pytest.mark.skip("Test needs to be updated")
def test_status(client):
    rv = client.get("/")
    assert b"Status" in rv.data


@pytest.mark.skip("Test needs to be updated")
def test_add_endpoint_page(client):
    rv = client.get("/endpoint/add")
    assert b"Add Dicom Endpoint" in rv.data


@pytest.mark.skip("Test needs to be updated")
def test_add_endpoint(client):

    post_data = dict(
        endpointName="test123",
        endpointAlgorithm="myAlgorithm",
        endpointType="listener",
        fromPort="7777",
        toHost="localhost",
        toPort="4242",
        toAETitle="aetTest",
        settings='{"testSetting": 123}',
    )
    rv = client.post("/endpoint/add", data=post_data)

    data_file = os.path.join(web_app.working_dir, "data.json")
    assert os.path.exists(data_file)
    with open(data_file) as f:
        data = json.load(f)
        print(data)
        assert data["endpoints"][0]["endpointName"] == "test123"
        assert data["endpoints"][0]["endpointAlgorithm"] == "myAlgorithm"
        assert data["endpoints"][0]["endpointType"] == "listener"
        assert data["endpoints"][0]["fromPort"] == "7777"
        assert data["endpoints"][0]["toHost"] == "localhost"
        assert data["endpoints"][0]["toPort"] == "4242"
        assert data["endpoints"][0]["toAETitle"] == "aetTest"
        assert data["endpoints"][0]["settings"]["testSetting"] == 123


@pytest.mark.skip("Test needs to be updated")
def test_view_endpoint(client):

    rv = client.get("/endpoint/0")

    print(rv.data)
    assert 0

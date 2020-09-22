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

from platipy.backend import db
import datetime
import json

from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import DeclarativeMeta

# This custom encoder takes care of converting our SQLAlchemy models to a JSON
# encodable format.
class AlchemyEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj.__class__, DeclarativeMeta):
            # an SQLAlchemy class
            fields = {}
            for field in [
                x for x in dir(obj) if not x.startswith("_") and x != "metadata"
            ]:
                data = obj.__getattribute__(field)

                try:
                    # this will fail on non-encodable values, like other classes
                    json.dumps(data)
                    fields[field] = data
                except TypeError:
                    if isinstance(data, datetime.datetime):
                        r = data.isoformat()

                        if data.microsecond:
                            r = r[:23] + r[26:]
                        if r.endswith("+00:00"):
                            r = r[:-6] + "Z"
                        fields[field] = r
                    elif isinstance(data, datetime.date):
                        fields[field] = data.isoformat()
                    elif isinstance(data, datetime.time):
                        if is_aware(data):
                            raise ValueError(
                                "JSON can't represent timezone-aware times."
                            )
                        r = data.isoformat()
                        if data.microsecond:
                            r = r[:12]
                        fields[field] = r
                    elif isinstance(data, DicomLocation):
                        fields[field] = self.default(data)
                    elif isinstance(data, list):
                        o = []
                        for d in data:
                            o.append(self.default(d))
                        fields[field] = o

            # a json-encodable dict
            return fields

        return json.JSONEncoder.default(self, obj)


def default_timeout():

    return datetime.datetime.utcnow() + datetime.timedelta(hours=24)


class APIKey(db.Model):
    __tablename__ = "APIKey"

    key = db.Column(db.String(80), primary_key=True)
    name = db.Column(db.String(80))

    is_admin = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return "{0}: {1} (Admin: {2})".format(self.name, self.key, self.is_admin)


class DicomLocation(db.Model):
    __tablename__ = "DicomLocation"

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(128))

    host = db.Column(db.String(128), nullable=False)
    port = db.Column(db.Integer, nullable=False)
    ae_title = db.Column(db.String(128))

    owner_key = db.Column(db.String(80), db.ForeignKey("APIKey.key"), nullable=False)

    def __repr__(self):
        return "{0} {1} {2}".format(self.host, self.port, self.ae_title)


class Dataset(db.Model):
    __tablename__ = "Dataset"

    id = db.Column(db.Integer, primary_key=True)
    owner_key = db.Column(db.String(80), db.ForeignKey("APIKey.key"), nullable=False)

    input_data_objects = relationship(
        "DataObject",
        primaryjoin="and_(DataObject.dataset_id == Dataset.id, DataObject.is_input == True)",
    )
    output_data_objects = relationship(
        "DataObject",
        primaryjoin="and_(DataObject.dataset_id == Dataset.id, DataObject.is_input == False)",
    )

    # The Dicom location from which to retrieve data
    from_dicom_location_id = db.Column(db.Integer, db.ForeignKey("DicomLocation.id"))
    from_dicom_location = relationship(
        "DicomLocation", foreign_keys=[from_dicom_location_id]
    )

    # The Dicom location to send data to
    to_dicom_location_id = db.Column(db.Integer, db.ForeignKey("DicomLocation.id"))
    to_dicom_location = relationship(
        "DicomLocation", foreign_keys=[to_dicom_location_id]
    )

    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    timeout = db.Column(db.DateTime, nullable=False, default=default_timeout)

    def __repr__(self):
        return "{0}: {1}".format(self.id, self.timestamp)


class DataObject(db.Model):
    __tablename__ = "DataObject"

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("Dataset.id"), nullable=False)
    dataset = relationship("Dataset")
    is_input = db.Column(db.Boolean, default=False)

    path = db.Column(db.String(256))

    type = db.Column(db.String(32), nullable=False, default="FILE")

    series_instance_uid = db.Column(db.String(256))

    meta_data = db.Column(db.JSON)
    status = db.Column(db.JSON)

    is_fetched = db.Column(db.Boolean, default=False)
    is_sent = db.Column(db.Boolean, default=False)

    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)

    parent_id = db.Column(db.Integer, db.ForeignKey("DataObject.id"), index=True)
    parent = relationship("DataObject", remote_side=[id])
    children = relationship(
        "DataObject", primaryjoin="and_(DataObject.parent_id == DataObject.id)"
    )

    def __repr__(self):
        return "{0} - {1}: {2}".format(self.dataset_id, self.id, self.timestamp)

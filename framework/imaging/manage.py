from .app import web_app

from .models import db, APIKey

import uuid

def create_db():
    """
    Create the Database used by the framework
    """

    db.create_all()

    
def add_api_key(name):
    """
    Add a new API Key with the given name

    name: Name of the API Key
    """

    ak = APIKey(key=str(uuid.uuid4()), name=name)
    db.session.add(ak)
    db.session.commit()

    print(ak.key)

def list_api_keys():

    for ak in APIKey.query.all():
        print(ak)
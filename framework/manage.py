from .app import web_app
from .models import db, APIKey

from sqlalchemy.exc import OperationalError
import uuid
import click

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

    return ak.key


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.pass_context
@click.option('--add', '-a', help="Name of api key to add")
@click.option('--list', '-l', is_flag=True, help="List all API keys")
def key(ctx, add, list):

    try:
        keys = APIKey.query.all()

        if add in [k.name for k in keys]:
            click.echo("Error: An API Key with that name already exists.")
            return

        if(add):
            api_key = add_api_key(add)

            click.echo('Key added: {0}'.format(api_key))

        if(list):
            for ak in APIKey.query.all():
                click.echo(ak)
    except OperationalError as oe:
        click.echo("An error occurred. Be sure to init the db with the 'initdb' command.")


@cli.command()
@click.pass_context
def initdb(ctx):

    create_db()

    click.echo('DB Created')

if __name__ == '__main__':
    cli(obj={})
    
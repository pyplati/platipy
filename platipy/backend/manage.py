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

import subprocess
from multiprocessing import Process
import uuid

from sqlalchemy.exc import OperationalError
import click
import redis

from .models import db, APIKey


def create_db():
    """
    Create the Database used by the framework
    """

    db.create_all()


def add_api_key(name, is_admin):
    """
    Add a new API Key with the given name

    name: Name of the API Key
    """

    ak = APIKey(key=str(uuid.uuid4()), name=name, is_admin=is_admin)
    db.session.add(ak)
    db.session.commit()

    return ak.key


@click.group()
@click.pass_context
def cli(_):
    pass


@cli.command()
@click.pass_context
@click.option("--add", "-a", help="Name of api key to add")
@click.option("--list", "-l", is_flag=True, help="List all API keys")
@click.option(
    "--super",
    "-s",
    is_flag=True,
    help="API Key has super user priviledges (has access to other application's data)",
)
def key(ctx, add, list, super):

    try:
        keys = APIKey.query.all()

        if add in [k.name for k in keys]:
            click.echo("Error: An API Key with that name already exists.")
            return

        if add:
            api_key = add_api_key(add, super)

            click.echo("Key added: {0}".format(api_key))

        if list:
            for ak in APIKey.query.all():
                click.echo(ak)
    except OperationalError as oe:
        click.echo("An error occurred. Be sure to init the db with the 'initdb' command.")


@cli.command()
@click.pass_context
def initdb(ctx):

    create_db()

    click.echo("DB Created")


@cli.command()
@click.pass_context
@click.option("--ip", "-ip", help="IP address of service", required=True)
def ssl(ctx, ip):

    command = [
        "openssl",
        "req",
        "-new",
        "-x509",
        "-days",
        "365",
        "-nodes",
        "-subj",
        f"/CN={ip}",
        "-addext",
        f"subjectAltName = IP:{ip}",
        "-out",
        "service.crt",
        "-keyout",
        "service.key",
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    click.echo(stdout)
    click.echo(stderr)

    click.echo("SSL Certificates Created. Service restart required.")


def run_celery():

    celery_command = [
        "celery",
        "--app=service:celery",
        "beat",
        "--loglevel=INFO",
    ]

    process = subprocess.Popen(celery_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Running celery worker & beat")
    stdout, stderr = process.communicate()

    click.echo(stdout)
    click.echo(stderr)


@cli.command()
@click.pass_context
@click.option("--port", "-p", help="Port on which to run service", default=8000)
def run(ctx, port):

    r = redis.client.Redis()
    try:
        r.ping()
    except redis.exceptions.ConnectionError:
        click.echo("redis is not running. Start with command: `redis-server --daemonize yes`")
        return

    process_celery = Process(target=run_celery)
    process_celery.start()

    service_command = [
        "gunicorn",
        "-b",
        f":{port}",
        "--timeout",
        "300",
        "--graceful-timeout",
        "60",
        "service:app",
    ]

    process = subprocess.Popen(
        f"gunicorn -b :{port} service:app",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    click.echo("Running Service (with gunicorn)")
    process.communicate()


if __name__ == "__main__":
    cli(obj={})

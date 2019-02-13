sudo apt-get update

sudo apt-get install -y python3.6-dev

sudo apt-get install -y python3-pip

sudo apt-get install -y python3-venv

sudo apt-get install -y redis-server

sudo apt-get install -y orthanc

mkdir ~/environments
cd ~/environments

python3 -m venv impit_env

source ~/environments/impit_env/bin/activate
echo ". ~/environments/impit_env/bin/activate" > ~/.profile

export PYTHONPATH="${PYTHONPATH}:/"
echo 'export PYTHONPATH="${PYTHONPATH}:/"' >> ~/.bashrc

pip install -r /impit/requirements.txt

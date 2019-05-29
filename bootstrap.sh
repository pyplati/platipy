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
echo 'export PYTHONPATH="${PYTHONPATH}:/"' >> ~/.profile

pip install -r /impit/requirements.txt


# Install Docker
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
#sudo groupadd docker
sudo usermod -aG docker vagrant

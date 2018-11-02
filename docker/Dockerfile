# FROM mono:4.6.2.16
#
# RUN curl http://dist.opensimulator.org/opensim-0.9.0.1.tar.gz -s | tar xzf -
# COPY docker/Regions.ini /opensim-0.9.0.1/bin/Regions/Regions.ini
# COPY docker/OpenSim.ini /opensim-0.9.0.1/bin/OpenSim.ini
# EXPOSE 9000
# WORKDIR /opensim-0.9.0.1/bin

# FROM stanfordnmbl/opensim-rl:latest
# FROM conda/miniconda3:latest
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# clean up and refresh apt-get index
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# install wget, a dependency of the conda install
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# download conda
ENV HOME="/root"
RUN ["/bin/bash", "-c", "wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O $HOME/miniconda.sh"]
RUN chmod 0755 $HOME/miniconda.sh
RUN ["/bin/bash", "-c", "$HOME/miniconda.sh -b -p $HOME/conda"]
ENV PATH="$HOME/conda/bin:$PATH"
RUN rm $HOME/miniconda.sh

RUN conda install python=3.6.1

RUN conda update conda && \
    conda install pytorch torchvision cuda80 -c pytorch

# RUN conda install pytorch-cpu torchvision-cpu -c pytorch

RUN conda update conda && \
    conda install -c kidzik opensim

RUN conda update conda && \
    conda install -c conda-forge lapack

RUN apt-get update && \
    apt-get install -y build-essential git && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN pip install pip --upgrade
RUN pip install git+https://github.com/lemire/fastrand.git
RUN pip install git+https://github.com/stanfordnmbl/osim-rl.git

RUN mkdir /shared
WORKDIR /root/src
COPY . /root/src
# CMD python /root/src/main.py
CMD python /root/src/main.py  -seed_pop f -save_folder /shared -pop_size 200

# RUN apt-get update && \
#     apt-get install -y python3-pip && \
#     apt-get clean autoclean && \
#     apt-get autoremove -y
# RUN pip3 install pip --upgrade
#
# RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
# RUN pip3 install torchvision

# CMD [ "mono", "./OpenSim.exe" ]

# FROM ubuntu:16.04
#
# RUN apt-get update  && \
#     apt-get upgrade -y && \
#     apt-get clean autoclean && \
#     apt-get autoremove -y
#
# RUN apt-get update && \
#     apt-get install -y python3-pip && \
#     apt-get clean autoclean && \
#     apt-get autoremove -y
# RUN pip3 install pip --upgrade
# WORKDIR /root
#
# ################################
# # Install apt-get Requirements #
# ################################
#
# RUN pip3 install --upgrade pip
#
# COPY requirements.txt /tmp/requirements.txt
# RUN pip3 install -r /tmp/requirements.txt
#
# RUN pip3 install -e .
#
# ADD patches/sitecustomize.py /usr/lib/python3.5/sitecustomize.py
#
# CMD python3 /root/src/deep_greens/envs/greenhouse_api/poll.py

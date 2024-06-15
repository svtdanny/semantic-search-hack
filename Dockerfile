FROM ubuntu:20.04

# miniconda + python3.11 installation

WORKDIR /

RUN apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y wget vim bzip2

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
	bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda update conda -y

# conda deps installation

WORKDIR /opt/app
COPY environment.yml /opt/app/
RUN conda env create -f environment.yml

# backend installation
COPY backend /opt/app/backend
COPY embed /opt/app/embed

# for opencv
RUN apt-get -y update \
&& apt-get -y upgrade \
apt-get install -y ffmpeg

CMD conda init bash && conda activate hack-back \
&& PYTHONPATH=$PYTHONPATH:/opt/app:/opt/app/embed \
fastapi dev backend/service.py
# fastapi run backend/service.py

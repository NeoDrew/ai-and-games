FROM nvidia/cuda:12.3.0-runtime-ubuntu20.04

ENV HOME="/home/hex"
ARG UID
RUN useradd -u $UID --create-home hex

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libgdbm-dev \
        libdb5.3-dev \
        libbz2-dev \
        libexpat1-dev \
        liblzma-dev \
        tk-dev \
        wget \
        curl \
        git \
        default-jre \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# adjust version if needed
ENV PYTHON_VERSION=3.11.8

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd / && rm -rf Python-${PYTHON_VERSION}* && \
    ln -s /usr/local/bin/python3.11 /usr/local/bin/python3 && \
    ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip3

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 2 && \
    update-alternatives --set python3 /usr/local/bin/python3.11

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11


# install requirements
RUN python3.11 -m pip install --upgrade pip setuptools wheel

####### Use if you have GPU and want CUDA support#######
# RUN python3.11 -m pip install tensorflow[and-cuda]==2.19.0
# RUN python3.11 -m pip install --timeout=1000 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

####### Use if you do not need CUDA support#######
RUN python3.11 -m pip install tensorflow==2.19.0
RUN python3.11 -m pip install --timeout=1000 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

RUN python3.11 -m pip install numpy==2.1.3 pandas==2.3.1 scikit-learn


WORKDIR /home/hex

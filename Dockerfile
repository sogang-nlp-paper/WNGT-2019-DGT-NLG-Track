FROM nvidia/cuda:8.0-devel-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
ENV CUDNN_VERSION 5.1.10
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn5=$CUDNN_VERSION-1+cuda8.0 \
            libcudnn5-dev=$CUDNN_VERSION-1+cuda8.0 && \
    rm -rf /var/lib/apt/lists/*

# from kaixhin/torch/dockerfile
RUN apt-get update && apt-get install -y \
		git \
		software-properties-common \
		libssl-dev \
		libzmq3-dev \
		python-dev \
		python-pip \
		python-zmq \
		sudo
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
 	bash install-deps
## older version of gcc is needed for torch...
RUN sudo add-apt-repository ppa:ubuntu-toolchain-r/test && \
	sudo apt-get update && \
	sudo apt-get install -y gcc-4.9 g++-4.9 && \
	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9
## gcc and gfortran has to be in same version...
RUN sudo apt-get install -y gfortran-4.9 && \
    sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-4.9 60
## hack for installing torch
ENV TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
WORKDIR /root/torch
RUN ./install.sh
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# added by hwijeen
RUN apt-get install -y \
		vim \
		gcc-4.8-plugin-de \
		wget
## install lua
RUN mkdir -p /home/Downloads
WORKDIR /home/Downloads
RUN	curl -R -O http://www.lua.org/ftp/lua-5.3.5.tar.gz && \
	tar zxf lua-5.3.5.tar.gz && \
	cd lua-5.3.5 && \
	make linux test && \
	sudo make install
## install hdf5
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.20/src/hdf5-1.8.20.tar.gz && \
	tar -zxvf hdf5-1.8.20.tar.gz && \
	cd hdf5-1.8.20 && \
	make install
ENV PATH=/home/Downloads/hdf5-1.8.20/hdf5:$PATH
RUN luarocks install hdf5 20-0
## source code
RUN mkdir -p /home/WNGT2019
WORKDIR /home/WNGT2019
COPY . ./

FROM xilinx/xilinx_runtime_base:alveo-2023.2-ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# install basic dependencies and set locale (UTF-8 is required by some Vitis components)
RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  wget \
  git \
  sudo \
  locales \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# 2. enable i386 architecture for 32-bit dependencies
RUN dpkg --add-architecture i386 && apt-get update

# 3. install Vitis dependencies
RUN apt-get install -y --no-install-recommends \
  gparted \
  xinetd \
  gawk \
  gcc \
  net-tools \
  libncurses-dev \
  openssl \
  libssl-dev \
  flex \
  bison \
  autoconf \
  libtool \
  texinfo \
  zlib1g-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
  iproute2 \
  make \
  libncurses5-dev \
  libselinux1 \
  wget \
  diffstat \
  chrpath \
  socat \
  tar \
  unzip \
  gzip \
  python3 \
  tofrodos \
  lsb-release \
  libftdi1 \
  libftdi1-2

RUN apt-get update && apt-get install -y --no-install-recommends \
  lib32stdc++6 \
  libgtk2.0-0:i386 \
  libfontconfig1:i386 \
  libx11-6:i386 \
  libxext6:i386 \
  libxrender1:i386 \
  libsm6:i386 \
  openssh-client

RUN apt-get update && apt-get install -y --no-install-recommends \
  debianutils \
  iputils-ping \
  libegl1-mesa \
  libsdl1.2-dev \
  python3 \
  cpio \
  gnupg \
  zlib1g:i386 \
  perl \
  xvfb

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc-multilib \
  build-essential \
  cmake \
  automake \
  g++ \
  python3-pip \
  xz-utils \
  python3-git \
  python3-jinja2 \
  python3-pexpect

RUN apt-get update && apt-get install -y --no-install-recommends \
  liberror-perl \
  xtrans-dev \
  libxcb-randr0-dev \
  libxcb-xtest0-dev \
  libxcb-xinerama0-dev \
  libxcb-shape0-dev \
  libxcb-xkb-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
  util-linux \
  sysvinit-utils \
  ocl-icd-libopencl1 \
  opencl-headers \
  ocl-icd-opencl-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
  libncurses5 \
  libncurses5-dev \
  libncursesw5 \
  libncursesw5-dev \
  libncurses5:i386 \
  libtinfo5 \
  libstdc++6:i386 \
  libgtk2.0-0:i386 \
  && rm -rf /var/lib/apt/lists/*

# 4. set bash as default shell
RUN echo "dash dash/sh boolean false" | debconf-set-selections \
  && dpkg-reconfigure -f noninteractive dash

# 5. allow passwordless sudo for all users
RUN echo "ALL ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

CMD ["/bin/bash"]

# syntax=docker/dockerfile:1

# base image
FROM ubuntu:22.04 as base
SHELL ["/bin/bash", "-c"]

# username, uid, gid
ARG USER=user
ARG UID=1000
ARG GID=1000
ARG OBELISK_ROOT=/
ENV USER=$USER
ENV UID=$UID
ENV GID=$GID
ENV OBELISK_ROOT=$OBELISK_ROOT

# set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# basic dependencies from docker_setup.sh (up until sudo and below)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    clang-tools-12 \
    nano \
    vim \
    git \
    libeigen3-dev \
    x11-apps \
    locales \
    python3.10 \
    python3.10-venv \
    python3-pip \
    sudo && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8

# create non-root user with sudo privileges for certain commands
RUN groupadd --gid $GID $USER && \
    useradd --uid $UID --gid $GID -m $USER -d /home/${USER} --shell /usr/bin/bash && \
    echo "${USER}:password" | chpasswd && \
    usermod -aG sudo ${USER} && \
    echo "%${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# switch to new user and workdir
USER ${UID}

RUN echo "About to pull obelisk."

# clone Obelisk into this docker image at the $OBELISK_ROOT path
WORKDIR /home/${USER}
RUN git clone -b unitree-interface --single-branch https://github.com/Caltech-AMBER/obelisk.git /home/${USER}/obelisk
ENV OBELISK_ROOT=/home/${USER}/obelisk

# you must run the setup script from a directory where the user has permissions
# run docker setup script in Dockerfile
WORKDIR /home/${USER}
RUN source /home/${USER}/obelisk/setup.sh --install-sys-deps --cyclone-perf --obk-aliases --basic
RUN sudo apt update && sudo apt install ros-humble-grid-map-msgs

# Set up python environment
WORKDIR /home/${USER}/repos/go2-navigation
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

ENV ROS_DOMAIN_ID=5
ENV OBELISK_BUILD_UNITREE=true
RUN echo 'alias launch_go2_rl_vel="obk-launch config_file_path=${GO2_NAVIGATION_ROOT}/install/go2_rl_vel_tracking/share/go2_rl_vel_tracking/config/rl_vel_tracking.yml device_name=onboard"' >> ~/.bashrc
RUN echo 'alias launch_go2_dtmpc="obk-launch config_file_path=${GO2_NAVIGATION_ROOT}/install/go2_dyn_tube_mpc/share/go2_dyn_tube_mpc/config/dynamic_tube_mpc.yml device_name=onboard"' >> ~/.bashrc
RUN echo 'alias launch_go2_high_level_test="obk-launch config_file_path=${GO2_NAVIGATION_ROOT}/install/go2_dyn_tube_mpc/share/go2_dyn_tube_mpc/config/high_level_test.yml device_name=onboard"' >> ~/.bashrc

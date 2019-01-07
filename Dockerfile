FROM ubuntu:16.04
MAINTAINER Michael McDonald <m_j_mcdonald@berkeley.edu>
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install build-essential cmake gcc g++ git ipython minizip python-dev python-h5py python-numpy python-scipy python-sympy qt4-dev-tools libassimp-dev libavcodec-dev libavformat-dev libavformat-dev libboost-all-dev libboost-date-time-dev libbullet-dev libfaac-dev libglew-dev libgsm1-dev liblapack-dev liblog4cxx-dev libmpfr-dev libode-dev libogg-dev libpcrecpp0v5 libpcre3-dev libqhull-dev libqt4-dev libsoqt-dev-common libsoqt4-dev libswscale-dev libswscale-dev libvorbis-dev libx264-dev libxml2-dev libxvidcore-dev python-pip libglfw3 libgl1-mesa-glx libosmesa6 python-matplotlib software-properties-common libcairo2-dev libjasper-dev libpoppler-glib-dev libsdl2-dev libtiff5-dev libxrandr-dev libccd-dev -y

RUN pip install --upgrade --user sympy==0.7.1
RUN apt-get remove python-mpmath
RUN git clone https://github.com/rdiankov/collada-dom.git
WORKDIR /collada-dom
RUN mkdir build
WORKDIR /collada-dom/build
RUN cmake ..
RUN make -j4
RUN make install
WORKDIR /

# RUN echo 'deb http://security.ubuntu.com/ubuntu xenial-security main' >> /etc/apt/sources.list
# RUN apt-get update
# RUN git clone --branch OpenSceneGraph-3.4 https://github.com/openscenegraph/OpenSceneGraph.git
# WORKDIR OpenSceneGraph 
# RUN mkdir build 
# WORKDIR /OpenSceneGraph/build
# RUN cmake .. -DDESIRED_QT_VERSION=4
# RUN make install_ld_conf
# RUN make -j4
# RUN make install
# WORKDIR /

RUN ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen
RUN git clone https://github.com/flexible-collision-library/fcl.git
WORKDIR /fcl
RUN git checkout 0.5.0
RUN mkdir build 
WORKDIR /fcl/build
RUN cmake ..
RUN make -j4
RUN make install
WORKDIR /

RUN git clone --branch latest_stable https://github.com/rdiankov/openrave.git
WORKDIR /openrave
RUN git checkout 9c79ea260e1c009b0a6f7c03ec34f59629ccbe2c
# RUn git checkout master
RUN mkdir build 
WORKDIR /openrave/build
# RUN cmake .. -DOSG_DIR=/usr/local/lib64/
RUN cmake -OPT_QTOSG_VIEWER=OFF -OPT_OCTAVE=OFF -OPT_MATLAB=OFF -OPT_EXTRA_ROBOTS=OFF ..
RUN make -j4
RUN make install

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(openrave-config --python-dir)/openravepy/_openravepy_
RUN export PYTHONPATH=$PYTHONPATH:$(openrave-config --python-dir)

WORKDIR /
RUN apt-get install wget -y
RUN mkdir -p /root/.mujoco && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && unzip mujoco.zip -d /root/.mujoco && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
RUN pip install git+git://github.com/deepmind/dm_control.git

RUN pip install --upgrade pip
RUN pip install gym --ignore-installed

RUN git clone https://github.com/m-j-mcdonald/BaxterGym.git
WORKDIR /BaxterGym
RUN pip install -e .
WORKDIR /

RUN mkdir -p /root/.openrave
RUN cp -r BaxterGym/baxter_gym/robot_info/baxter/iksolver/kinematics.a9244a345a56872b76cadb57277d0618 /root/.openrave/

RUN echo "if xrandr -q | grep ' connected' ; then echo 'Found monitor' ; export MUJOCO_GL=glfw ; else echo 'No monitor connected' ; export MUJOCO_GL=osmesa ; fi" >> /root/.bashrc

RUN apt-get install vim -y

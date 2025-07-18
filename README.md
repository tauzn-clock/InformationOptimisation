<div align="center">
  
<h2> Plane detection and ranking via model information optimisation </h2>
<p>Daoxin Zhong, Jun Li, Michael Chuah</p>

</div>

TLDR: Given an depth image with known intrinsics, find an ordered list of planes that represents the most likely model by minimising model information.

---
### Theoretical Overview
---
### Docker

Build the docker image using the following command:

```
docker build 
    --ssh default=$SSH_AUTH_SOCK 
    -t info_opt .
```

Run the docker image using the following command:

```
docker run  
    -it 
    -v <path_of_scratchdata>:/scratchdata 
    --gpus all 
    --shm-size 16g  
    -d  
    --network=host  
    --restart unless-stopped  
    --env="DISPLAY"  
    --env="QT_X11_NO_MITSHM=1"  
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"   
    --device=/dev/ttyUSB0  
    -e DISPLAY=unix$DISPLAY  
    --privileged 
    info_opt
```
---
### Build

#### Python

```
git clone https://github.com/tauzn-clock/InformationOptimisation
pip3 install -r requirements.txt
```

#### C++

```
git clone https://github.com/tauzn-clock/InformationOptimisation
cd InformationOptimisation/cpp
chmod +x ./requirements.sh
./requirements.sh
mkdir build && cd build
cmake ..
make
```

The C++ implementation does not allow for image segmentation via SAM, so planes are found from the full depth image directly.

---

### Run

The bulk of the code is written in Python.

`python3 demo.py`
`python3 video.py`

C++ implementation is available in the `cpp` directory.

---

### Citation

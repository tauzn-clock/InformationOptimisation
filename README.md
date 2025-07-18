<div align="center">
  
<h2> Plane detection and ranking via model information optimisation </h2>
<p>Daoxin Zhong, Jun Li, Michael Chuah</p>

</div>

TLDR: Given an depth image with known intrinsics, find an ordered list of planes that represents the most likely model by minimising model information.

---
### Data Format

```
root_dir/
├── rgb/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── depth/
│   ├── 0.png
│   ├── 1.png
│   └── ...
```

Get the NYU dataset and all predicted planes from [here](https://drive.google.com/file/d/11PlNTvWpEvgwYDm7KMCCEbTTzira6-wV/view?usp=drive_link).

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

#### Python

From `python` directory, run:

```
python3 demo.py ./nyu.yaml
```
The noise function can be changed at line 53 of `python/demo.py`.
#### C++

From `cpp/build` directory, run:

```
./main ../src/nyu.yaml
```
The noise function can be changed at line 93 of `cpp/src/information_optimisation.cpp`.

YAML Parameters:
`file_path`: Path to data
`img_count`: Number of images to process.

`camera_params`: Camera parameters, includes focal lengths (fx, fy), principal point (cx, cy).
`depth_max` : Maximum depth value (in meters).
`resolution` : Depth image resolution (in meters)

`conf`: Confidence level for plane fitting.
`inlier_th`: Assumed Inlier threshold for plane fitting.
`max_plane`: Maximum number of models to test.

`use_sam`: Use Segment Anything Model (SAM) for plane segmentation.
`sam_conf`: Confidence level for plane fitting in each SAM region.
`sam_inlier_th`: Assumed Inlier threshold for plane fitting in each SAM region.
`sam_max_plane`: Maximum number of models to test in each SAM region.

---

### Citation

# camera-intrinsic-model
[![crate](https://img.shields.io/crates/v/camera-intrinsic-model.svg)](https://crates.io/crates/camera-intrinsic-model)

A pure rust camera intrinsic model library. Including
* project / unproject points
* undistort and remap image

### Supported camera models are
* Extended Unified (EUCM)
* Extended Unified with Tangential (EUCMT)
* Unified Camera Model (UCM)
* Kannala Brandt (KB4) aka OpenCV Fisheye
* OpenCV (OPENCV5) aka `plumb_bob` in ROS
* F-theta (FTHETA) by NVidia

For calibration to get the precise parameters. Please use [camera-intrinsic-calibration](https://github.com/powei-lin/camera-intrinsic-calibration-rs)

## Examples
```sh
# undistort and remap
cargo run -r --example remap

# undistort and rectify
cargo run -r --example stereo_rectify
```

## Acknowledgements
Links:
* https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
* https://gitlab.com/VladyslavUsenko/basalt-headers
* https://github.com/itt-ustutt/num-dual

Papers:

* Usenko, Vladyslav, Nikolaus Demmel, and Daniel Cremers. "The double sphere camera model." 2018 International Conference on 3D Vision (3DV). IEEE, 2018.

## TODO
* [x] Stereo Rectify
* [x] FTheta Model
* [ ] Python bindings
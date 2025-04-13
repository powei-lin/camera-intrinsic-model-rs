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
* Fov Camera (FOV_CAMERA)

For calibration to get the precise parameters. Please use [camera-intrinsic-calibration](https://github.com/powei-lin/camera-intrinsic-calibration-rs)

## Examples
```sh
# undistort and remap
cargo run -r --example remap

# undistort and rectify
cargo run -r --example stereo_rectify
```

## Benchmark
Remapping to 1024x1024 on m4 mac mini.
```
╭───────────────────────────────────────────────────────────────────────╮
│                                 remap                                 │
├────────────────┬──────┬───────────┬───────────┬───────────┬───────────┤
│ benchmark      │ args │   fastest │    median │      mean │    stddev │
├────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
│ mono8 normal   │ None │ 732.17 µs │ 827.00 µs │ 858.60 µs │  94.88 µs │
│ mono8 fast     │ None │ 272.00 µs │ 342.25 µs │ 360.68 µs │ 228.13 µs │
│ rgb8 normal    │ None │   1.87 ms │   2.00 ms │   2.04 ms │ 143.60 µs │
│ rgb8 fast      │ None │ 751.67 µs │ 824.54 µs │ 851.30 µs │  79.45 µs │
╰────────────────┴──────┴───────────┴───────────┴───────────┴───────────╯
```

## Acknowledgements
Links:
* https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
* https://gitlab.com/VladyslavUsenko/basalt-headers

Papers:

* Usenko, Vladyslav, Nikolaus Demmel, and Daniel Cremers. "The double sphere camera model." 2018 International Conference on 3D Vision (3DV). IEEE, 2018.
* Frédréric, Devernay, and Faugeras Olivier. "Straight lines have to be straight: Automatic calibration and removal of distortion from scenes of structured enviroments." Mach. Vision Appl 13.1 (2001): 14-24.

## TODO
* [x] Stereo Rectify
* [x] FTheta Model
* [ ] Python bindings
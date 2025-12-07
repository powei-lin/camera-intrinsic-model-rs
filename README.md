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
Remapping to 1024x1024 on m4 mac mini (26.1).
```
╭───────────────────────────────────────────────────────────────────────╮
│                                 remap                                 │
├────────────────┬──────┬───────────┬───────────┬───────────┬───────────┤
│ benchmark      │ args │   fastest │    median │      mean │    stddev │
├────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
│ mono8 normal   │ None │ 883.92 µs │ 918.38 µs │ 945.79 µs │  66.20 µs │
│ mono8 fast     │ None │ 329.83 µs │ 352.50 µs │ 359.58 µs │  22.43 µs │
│ rgb8 normal    │ None │   1.64 ms │   1.70 ms │   1.71 ms │  84.11 µs │
│ rgb8 fast      │ None │ 481.75 µs │ 508.62 µs │ 520.94 µs │  39.45 µs │
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
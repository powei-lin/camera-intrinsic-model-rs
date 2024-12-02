# camera-intrinsic-model-rs
[![crate](https://img.shields.io/crates/v/camera-intrinsic-model.svg)](https://crates.io/crates/camera-intrinsic-model)

A pure rust camera intrinsic model library. Including
* project / unproject points
* undistort and remap image

## Examples
```sh
# undistort and remap
cargo run -r --example remap
```

## Acknowledgements
Links:
* https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
* https://gitlab.com/VladyslavUsenko/basalt-headers
* https://github.com/itt-ustutt/num-dual

Papers:

* Usenko, Vladyslav, Nikolaus Demmel, and Daniel Cremers. "The double sphere camera model." 2018 International Conference on 3D Vision (3DV). IEEE, 2018.

## TODO
* [ ] Stereo Rectify
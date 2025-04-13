//! camera-intrinsic-model is a library for distort/undistort images.
//! # Examples
//!
//! ```
//! use camera_intrinsic_model::*;
//! let model = model_from_json("data/eucm0.json");
//! let new_w_h = 1024;
//! let p = model.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
//! let (xmap, ymap) = model.init_undistort_map(&p, (new_w_h, new_w_h), None);
//! // let remaped = remap(&img, &xmap, &ymap);
//! ```
pub mod generic_functions;
pub mod generic_model;
pub mod io;
pub mod models;

pub use generic_functions::*;
pub use generic_model::*;
pub use io::*;
pub use models::*;

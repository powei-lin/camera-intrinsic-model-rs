use super::generic_model::*;

use nalgebra as na;
use rayon::prelude::*;

/// Returns xmap and ymap for remaping
///
/// # Arguments
///
/// * `camera_model` - any camera model
/// * `projection_mat` - new camera matrix for the undistorted image
/// * `new_w_h` - new image width and height for the undistorted image
/// * `rotation` - Optional rotation, normally for stereo rectify
///
/// # Examples
///
/// ```
/// use camera_intrinsic_model::*;
/// let model = model_from_json("eucm.json");
/// let new_w_h = 1024;
/// let p = model.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
/// let (xmap, ymap) = model.init_undistort_map(&p, (new_w_h, new_w_h), None);
/// // let remaped = remap(&img, &xmap, &ymap);
/// ```
pub fn init_undistort_map(
    camera_model: &dyn CameraModel<f64>,
    projection_mat: &na::Matrix3<f64>,
    new_w_h: (u32, u32),
    rotation: Option<na::Rotation3<f64>>,
) -> (na::DMatrix<f32>, na::DMatrix<f32>) {
    if projection_mat.shape() != (3, 3) {
        panic!("projection matrix has the wrong shape");
    }
    let fx = projection_mat[(0, 0)];
    let fy = projection_mat[(1, 1)];
    let cx = projection_mat[(0, 2)];
    let cy = projection_mat[(1, 2)];
    let rmat_inv = rotation
        .unwrap_or(na::Rotation3::identity())
        .inverse()
        .matrix()
        .to_owned();
    let p3ds: Vec<na::Vector3<f64>> = (0..new_w_h.1)
        .into_par_iter()
        .flat_map(|y| {
            (0..new_w_h.0)
                .into_par_iter()
                .map(|x| {
                    rmat_inv * na::Vector3::new((x as f64 - cx) / fx, (y as f64 - cy) / fy, 1.0)
                })
                .collect::<Vec<na::Vector3<f64>>>()
        })
        .collect();
    let p2ds = camera_model.project(&p3ds);
    let (xvec, yvec): (Vec<f32>, Vec<f32>) = p2ds
        .par_iter()
        .map(|xy| {
            if let Some(xy) = xy {
                (xy[0] as f32, xy[1] as f32)
            } else {
                (f32::NAN, f32::NAN)
            }
        })
        .unzip();
    let xmap = na::DMatrix::from_vec(new_w_h.1 as usize, new_w_h.0 as usize, xvec);
    let ymap = na::DMatrix::from_vec(new_w_h.1 as usize, new_w_h.0 as usize, yvec);
    (xmap, ymap)
}

/// Returns xmap and ymap for remaping
///
/// # Arguments
///
/// * `camera_model` - any camera model
/// * `balance` - [0-1] zero means no black margin
/// * `new_image_w_h` - optional new image width and height, default using the original w, h
///
/// # Examples
///
/// ```
/// use camera_intrinsic_model::*;
/// let model = model_from_json("eucm.json");
/// let new_w_h = 1024;
/// let p = model.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
/// let (xmap, ymap) = model.init_undistort_map(&p, (new_w_h, new_w_h), None);
/// ```
pub fn estimate_new_camera_matrix_for_undistort(
    camera_model: &dyn CameraModel<f64>,
    balance: f64,
    new_image_w_h: Option<(u32, u32)>,
) -> na::Matrix3<f64> {
    if !(0.0..=1.0).contains(&balance) {
        panic!("balance should be [0.0-1.0], got {}", balance);
    }
    let params = camera_model.params();
    let cx = params[2];
    let cy = params[3];
    let w = camera_model.width();
    let h = camera_model.height();
    let p2ds = vec![
        na::Vector2::new(cx, 0.0),
        na::Vector2::new(w - 1.0, cy),
        na::Vector2::new(cx, h - 1.0),
        na::Vector2::new(0.0, cy),
    ];
    let undist_pts = camera_model.unproject(&p2ds);
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for p in undist_pts {
        let p = p.unwrap();
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    min_x = min_x.abs();
    min_y = min_y.abs();
    let (new_w, new_h) = if let Some((new_w, new_h)) = new_image_w_h {
        (new_w, new_h)
    } else {
        (camera_model.width() as u32, camera_model.height() as u32)
    };
    let new_cx = new_w as f64 * min_x / (min_x + max_x);
    let new_cy = new_h as f64 * min_y / (min_y + max_y);
    let fx = new_w as f64 / (min_x + max_x);
    let fy = new_h as f64 / (min_y + max_y);
    let fmin = fx.min(fy);
    let fmax = fx.max(fy);
    let new_f = balance * fmin + (1.0 - balance) * fmax;

    let mut out = na::Matrix3::identity();
    out[(0, 0)] = new_f;
    out[(1, 1)] = new_f;
    out[(0, 2)] = new_cx;
    out[(1, 2)] = new_cy;
    out
}

/// Returns xmap and ymap for remaping
///
/// # Arguments
///
/// * `camera_model` - any camera model
/// * `balance` - [0-1] zero means no black margin
/// * `new_image_w_h` - optional new image width and height, default using the original w, h
///
/// # Examples
///
/// ```
/// use camera_intrinsic_model::*;
/// use nalgebra as na;
/// let model0 = model_from_json("data/eucm0.json");
/// let model1 = model_from_json("data/eucm1.json");
/// let tvec = na::Vector3::new(
///     -0.10098947190325333,
///     -0.0020811599784744455,
///     -0.0012888359197775197,
/// );
/// let quat = na::Quaternion::new(
///     0.9997158799903332,
///     0.02382966001551074,
///     -0.00032454324393309654,
///     0.00044863167728445325,
/// );
/// let rvec = na::UnitQuaternion::from_quaternion(quat).scaled_axis();
/// let (r0, r1, p) = stereo_rectify(&model0, &model1, &rvec, &tvec, None);
/// let image_w_h = (
///     model0.width().round() as u32,
///     model0.height().round() as u32,
/// );
/// let (xmap0, ymap0) = model0.init_undistort_map(&p, image_w_h, Some(r0));
/// let (xmap1, ymap1) = model1.init_undistort_map(&p, image_w_h, Some(r1));
/// ```
pub fn stereo_rectify(
    camera_model0: &GenericModel<f64>,
    camera_model1: &GenericModel<f64>,
    rvec: &na::Vector3<f64>,
    tvec: &na::Vector3<f64>,
    new_image_w_h: Option<(u32, u32)>,
) -> (na::Rotation3<f64>, na::Rotation3<f64>, na::Matrix3<f64>) {
    // compensate for rotation first
    let r_half_inv = -rvec / 2.0;

    // another rotation for compensating the translation
    // use translation to determine x axis rectify or y axis rectify
    let rotated_t = na::Rotation3::from_scaled_axis(r_half_inv) * tvec;
    let idx = if rotated_t.x.abs() > rotated_t.y.abs() {
        0
    } else {
        1
    };
    let axis_to_rectify = rotated_t[idx];
    let translation_norm = rotated_t.norm();
    let mut unit_vector = na::Vector3::zeros();
    if axis_to_rectify > 0.0 {
        unit_vector[idx] = 1.0;
    } else {
        unit_vector[idx] = -1.0;
    }

    let mut axis_for_rotation = rotated_t.cross(&unit_vector);
    let axis_norm = axis_for_rotation.norm();
    let mut rotation_for_compensating_translation = na::Rotation3::identity();
    if axis_norm > 0.0 {
        axis_for_rotation /= axis_norm;
        axis_for_rotation *= (axis_to_rectify.abs() / translation_norm).acos();
        rotation_for_compensating_translation = na::Rotation3::from_scaled_axis(axis_for_rotation);
    }

    let rotation_for_cam0 =
        rotation_for_compensating_translation * na::Rotation3::from_scaled_axis(-r_half_inv);
    let rotation_for_cam1 =
        rotation_for_compensating_translation * na::Rotation3::from_scaled_axis(r_half_inv);

    // new projection matrix
    let new_cam_mat0 = camera_model0.estimate_new_camera_matrix_for_undistort(0.0, new_image_w_h);
    let new_cam_mat1 = camera_model1.estimate_new_camera_matrix_for_undistort(0.0, new_image_w_h);
    let mut avg_new_mat = na::Matrix3::identity();
    let avg_f = (new_cam_mat0[(0, 0)] + new_cam_mat1[(0, 0)]) / 2.0;
    avg_new_mat[(0, 0)] = avg_f;
    avg_new_mat[(1, 1)] = avg_f;
    let avg_cx = (new_cam_mat0[(0, 2)] + new_cam_mat1[(0, 2)]) / 2.0;
    avg_new_mat[(0, 2)] = avg_cx;
    let avg_cy = (new_cam_mat0[(1, 2)] + new_cam_mat1[(1, 2)]) / 2.0;
    avg_new_mat[(1, 2)] = avg_cy;
    (rotation_for_cam0, rotation_for_cam1, avg_new_mat)
}
